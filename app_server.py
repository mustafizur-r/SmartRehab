"""
app_server.py  v6 — SmartRehab
================================
Unified FastAPI server.

TWO REFINEMENT MODES unified into one /refine_motion/ endpoint:
  1. CLINICAL GAIT  — user describes a gait syndrome in natural language
     "right foot drags on the floor", "Parkinson's with freezing", etc.
     → parsed → impairment_state dict → apply_impairment()

  2. CUSTOM JOINT OFFSET — user describes a body-part adjustment
     "patient's hands on their chest", "left knee 20 degrees more bent", etc.
     → parsed → custom_offsets list → apply_custom_offset()

ADDITIVE SESSION MODEL:
  - Base BVH is generated once by /gen_text2motion/ and backed up.
  - Every /refine_motion/ call ALWAYS starts from the clean base BVH.
  - impairment_state and custom_offsets are BOTH stored in the session.
  - Each call merges NEW params into the existing session state.
  - apply_all(base, output, impairment_state, custom_offsets) applies both layers.
  - Calling /refine_motion/ again with new params: previous state is preserved,
    new params are merged on top — fully additive.
"""

import os
import re
import uuid
import math
import shutil
import platform
import subprocess
import unicodedata

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from starlette.concurrency import run_in_threadpool

from langdetect import detect, DetectorFactory
from dotenv import load_dotenv
from openai import OpenAI

from questionnaire.router import router as questionnaire_router
from questionnaire.store_dual import init_dual_store

# Unified engine — impairments + custom offsets in one module
from bvh_impairment_engine import (
    apply_all,
    apply_impairment,
    parse_custom_offset,
    merge_offsets,
    CUSTOM_OFFSET_PROMPT,
)

DetectorFactory.seed = 0
load_dotenv()


# =============================================================================
# SECTION 1 — Logging
# =============================================================================

def _debug(msg: str) -> None:
    print(msg, flush=True)


# =============================================================================
# SECTION 2 — Settings
# =============================================================================

class Settings(BaseModel):
    openai_api_key:         str = Field("",              alias="OPENAI_API_KEY")
    openai_model_translate: str = Field("gpt-4o-mini",   alias="OPENAI_MODEL_TRANSLATE")
    openai_model_rewrite:   str = Field("gpt-4o-mini",   alias="OPENAI_MODEL_REWRITE")
    openai_base_url:        str = Field("",              alias="OPENAI_BASE_URL")
    openai_timeout_seconds: int = Field(60,              alias="OPENAI_TIMEOUT_SECONDS")
    ollama_model_translate: str = Field("llama3",        alias="OLLAMA_MODEL_TRANSLATE")
    ollama_model_rewrite:   str = Field("llama3",        alias="OLLAMA_MODEL_REWRITE")
    google_credentials:     str = Field("",              alias="GOOGLE_APPLICATION_CREDENTIALS")
    vertex_project_id:      str = Field("",              alias="VERTEX_PROJECT_ID")
    vertex_location:        str = Field("us-central1",   alias="VERTEX_LOCATION")
    vertex_model:           str = Field("gemini-2.5-pro-preview-06-05", alias="VERTEX_MODEL")


settings = Settings.model_validate(os.environ)

_USE_OPENAI = bool(settings.openai_api_key and settings.openai_api_key.strip())
_USE_VERTEX = bool(settings.google_credentials and settings.vertex_project_id)

if _USE_VERTEX:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.google_credentials
    _debug(f"[Startup] Vertex AI enabled. project={settings.vertex_project_id} model={settings.vertex_model}")
else:
    _debug("[Startup] Vertex AI not configured.")

client = None
if _USE_OPENAI:
    client = OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url or None,
        timeout=settings.openai_timeout_seconds,
    )
    _debug(f"[Startup] OpenAI enabled. base_url={settings.openai_base_url or '(default)'}")
else:
    _debug("[Startup] OpenAI not configured — using Ollama fallback.")


# =============================================================================
# SECTION 3 — LLM Callers
# =============================================================================

def _vertex_text(prompt: str, purpose: str) -> str:
    try:
        import vertexai
        from vertexai.generative_models import GenerativeModel, GenerationConfig
    except ImportError:
        raise RuntimeError("google-cloud-aiplatform not installed.")

    _debug(f"[LLM] provider=VertexAI purpose={purpose}")
    vertexai.init(project=settings.vertex_project_id, location=settings.vertex_location)
    model  = GenerativeModel(settings.vertex_model)
    config = GenerationConfig(temperature=0.2, max_output_tokens=2048, top_p=0.9)
    out    = (model.generate_content(prompt, generation_config=config).text or "").strip()
    out    = re.sub(r"^```(?:text)?\s*", "", out, flags=re.IGNORECASE)
    out    = re.sub(r"\s*```$", "", out).strip()
    _debug(f"[VertexAI] success — {len(out.split())} words")
    return out


def _openai_text(prompt: str, model: str, max_output_tokens: int, purpose: str) -> str:
    if client is None:
        raise RuntimeError("OpenAI client not configured.")
    _debug(f"[LLM] provider=OpenAI purpose={purpose} model={model}")
    system_msg = "You follow instructions exactly and return plain text only."

    def _norm(s):
        return re.sub(r"\s{2,}", " ", (s or "").replace("\r", " ").strip()).strip(" \"\u2018\u2019\u201c\u201d`'")

    def _strip_fences(s):
        s = re.sub(r"^```(?:text)?\s*", "", (s or "").strip(), flags=re.IGNORECASE)
        return re.sub(r"\s*```$", "", s).strip()

    def _salvage(s):
        chunks = [c.strip() for c in re.split(r"\n\s*\n", (s or "").strip()) if c.strip()]
        cands  = [c for c in chunks if re.search(r"[.!?]", c)]
        return cands[-1] if cands else (chunks[-1] if chunks else s)

    def _is_meta(s):
        low = (s or "").lower()
        return any(t in low for t in ["okay, let's", "the user wants", "first, i need",
                                       "analysis", "thinking", "reasoning"])

    try:
        if hasattr(client, "responses"):
            resp = client.responses.create(
                model=model,
                input=[{"role": "system", "content": system_msg},
                       {"role": "user",   "content": prompt}],
                temperature=0.2,
                max_output_tokens=int(max_output_tokens),
            )
            out = _norm(_strip_fences(getattr(resp, "output_text", "") or ""))
            if out:
                return _norm(_strip_fences(_salvage(out))) if _is_meta(out) else out
    except Exception as e:
        _debug(f"[LLM] Responses API failed: {e}")

    if hasattr(client, "chat") and hasattr(client.chat, "completions"):
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system_msg},
                      {"role": "user",   "content": prompt}],
            temperature=0.2,
            max_tokens=int(max_output_tokens),
        )
        out = _norm(_strip_fences((resp.choices[0].message.content or "").strip()))
        return _norm(_strip_fences(_salvage(out))) if _is_meta(out) else out

    raise RuntimeError("OpenAI SDK has no usable API.")


def _pick_ollama_model(preferred: str) -> str:
    try:
        import ollama
        resp   = ollama.list()
        models = resp.get("models", []) if isinstance(resp, dict) else getattr(resp, "models", []) or []
        have   = [(m.get("model","") if isinstance(m,dict) else getattr(m,"model","")) for m in models]
        have   = [x for x in have if x]
        qwen3  = sorted([m for m in have if m.lower().startswith("qwen3")], reverse=True)
        chosen = qwen3[0] if qwen3 else (preferred or (have[0] if have else "llama3"))
        _debug(f"[Ollama] chosen_model={chosen}")
        return chosen
    except Exception as e:
        _debug(f"[Ollama] list failed: {e}")
        return preferred or "llama3"


def _ollama_text(prompt: str, model: str, num_predict: int, purpose: str) -> str:
    import json, ollama
    _debug(f"[LLM] provider=Ollama purpose={purpose} model={model}")

    def _norm(s):
        return re.sub(r"\s{2,}", " ", (s or "").replace("\r"," ").replace("\n"," ")
                      ).strip(" \"\u201c\u201d\u2018\u2019")

    def _strip_fences(s):
        s = re.sub(r"^```(?:json)?\s*", "", (s or "").strip(), flags=re.IGNORECASE)
        return re.sub(r"\s*```$", "", s).strip()

    def _best(s):
        chunks = [c.strip() for c in re.split(r"\n\s*\n", (s or "").strip()) if c.strip()]
        cands  = [c for c in chunks if re.search(r"[.!?]", c)]
        return cands[-1] if cands else (chunks[-1] if chunks else s)

    def _is_reasoning(s):
        low = (s or "").lower()
        return any(t in low for t in ["okay, let's","the user wants","first, i need",
                                       "analysis","thinking","reasoning","step by step"])

    def _call(sys_msg, use_json):
        kw   = {"format":"json"} if use_json else {}
        resp = ollama.chat(
            model=model,
            messages=[{"role":"system","content":sys_msg},
                      {"role":"user","content":prompt}],
            options={"temperature":0.0,"top_p":0.9,"num_ctx":4096,"num_predict":int(num_predict)},
            **kw,
        )
        msg = getattr(resp, "message", None)
        return (getattr(msg,"content","") or "").strip() if msg else ""

    try:
        raw  = _strip_fences(_call(
            "Return ONLY a JSON object with key 'text' containing the final plain-English paragraph.", True))
        data = {}
        try: data = json.loads(raw)
        except Exception:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                try: data = json.loads(m.group(0))
                except: pass
        out = _norm(data.get("text","") if isinstance(data,dict) else "")
        if out:
            return _best(out) if _is_reasoning(out) else out
    except Exception as e:
        _debug(f"[Ollama] json mode failed: {e}")

    try:
        out = _norm(_strip_fences(_call(
            "Return ONLY the final paragraph as plain text. Never return empty.", False)))
        return _best(out) if _is_reasoning(out) else out
    except Exception as e:
        _debug(f"[Ollama] chat failed: {e}")
        return ""


def _llm_call(prompt_str: str, purpose: str, max_tokens: int = 400) -> str:
    """Route through Vertex → OpenAI → Ollama, return raw text or ''."""
    if _USE_VERTEX:
        try: return _vertex_text(prompt_str, purpose)
        except Exception as e: _debug(f"[LLM] Vertex failed ({purpose}): {e}")
    if _USE_OPENAI:
        try: return _openai_text(prompt_str, settings.openai_model_rewrite, max_tokens, purpose)
        except Exception as e: _debug(f"[LLM] OpenAI failed ({purpose}): {e}")
    try:
        model = _pick_ollama_model(settings.ollama_model_rewrite)
        return _ollama_text(prompt_str, model, max_tokens, purpose)
    except Exception as e:
        _debug(f"[LLM] Ollama failed ({purpose}): {e}")
    return ""


# =============================================================================
# SECTION 4 — Translation
# =============================================================================

def _is_english(text: str) -> bool:
    if not text: return True
    if re.search(r"[぀-ヿ一-鿿가-힯㐀-䶿]", text): return False
    if re.search(r"[؀-ۿ֐-׿฀-๿ऀ-ॿ]", text): return False
    try:
        text.encode("ascii")
        return True
    except UnicodeEncodeError:
        pass
    try: return (detect(text) or "").lower().startswith("en")
    except: return True


def translate_to_english(raw_text: str) -> str:
    cleaned = unicodedata.normalize("NFKC", raw_text).strip()
    if _is_english(cleaned):
        _debug(f"[Translator] English detected, skipping translation.")
        return cleaned

    tmpl = (
        "You are a professional Japanese/Chinese/English medical translator. "
        "Translate the user's text into natural, idiomatic ENGLISH only. "
        "Preserve clinical/anatomical terms precisely. No notes or headings.\n\n"
        f"Text:\n{cleaned}"
    )

    if _USE_VERTEX:
        try:
            out = _vertex_text(tmpl, "translate")
            out = re.sub(r"\s{2,}", " ", out.replace("\n"," ")).strip(" '\"")
            if out and not re.search(r"[\u3040-\u30FF\u4E00-\u9FFF]", out):
                return out
        except Exception as e: _debug(f"[Translator] Vertex failed: {e}")

    if _USE_OPENAI:
        try:
            out = _openai_text(tmpl, settings.openai_model_translate, 1024, "translate")
            out = re.sub(r"\s{2,}", " ", out.replace("\n"," ")).strip(" '\"\u201c\u201d\u2018\u2019`")
            return out or cleaned
        except Exception as e: _debug(f"[Translator] OpenAI failed: {e}")

    try:
        model = _pick_ollama_model(settings.ollama_model_translate)
        out   = _ollama_text(tmpl, model, 1024, "translate")
        return out or cleaned
    except Exception as e: _debug(f"[Translator] Ollama failed: {e}")

    return cleaned


# =============================================================================
# SECTION 5 — Prompt Rewriter (SnapMoGen Table 7)
# =============================================================================

_SNAPMOGEN_SYSTEM = (
    "Your task is to rewrite text prompts of user inputs for a text-to-motion generation model inference. "
    "This model generates 3D human motion data from text, you need to understand the intent of the user input "
    "and describe how the human body should move in detail, and give me the proper duration of the motion clip, "
    "usually from 4 to 12 seconds.\n\n"
    "Instructions:\n"
    "1. Make sure the rewritten prompts describe the human motion without major information loss.\n"
    "2. Be related to human body movements—the tool is not able to generate anything else.\n"
    "3. The rewritten prompt should be around 60 words, no more than 100.\n"
    "4. Use a clear, descriptive, and precise tone.\n"
    "5. Be creative and make the motion interesting and expressive.\n"
    "6. Feel free to add physical movement details.\n\n"
    "Examples:\n"
    "Input: Shooting a basketball.\n"
    "Rewrite: The person stands neutrally, then leans forward, spreading their legs wide. "
    "They simulate basketball dribbling with hand gestures, moving their hips side to side. "
    "The left hand performs dribbling actions. They pause, turn left, put the right leg forward, "
    "and squat slightly before simulating a basketball shot with a small jump.\n"
    "Length: 8 seconds\n\n"
    "Input: Zombie walk.\n"
    "Rewrite: The person shuffles forward with a stiff, dragging motion, one foot scraping the ground as it moves. "
    "His arms hang loosely by its sides, occasionally jerking forward as it staggers with uneven steps.\n"
    "Length: 6 seconds\n\n"
    "Now rewrite the following input. Return ONLY two lines:\n"
    "Line 1: Rewrite: <your rewritten description>\n"
    "Line 2: Length: <N> seconds"
)


def _parse_rewrite_output(raw: str) -> tuple:
    raw     = (raw or "").strip()
    len_m   = re.search(r"Length:\s*(\d+)", raw, re.IGNORECASE)
    secs    = max(4, min(12, int(len_m.group(1)) if len_m else 8))
    raw_no  = re.sub(r"Length:\s*\d+\s*seconds?", "", raw, flags=re.IGNORECASE).strip()
    desc_m  = re.search(r"(?:Rewrite:|Output:)\s*(.+)", raw_no, re.IGNORECASE | re.DOTALL)
    desc    = desc_m.group(1).strip() if desc_m else ""
    if not desc:
        for line in [l.strip() for l in raw_no.splitlines() if l.strip()]:
            if len(line) > 20 and not re.match(r"^(length|rewrite|input|output):", line, re.I):
                desc = line; break
        if not desc: desc = raw_no
    desc = re.sub(r"^(Rewrite:|Output:)\s*", "", desc, flags=re.IGNORECASE).strip()
    desc = re.sub(r"\s{2,}", " ", desc.replace("\n"," ")).strip('"\'')
    starters = re.compile(r"^(the person|a person|he |she |they |the patient|the individual)", re.IGNORECASE)
    if desc and not starters.match(desc):
        desc = "The person " + desc[0].lower() + desc[1:]
    return desc, secs


def _rewrite_fallback(text: str) -> tuple:
    t = (text or "").lower()
    if any(w in t for w in ["limp","drag","foot drop","drop foot","hemipleg"]):
        side = "right" if "right" in t else "left" if "left" in t else "right"
        opp  = "left" if side == "right" else "right"
        return (f"The person walks forward with an uneven, labored gait. The {side} foot barely clears "
                f"the ground during swing. The {side} leg takes a shorter step and the torso leans toward "
                f"the {side} side. The {side} arm swings with less amplitude."), 8
    if any(w in t for w in ["slow","shuffle","elderly","parkinson"]):
        return ("The person walks forward at a slow, cautious pace with short, careful steps. "
                "The knees remain slightly flexed and the feet barely lift from the ground. "
                "The arms swing minimally and the torso leans slightly forward."), 7
    return ("The person walks forward at a steady, natural pace. The arms swing alternately at the sides "
            "in rhythm with the opposite leg. Each foot lifts cleanly during swing and contacts the ground "
            "heel-first. The torso remains upright with small lateral shifts as weight transfers smoothly."), 7


def rewrite_prompt_auto(raw_text: str) -> tuple:
    text        = unicodedata.normalize("NFKC", (raw_text or "")).strip()
    full_prompt = _SNAPMOGEN_SYSTEM + f"\n\nInput: {text}"

    raw = _llm_call(full_prompt, "snapmogen_rewrite", max_tokens=300)
    if raw:
        desc, secs = _parse_rewrite_output(raw)
        if desc and len(desc.split()) >= 10:
            _debug(f"[Rewriter] success — {len(desc.split())} words, {secs}s")
            return desc, secs

    _debug("[Rewriter] Using rule-based fallback")
    return _rewrite_fallback(text)


# =============================================================================
# SECTION 6 — Intent Classifier: gait syndrome vs custom joint offset
# =============================================================================

_INTENT_CLASSIFY_PROMPT = """You are a motion-editor assistant. The user has typed a refinement instruction.
Classify whether it is:
  "gait"   — describes a clinical walking pattern, syndrome, or disease
              (e.g. "foot drop", "Parkinson's", "hemiplegic", "ataxia", "scissor gait")
  "offset" — describes a specific body-part position or angle adjustment
              (e.g. "hands on chest", "knee more bent", "head down", "arms raised")

Return ONLY the single word: gait  OR  offset

Examples:
"right foot drags on the floor"          → gait
"Parkinson's with freezing"              → gait
"moderate right hemiplegia post-stroke"  → gait
"patient's hands should be on their chest" → offset
"left knee 20 degrees more bent"         → offset
"head looking down"                      → offset
"both elbows slightly bent"              → offset
"make gait more severe"                  → gait
"reset everything"                       → offset

Input: "{prompt}"
Answer:"""

_GAIT_KEYWORDS = {
    # syndromes / conditions
    "parkinson","hemipleg","hemipar","stroke","ataxia","ataxic","cerebellar",
    "scissor","crouch","dystoni","chorea","huntington","festinat","freez",
    "foot drop","drop foot","footdrop","antalgic","limping","limp","equinus",
    "dipleg","myopath","waddl","tabetic","sensory atax","steppage",
    "cp gait","cerebral palsy","spastic","spasticity","muscular dystrophy",
    "leg length","lld","vaulting","posterior lurch","hip extensor",
    # primitive gait params
    "stride asymmetry","hip hike","trunk lean","arm swing","cadence",
    "wide base","forward lean","toe drag","heel slap","toe walk",
    # severity language in a gait context
    "gait more severe","gait worse","increase severity","make it worse",
    "make gait","gait reset","normal gait","healthy gait",
}
_OFFSET_KEYWORDS = {
    "chest","elbow","wrist","shoulder","arm","hand",
    "head down","head up","head looking","chin","neck","tuck",
    "knee bend","knee more","knee bent","knee flex",
    "hip flex","hip abduct","hip rotate",
    "foot turn","foot splay","toes out","toes in",
    "trunk forward","trunk lean more","spine","hunch",
    "raised","lowered","extended","offset","degree",
    "reset all custom","reset custom","reset adjustment","clear adjustment",
    "reset left","reset right","reset head","reset knee","reset elbow","reset shoulder","°",
}


def _classify_intent(prompt: str) -> str:
    """Return 'gait' or 'offset'. Uses keyword heuristic first, LLM if ambiguous."""
    t = prompt.lower()

    gait_hits   = sum(1 for kw in _GAIT_KEYWORDS   if kw in t)
    offset_hits = sum(1 for kw in _OFFSET_KEYWORDS if kw in t)

    # Clear winner from keywords
    if gait_hits > 0 and offset_hits == 0:
        _debug(f"[Classifier] gait (keyword, hits={gait_hits})")
        return "gait"
    if offset_hits > 0 and gait_hits == 0:
        _debug(f"[Classifier] offset (keyword, hits={offset_hits})")
        return "offset"

    # Ambiguous or no keywords — ask LLM
    llm_prompt = _INTENT_CLASSIFY_PROMPT.replace("{prompt}", prompt)
    raw = _llm_call(llm_prompt, "intent_classify", max_tokens=5).strip().lower()
    result = "gait" if "gait" in raw else "offset"
    _debug(f"[Classifier] LLM says '{raw}' → {result}")
    return result


# =============================================================================
# SECTION 7 — Gait Impairment Parser  (clinical NL → state dict)
# =============================================================================

_IMPAIRMENT_PARSE_PROMPT = """You are an expert clinical gait analyst for a biomechanics simulation system.

The user describes a patient's walking problem in ANY natural language.
Your job: map their description to simulation parameters and return ONLY a valid JSON object.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AVAILABLE PARAMETERS (float 0.0 to 1.0)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FOOT/ANKLE:
  ankle_drop_right / ankle_drop_left         → foot drop, toe drag, steppage
  equinus_right / equinus_left               → toe walking, heel never contacts

KNEE:
  knee_stiffness_right / knee_stiffness_left → stiff knee, rectus femoris spasticity

STRIDE:
  stride_asymmetry_right / stride_asymmetry_left → shorter step, uneven gait

TRUNK:
  trunk_lean_right / trunk_lean_left         → Trendelenburg, body tilts sideways
  forward_lean                               → stooped, Parkinson posture, camptocormia

PELVIS:
  hip_hike_right / hip_hike_left             → pelvis hikes up on swing side

ARMS:
  arm_swing_reduction_right / arm_swing_reduction_left → arm stiff, hemiplegic arm

SPEED/RHYTHM:
  cadence_reduction                          → walks slowly, bradykinesia
  festinating_gait                           → festination, steps get faster
  freezing_of_gait                           → feet glued, motor block, FOG

BASE:
  wide_base                                  → wide stance, ataxic base

NEUROLOGICAL NOISE:
  ataxic_gait                                → cerebellar stagger, dysmetria (bilateral)
  choreic_gait                               → Huntington's, involuntary jerks

UNILATERAL STRUCTURAL:
  dystonic_right / dystonic_left             → foot twists inward, equinovarus dystonia
  hip_extensor_weakness_right / left         → posterior lurch at heel strike
  leg_length_short_right / left              → shorter leg, vaulting

BILATERAL STRUCTURAL:
  sensory_ataxia                             → stomps, watches feet, proprioception loss
  waddling_gait                              → isolated bilateral hip drop

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMPOUND SYNDROMES (single key → full pattern)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  hemiplegic_right / hemiplegic_left   → stroke, hemiplegia, one-sided weakness
  parkinsonian_shuffle                 → Parkinson's, PD gait, shuffling
  cerebellar_ataxia                    → full cerebellar (ataxia+wide base+titubation)
  crouch_gait                          → CP crouch, persistent flexion
  scissor_gait                         → legs crossing, adductor spasticity
  diplegic                             → spastic diplegia, bilateral CP
  myopathic                            → muscular dystrophy, waddling+hyperlordosis
  antalgic_right / antalgic_left       → pain avoidance, limping due to pain

SYNDROME SELECTION RULES:
  "Parkinson's" alone           → parkinsonian_shuffle
  "Parkinson's + freezing"      → parkinsonian_shuffle + freezing_of_gait
  "Parkinson's + festination"   → parkinsonian_shuffle + festinating_gait
  "cerebellar ataxia" / "titubation" → cerebellar_ataxia (not ataxic_gait)
  "mild ataxia" / "just wobbly" → ataxic_gait (not cerebellar_ataxia)
  "bilateral CP"                → diplegic (not crouch+scissor separately)
  "muscular dystrophy"          → myopathic
  "stroke one side"             → hemiplegic_right or hemiplegic_left
  "Huntington's"                → choreic_gait

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SEVERITY MAPPING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  barely / very mild         = 0.2
  mild / slight              = 0.35
  moderate / noticeable      = 0.6
  severe / significant       = 0.8
  very severe / extreme      = 0.95
  (no qualifier)             = 0.6

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONVERSATIONAL ACTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  "more severe" / "worse"        → {"action": "increase_all"}
  "less severe" / "milder"       → {"action": "decrease_all"}
  "reset" / "normal" / "clear"   → {}
  "remove X"                     → set X to 0.0

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"right foot drags on the floor when they walk"
→ {"ankle_drop_right": 0.7, "stride_asymmetry_right": 0.5}

"Parkinson's with freezing episodes"
→ {"parkinsonian_shuffle": 0.7, "freezing_of_gait": 0.6}

"moderate right hemiplegia post-stroke"
→ {"hemiplegic_right": 0.6}

"cerebellar ataxia with head bobbing"
→ {"cerebellar_ataxia": 0.75}

"slight left foot drop"
→ {"ankle_drop_left": 0.35, "stride_asymmetry_left": 0.25}

"muscular dystrophy, waddling gait"
→ {"myopathic": 0.75}

"spastic diplegia, both legs affected"
→ {"diplegic": 0.75}

"Huntington's, involuntary jerks while walking"
→ {"choreic_gait": 0.7}

"can't feel feet, stomps and watches floor"
→ {"sensory_ataxia": 0.75}

"make it worse"
→ {"action": "increase_all"}

"go back to normal gait"
→ {}

Now extract parameters for:
Input: "{prompt}"
Output:"""


def _rule_based_impairment_parse(text: str) -> dict:
    t = text.lower()
    severity = (
        0.20 if any(w in t for w in ["barely","very mild","tiny","very slight"]) else
        0.35 if any(w in t for w in ["slight","mild","little","minor","early"]) else
        0.90 if any(w in t for w in ["severe","significant","heavy","marked","extreme","complete"]) else
        0.60
    )
    side = "right" if "right" in t else "left" if "left" in t else None

    if any(w in t for w in ["go back to normal","reset","healthy","clear all","start over","remove all","normal gait"]):
        return {}
    if any(w in t for w in ["make it worse","more severe","increase all","worsen"]):
        return {"action": "increase_all"}
    if any(w in t for w in ["less severe","reduce","milder","tone it down","decrease"]):
        return {"action": "decrease_all"}

    # Compounds first
    if any(w in t for w in ["muscular dystrophy","duchenne","becker md","limb-girdle","myopathic"]):
        return {"myopathic": severity}
    if any(w in t for w in ["spastic diplegia","diplegic","bilateral cp","both legs cp"]):
        return {"diplegic": severity}
    if any(w in t for w in ["hemipleg","hemiparesis","stroke","post-stroke","one side weak"]):
        s = side or "right"
        return {f"hemiplegic_{s}": severity}
    if any(w in t for w in ["parkinson","pd gait","petits pas"]):
        params = {"parkinsonian_shuffle": severity}
        if any(w in t for w in ["festination","festinating","faster and faster","steps getting"]):
            params["festinating_gait"] = round(severity * 0.85, 2)
        if any(w in t for w in ["freez","feet glued","motor block","sudden stop"]):
            params["freezing_of_gait"] = round(severity * 0.80, 2)
        return params
    if any(w in t for w in ["cerebellar ataxia","cerebellar stroke","cerebellar syndrome",
                              "spinocerebellar","sca","titubation","alcoholic cerebellar"]):
        return {"cerebellar_ataxia": severity}
    if any(w in t for w in ["crouch gait","crouched walking","hamstring contracture","persistent flexion"]):
        return {"crouch_gait": severity}
    if any(w in t for w in ["scissor","scissors","adductor spasticity","legs crossing"]):
        return {"scissor_gait": severity}
    if any(w in t for w in ["antalgic","pain avoidance","hip pain","knee pain","painful leg"]):
        s = side or "right"
        return {f"antalgic_{s}": severity}

    # Neurological noise
    if any(w in t for w in ["sensory ataxia","propriocept","dorsal column","tabetic","b12",
                              "stomps","slaps ground","watches feet","can't feel feet"]):
        return {"sensory_ataxia": severity}
    if any(w in t for w in ["chorea","huntington","choreic","involuntary jerk","tardive dyskinesia"]):
        return {"choreic_gait": severity}
    if any(w in t for w in ["ataxia","ataxic","stagger","lurch","cerebellar","wobbly","uncoordinated"]):
        return {"ataxic_gait": severity, "wide_base": round(severity * 0.6, 2)}

    # Festination / freezing standalone
    if any(w in t for w in ["festination","festinating","faster and faster steps","propulsive"]):
        return {"festinating_gait": severity}
    if any(w in t for w in ["freezing","feet glued","motor block","sudden stop","can't start"]):
        return {"freezing_of_gait": severity}

    # Unilateral structural
    if any(w in t for w in ["dystonia","dystonic","foot twists","equinovarus","runner's dystonia"]):
        s = side or "right"
        return {f"dystonic_{s}": severity}
    if any(w in t for w in ["equinus","toe walking","tiptoe","heel never","walks on toes"]):
        s = side or "right"
        return {f"equinus_{s}": severity}
    if any(w in t for w in ["leg length","shorter leg","lld","vaulting","unequal leg"]):
        s = side or "right"
        return {f"leg_length_short_{s}": severity}
    if any(w in t for w in ["posterior lurch","trunk lurches back","gluteus maximus","hip extensor"]):
        s = side or "right"
        return {f"hip_extensor_weakness_{s}": severity}

    # Primitives — may combine
    params = {}
    if any(w in t for w in ["foot drop","drop foot","footdrop","toe drag","drags foot",
                              "foot drags","drags on floor","can't lift foot","steppage"]):
        s = side or "right"
        params[f"ankle_drop_{s}"]          = severity
        params[f"stride_asymmetry_{s}"]    = round(severity * 0.75, 2)
        params[f"arm_swing_reduction_{s}"] = round(severity * 0.55, 2)
    if any(w in t for w in ["stiff knee","knee stiff","rigid knee","straight leg swing","rectus femoris"]):
        s = side or "right"
        params[f"knee_stiffness_{s}"] = severity
        params[f"hip_hike_{s}"]       = round(severity * 0.65, 2)
    if any(w in t for w in ["hip hike","pelvic hike","circumduction","hitch","pelvis rises"]):
        s = side or "right"
        params[f"hip_hike_{s}"] = severity
    if any(w in t for w in ["trunk lean","trendelenburg","leans to","tilts to","lateral lean"]):
        if side: params[f"trunk_lean_{side}"] = severity
        else:    params["trunk_lean_right"]   = round(severity * 0.65, 2)
    if any(w in t for w in ["arm swing","reduced arm","arm held","spastic arm"]):
        s = side or "right"
        params[f"arm_swing_reduction_{s}"] = severity
    if any(w in t for w in ["walks slowly","slow gait","cadence","bradykinesia","takes small steps"]):
        params["cadence_reduction"] = min(0.75, round(severity * 0.85, 2))
    if any(w in t for w in ["asymmetric","uneven step","shorter step","step length"]):
        s = side or "right"
        if f"ankle_drop_{s}" not in params:
            params[f"stride_asymmetry_{s}"] = severity
    if any(w in t for w in ["forward lean","stoop","hunch","bent forward","camptocormia"]):
        params["forward_lean"] = severity
    if any(w in t for w in ["wide base","wide stance","feet wide","broad base"]):
        params["wide_base"] = severity
    if any(w in t for w in ["waddle","waddling","duck walk","rolls side to side"]):
        if "myopathic" not in params:
            params["waddling_gait"] = severity

    return params


def _parse_impairment_prompt(user_prompt: str, current_state: dict) -> dict:
    """LLM → rule fallback → merge with session state. Returns new merged state."""
    import json as _json
    llm_prompt = _IMPAIRMENT_PARSE_PROMPT.replace("{prompt}", user_prompt)
    raw        = _llm_call(llm_prompt, "impairment_parse", max_tokens=300)

    new_params = {}
    if raw:
        m = re.search(r"\{.*\}", (raw or "").strip(), re.DOTALL)
        if m:
            try: new_params = _json.loads(m.group(0))
            except: pass

    if not new_params:
        _debug("[ImpairmentParser] Rule-based fallback")
        new_params = _rule_based_impairment_parse(user_prompt)

    _debug(f"[ImpairmentParser] extracted: {new_params}")

    action = new_params.pop("action", None)

    # ── Strip any non-numeric metadata keys the LLM may have added ──────────
    # e.g. {"action": "increase_specific", "parameter": "ankle_drop_right"}
    # After popping "action", "parameter" would be left — float() would crash.
    numeric_params = {}
    for k, v in new_params.items():
        try:
            numeric_params[k] = float(v)
        except (TypeError, ValueError):
            _debug(f"[ImpairmentParser] Skipping non-numeric key '{k}': {v!r}")

    merged = dict(current_state)

    if action == "increase_all":
        for k, v in merged.items():
            merged[k] = round(min(1.0, float(v) * 1.3), 2)
    elif action == "decrease_all":
        for k, v in merged.items():
            merged[k] = round(max(0.0, float(v) * 0.7), 2)
    elif action == "increase_specific":
        # LLM said "increase this one param" — bump it by 30% or set to 0.6 if new
        param = new_params.get("parameter") or next(iter(numeric_params), None)
        if param and isinstance(param, str) and not param.replace("_","").isdigit():
            current_val = merged.get(param, 0.5)
            merged[param] = round(min(1.0, float(current_val) * 1.3), 2)
            _debug(f"[ImpairmentParser] increase_specific '{param}': {merged[param]}")
    elif not numeric_params and action is None:
        merged = {}  # reset
    else:
        for k, v in numeric_params.items():
            if v == 0.0:
                merged.pop(k, None)
            else:
                merged[k] = round(max(0.0, min(1.0, v)), 2)

    _debug(f"[ImpairmentParser] final state: {merged}")
    return merged


# def _parse_impairment_prompt(user_prompt: str, current_state: dict) -> dict:
#     """LLM → rule fallback → merge with session state. Returns new merged state."""
#     import json as _json
#     llm_prompt = _IMPAIRMENT_PARSE_PROMPT.replace("{prompt}", user_prompt)
#     raw        = _llm_call(llm_prompt, "impairment_parse", max_tokens=300)
#
#     new_params = {}
#     if raw:
#         m = re.search(r"\{.*\}", (raw or "").strip(), re.DOTALL)
#         if m:
#             try: new_params = _json.loads(m.group(0))
#             except: pass
#
#     if not new_params:
#         _debug("[ImpairmentParser] Rule-based fallback")
#         new_params = _rule_based_impairment_parse(user_prompt)
#
#     _debug(f"[ImpairmentParser] extracted: {new_params}")
#
#     action = new_params.pop("action", None)
#     merged = dict(current_state)
#
#     if action == "increase_all":
#         for k, v in merged.items(): merged[k] = round(min(1.0, float(v) * 1.3), 2)
#     elif action == "decrease_all":
#         for k, v in merged.items(): merged[k] = round(max(0.0, float(v) * 0.7), 2)
#     elif not new_params and action is None:
#         merged = {}  # reset
#     else:
#         for k, v in new_params.items():
#             if float(v) == 0.0: merged.pop(k, None)
#             else: merged[k] = round(max(0.0, min(1.0, float(v))), 2)
#
#     _debug(f"[ImpairmentParser] final state: {merged}")
#     return merged


# =============================================================================
# SECTION 8 — FastAPI App Setup
# =============================================================================

tags_metadata = [
    {"name": "root",          "description": "Landing page."},
    {"name": "status",        "description": "Server state endpoints."},
    {"name": "prompts",       "description": "Prompt IO endpoints."},
    {"name": "downloads",     "description": "Download FBX/ZIP/Video."},
    {"name": "generation",    "description": "Text-to-motion pipeline."},
    {"name": "animation",     "description": "Pre-generated animation selection."},
    {"name": "questionnaire", "description": "Questionnaire schema + responses."},
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_dual_store()
    try:
        yield
    finally:
        pass


app = FastAPI(
    title="GaitSimPT-Codes (Dockerized)",
    description=(
        "FastAPI service for generating and retargeting patient-specific gait motions.\n\n"
        "Swagger UI: /docs   ReDoc: /redoc"
    ),
    version="2.0.0",
    openapi_tags=tags_metadata,
    lifespan=lifespan,
)

app.include_router(questionnaire_router, prefix="/questionnaire", tags=["questionnaire"])

text_prompt_global  = None
server_status_value = 4


# =============================================================================
# SECTION 9 — Session Store
# =============================================================================

# Each session stores:
#   base_bvh       : str  — path to clean base BVH (never modified)
#   impairments    : dict — {param_key: severity 0-1}
#   custom_offsets : list — [{joint_key, delta, phase, label}, ...]
#
# ADDITIVE MODEL:
#   - impairments   : new call merges on top of existing (replace same key, add new)
#   - custom_offsets: same joint+phase → replace; new joint → append; delta==0 → remove
#   - apply_all() always starts from clean base_bvh and applies BOTH layers fresh

_sessions: dict = {}
user_phase_state = {"user_id": None, "phase": None}


def _get_session(session_id: str) -> dict:
    if session_id not in _sessions:
        _sessions[session_id] = {
            "base_bvh":       None,
            "base_video":     None,   # path to per-session base video copy
            "impairments":    {},
            "custom_offsets": [],
        }
    s = _sessions[session_id]
    if "custom_offsets" not in s: s["custom_offsets"] = []
    if "impairments"    not in s: s["impairments"]    = {}
    if "base_video"     not in s: s["base_video"]     = None
    return s


# =============================================================================
# SECTION 10 — Pydantic Models
# =============================================================================

class ServerStatus(BaseModel):
    status: int = Field(..., description="0=not running, 1=running, 2=initial, 3=compare")

class Prompt(BaseModel):
    prompt: str = Field(..., description="User input prompt (any language)")

class RefineMotionRequest(BaseModel):
    session_id: str = Field(..., description="Session ID from /gen_text2motion/")
    prompt:     str = Field(..., description="Natural language refinement (any language)")

class RefineMotionResponse(BaseModel):
    status:           str
    session_id:       str
    prompt:           str
    intent:           str
    impairment_state: dict
    custom_offsets:   list
    labels:           list
    message:          str

class GenText2MotionResponse(BaseModel):
    status:            str
    message:           str
    original_prompt:   str
    english_prompt:    str
    expressive_prompt: str
    session_id:        str = ""

class MessageResponse(BaseModel):       message: str
class StatusResponse(BaseModel):        status: str; server_status: int
class ServerStatusMessageResponse(BaseModel): message: str
class GetPromptsResponse(BaseModel):    status: str; prompt: str
class CompareTriggerResponse(BaseModel): status: str; index: int
class AnimationStateResponse(BaseModel): model: str
class SetAnimationResponse(BaseModel):  status: str; server_status: int; model: str; index: int
class AnimationSelectionResponse(BaseModel): model: str; animation_id: int
class UserIdPhaseSelection(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=64)
    phase:   int = Field(..., ge=1, le=2)


# =============================================================================
# SECTION 11 — Root
# =============================================================================

@app.get("/", response_class=HTMLResponse, tags=["root"])
async def landing():
    return """<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>GaitSimPT-Codes (Dockerized)</title>
<style>body{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;background:#f5f7fa;color:#333}
.container{max-width:800px;margin:4rem auto;background:#fff;padding:2.5rem;border-radius:8px;box-shadow:0 4px 12px rgba(0,0,0,.05)}
h1{margin-top:0;font-size:2rem;color:#222}p{line-height:1.6}
.button{display:inline-block;margin:1rem 0;padding:.6rem 1.2rem;background:#007ACC;color:#fff;text-decoration:none;font-weight:500;border-radius:4px}
.button:hover{background:#005A9E}
footer{margin-top:3rem;font-size:.9rem;color:#666;border-top:1px solid #e1e4e8;padding-top:1rem}
footer a{color:#007ACC;text-decoration:none}footer a:hover{text-decoration:underline}</style></head>
<body><div class="container"><h1>GaitSimPT-Codes v2</h1>
<p>Unified FastAPI service for generating and retargeting patient-specific gait motions.
Supports 27 clinical gait impairments + arbitrary body-part joint offsets.</p>
<a class="button" href="/docs">Swagger UI</a>&nbsp;
<a class="button" href="https://mustafizur-r.github.io/SmartRehab/" target="_blank" rel="noopener">Project Homepage</a>
<p>Start with <code>/gen_text2motion/</code> then refine with <code>/refine_motion/</code>.</p>
<footer><p><strong>Prepared by:</strong> Md Mustafizur Rahman</p>
<p>Master's student, Interactive Media Design Laboratory, Division of Information Science, NAIST</p>
<p>Email: <a href="mailto:mustafizur.cd@gmail.com">mustafizur.cd@gmail.com</a></p></footer>
</div></body></html>"""


# =============================================================================
# SECTION 12 — Status Endpoints
# =============================================================================

@app.get("/server_status/", tags=["status"], response_model=ServerStatusMessageResponse)
async def get_server_status():
    msgs = {0:"Server is not running.",1:"Server is running.",2:"Initial server status.",3:"compare"}
    return {"message": msgs.get(server_status_value, "Initializing")}


@app.post("/set_server_status/", tags=["status"], response_model=ServerStatusMessageResponse)
async def set_server_status(server_status: ServerStatus):
    global server_status_value
    status = server_status.status
    if status in (0, 1, 2, 3):
        server_status_value = status
        labels = {0:"not running",1:"running",2:"initial status",3:"compare"}
        return {"message": f"Server status set to '{labels[status]}'."}
    raise HTTPException(status_code=400, detail="Invalid status value. Provide 0, 1, 2, or 3.")


@app.post("/userid_phase_selection", tags=["status"])
async def set_userid_phase_selection(payload: UserIdPhaseSelection):
    user_phase_state["user_id"] = payload.user_id
    user_phase_state["phase"]   = payload.phase
    return {"message": "updated successfully"}


@app.get("/userid_phase_selection_status", tags=["status"])
async def get_userid_phase_selection_status():
    return user_phase_state


@app.get("/set_status", tags=["status"], response_model=StatusResponse)
def set_status(status: int = Query(..., ge=0, le=5)):
    global server_status_value
    server_status_value = status
    return {"status": "updated", "server_status": server_status_value}


# =============================================================================
# SECTION 13 — Prompt IO
# =============================================================================

@app.get("/get_prompts/", tags=["prompts"], response_model=GetPromptsResponse)
async def get_prompts():
    try:
        with open("input.txt","r",encoding="utf-8") as f:
            return {"status":"success","prompt":f.read()}
    except Exception as e:
        raise HTTPException(status_code=500, detail={"status":"failed","error":str(e)})


@app.post("/input_prompts/", tags=["prompts"], response_model=MessageResponse)
async def input_prompts(prompt: Prompt):
    try:
        with open("input.txt","w",encoding="utf-8") as f: f.write(prompt.prompt)
        return {"message": "Prompt saved successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving prompt: {e}")


# =============================================================================
# SECTION 14 — Download Endpoints
# =============================================================================

@app.get("/download_fbx/", tags=["downloads"])
async def download_fbx(filename: str = Query(...)):
    filepath = os.path.join("fbx_folder", filename)
    if os.path.exists(filepath):
        return FileResponse(filepath, media_type="application/octet-stream", filename=filename)
    raise HTTPException(status_code=404, detail="File not found")


@app.get("/download_video", tags=["downloads"])
async def download_video():
    p = "./video_result/Final_Fbx_Mesh_Animation.mp4"
    if not os.path.exists(p): return {"error": "Video not found."}
    return FileResponse(path=p, filename="Final_Fbx_Mesh_Animation.mp4", media_type="video/mp4")


@app.get("/download_zip/", tags=["downloads"])
async def download_zip(filename: str = Query(...)):
    filepath = os.path.join("fbx_zip_folder", filename)
    if os.path.exists(filepath):
        return FileResponse(filepath, media_type="application/octet-stream", filename=filename)
    raise HTTPException(status_code=404, detail="File not found")


# =============================================================================
# SECTION 15 — Blender Command Helper
# =============================================================================

def _blender_cmd(video_render: str) -> str:
    flag = f"--video_render={video_render.lower()}"
    sys  = platform.system()
    if sys == "Windows":
        return f'blender --background --addons KeeMapAnimRetarget --python "./bvh2fbx/bvh2fbx.py" -- {flag}'
    elif sys == "Darwin":
        return (f'"/Applications/Blender.app/Contents/MacOS/Blender" --background '
                f'--addons KeeMapAnimRetarget --python "./bvh2fbx/bvh2fbx.py" -- {flag}')
    return f'xvfb-run blender --background --addons KeeMapAnimRetarget --python ./bvh2fbx/bvh2fbx.py -- {flag}'


def _run_blender(video_render: str) -> None:
    cmd    = _blender_cmd(video_render)
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        fbx_ok   = os.path.exists("./fbx_folder/bvh_0_out.fbx")
        video_ok = os.path.exists("./video_result/Final_Fbx_Mesh_Animation.mp4")
        if fbx_ok or video_ok:
            _debug(f"[Blender] Exited {result.returncode} but outputs exist — treating as success")
        else:
            raise subprocess.CalledProcessError(result.returncode, cmd)


# =============================================================================
# SECTION 16 — Generation Endpoint
# =============================================================================

@app.get("/gen_text2motion/", tags=["generation"], response_model=GenText2MotionResponse)
async def gen_text2motion(
    text_prompt:  str = Query(...),
    video_render: str = Query("false"),
    session_id:   str = Query(""),
):
    base_prompt = text_prompt.strip()
    sid         = session_id.strip() if session_id.strip() else str(uuid.uuid4())
    session     = _get_session(sid)
    _debug(f"[Session] {sid} — new generation")

    with open("input.txt","w",encoding="utf-8") as f: f.write(base_prompt)
    with open("video_title.txt","w",encoding="utf-8") as f: f.write(base_prompt)

    english_prompt            = translate_to_english(base_prompt)
    expressive_prompt, length = rewrite_prompt_auto(english_prompt)
    frames                    = length * 30
    clean_base                = re.sub(r"\s+#.*", "", english_prompt).strip()
    final_prompt              = f"{clean_base} # {expressive_prompt} #{frames}"

    with open("input_en.txt","w",encoding="utf-8") as f: f.write(english_prompt)
    with open("rewrite_input.txt","w",encoding="utf-8") as f: f.write(final_prompt)
    _debug(f"[Gen] Prompt: {final_prompt[:120]}...")

    try:
        def run_pipeline():
            subprocess.run("python gen_momask_plus.py", shell=True, check=True)
            _run_blender(video_render)
            base_bvh    = "bvh_folder/bvh_0_out.bvh"
            base_backup = f"bvh_folder/bvh_0_out_base_{sid}.bvh"
            os.makedirs("bvh_folder", exist_ok=True)
            if os.path.exists(base_bvh):
                shutil.copy(base_bvh, base_backup)
            session["base_bvh"]       = base_backup
            session["impairments"]    = {}
            session["custom_offsets"] = []
            # Copy base video per session so it survives future refinements
            base_vid_src = "./video_result/Final_Fbx_Mesh_Animation.mp4"
            base_vid_dst = f"./video_result/base_{sid}.mp4"
            if os.path.exists(base_vid_src):
                shutil.copy(base_vid_src, base_vid_dst)
                session["base_video"] = base_vid_dst
                _debug(f"[Session] {sid} — base video saved: {base_vid_dst}")
            _debug(f"[Session] {sid} — base BVH backed up: {base_backup}")

        await run_in_threadpool(run_pipeline)
        return {
            "status":            "success",
            "message":           f"Base motion generated. Use session_id='{sid}' to refine.",
            "original_prompt":   base_prompt,
            "english_prompt":    english_prompt,
            "expressive_prompt": expressive_prompt,
            "session_id":        sid,
            "base_video_url":    f"/download_base_video/?session_id={sid}",
        }
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {e}")


# =============================================================================
# SECTION 17 — UNIFIED REFINE MOTION ENDPOINT
# =============================================================================

@app.post("/refine_motion/", tags=["generation"], response_model=RefineMotionResponse)
async def refine_motion(
    request:      RefineMotionRequest,
    video_render: str = Query("false"),
):
    """
    Unified refinement endpoint. Accepts TWO kinds of prompt in any language:

    GAIT SYNDROME — describe a clinical walking condition:
      "right foot drags on the floor when they walk"
      "Parkinson's with freezing episodes"
      "moderate right hemiplegia post-stroke"
      "make it more severe"
      "reset gait to normal"

    CUSTOM JOINT OFFSET — describe a body-part position/angle:
      "patient's hands should be on their chest"
      "left knee 20 degrees more bent"
      "head looking slightly down"
      "both elbows more bent"
      "reset left knee offset"
      "reset all custom adjustments"

    ADDITIVE: each call MERGES with the existing session state.
    Previous impairments and offsets are preserved and combined with new ones.
    Starting from the clean base BVH every time, so nothing accumulates errors.
    """
    sid     = request.session_id.strip()
    session = _get_session(sid)

    base_bvh = session.get("base_bvh")
    if not base_bvh or not os.path.exists(base_bvh):
        raise HTTPException(
            status_code=400,
            detail=f"No base motion for session '{sid}'. Call /gen_text2motion/ first."
        )

    english_prompt = translate_to_english(request.prompt)
    _debug(f"[Refine] session={sid} prompt={english_prompt!r}")

    # ── Step 1: Classify intent ───────────────────────────────────────────────
    intent = _classify_intent(english_prompt)
    _debug(f"[Refine] intent={intent}")

    labels = []

    if intent == "gait":
        # ── Parse gait params and merge into session ──────────────────────────
        new_imp = _parse_impairment_prompt(english_prompt, session["impairments"])
        session["impairments"] = new_imp
        _debug(f"[Refine] impairments now: {session['impairments']}")

    else:  # offset
        # ── Parse custom offset and merge into session ────────────────────────
        def _llm(p): return _llm_call(p, "custom_offset", max_tokens=400)
        result = parse_custom_offset(english_prompt, llm_caller=_llm)

        if result["reset_all"]:
            session["custom_offsets"] = []
            _debug(f"[Refine] custom offsets reset")
        else:
            session["custom_offsets"] = merge_offsets(
                session["custom_offsets"], result["offsets"]
            )
            labels = result["labels"]
        _debug(f"[Refine] custom_offsets now: {len(session['custom_offsets'])} entries")

    # ── Step 2: Apply BOTH layers from clean base ─────────────────────────────
    os.makedirs("bvh_folder", exist_ok=True)
    os.makedirs("impaired_bvh_folder", exist_ok=True)
    output_bvh  = "bvh_folder/bvh_0_out.bvh"
    archive_bvh = f"impaired_bvh_folder/session_{sid}_latest.bvh"

    seed = hash(sid) % (2 ** 31)

    def _write_video_title() -> None:
        imp  = session["impairments"]
        cust = session["custom_offsets"]
        parts = []
        if imp:
            parts.append("Imp:[" + ",".join(f"{k}={v:.1f}" for k,v in list(imp.items())[:3]) + "]")
        if cust:
            parts.append(f"Offset:{len(cust)}joints")
        title = request.prompt + (" | " + " ".join(parts) if parts else "")
        try:
            with open("video_title.txt","w",encoding="utf-8") as f: f.write(title[:120])
        except: pass

    try:
        def run_refinement():
            _write_video_title()

            apply_all(
                input_bvh        = base_bvh,
                output_bvh       = output_bvh,
                impairment_state = session["impairments"],
                custom_offsets   = session["custom_offsets"],
                seed             = seed,
            )
            shutil.copy(output_bvh, archive_bvh)
            _debug(f"[Refine] BVH written → {output_bvh}")

            _run_blender(video_render)

        await run_in_threadpool(run_refinement)

        n_imp  = len(session["impairments"])
        n_cust = len(session["custom_offsets"])
        msg = (
            f"Applied {n_imp} impairment(s) + {n_cust} custom offset(s). "
            f"Download FBX from /download_fbx/"
        ) if (n_imp or n_cust) else "Motion reset to base. Download FBX from /download_fbx/"

        return {
            "status":           "success",
            "session_id":       sid,
            "prompt":           request.prompt,
            "intent":           intent,
            "impairment_state": session["impairments"],
            "custom_offsets":   session["custom_offsets"],
            "labels":           labels,
            "message":          msg,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Refinement failed: {e}")


# =============================================================================
# SECTION 18 — Session State Endpoint
# =============================================================================

@app.get("/session_state/{session_id}", tags=["generation"])
async def get_session_state(session_id: str):
    session = _get_session(session_id)
    return {
        "session_id":          session_id,
        "has_base_motion":     bool(session.get("base_bvh")),
        "base_bvh":            session.get("base_bvh"),
        "base_video":          session.get("base_video"),
        "impairments":         session.get("impairments", {}),
        "impairment_count":    len(session.get("impairments", {})),
        "custom_offsets":      session.get("custom_offsets", []),
        "custom_offset_count": len(session.get("custom_offsets", [])),
    }


@app.get("/download_base_video/", tags=["downloads"],
         summary="Download the base (unmodified) video for a session")
async def download_base_video(session_id: str = Query(...)):
    session   = _get_session(session_id)
    base_vid  = session.get("base_video")
    if base_vid and os.path.exists(base_vid):
        return FileResponse(path=base_vid, filename="base_motion.mp4", media_type="video/mp4")
    # Fallback: try to find it by pattern
    candidate = f"./video_result/base_{session_id}.mp4"
    if os.path.exists(candidate):
        return FileResponse(path=candidate, filename="base_motion.mp4", media_type="video/mp4")
    raise HTTPException(status_code=404, detail="Base video not found for this session.")


@app.delete("/cleanup_session/", tags=["generation"],
            summary="Delete all BVH and video files for a session (call on new chat)")
async def cleanup_session(session_id: str = Query(...)):
    """
    Deletes:
      - bvh_folder/bvh_0_out_base_{session_id}.bvh
      - impaired_bvh_folder/session_{session_id}_latest.bvh
      - video_result/base_{session_id}.mp4
    Removes session from in-memory store.
    Does NOT delete the shared bvh_0_out.bvh or Final_Fbx_Mesh_Animation.mp4
    (those belong to the current working files).
    """
    deleted = []
    errors  = []

    files_to_delete = [
        f"bvh_folder/bvh_0_out_base_{session_id}.bvh",
        f"impaired_bvh_folder/session_{session_id}_latest.bvh",
        f"video_result/base_{session_id}.mp4",
    ]

    for path in files_to_delete:
        if os.path.exists(path):
            try:
                os.remove(path)
                deleted.append(path)
                _debug(f"[Cleanup] Deleted: {path}")
            except Exception as e:
                errors.append(f"{path}: {e}")
                _debug(f"[Cleanup] Failed to delete {path}: {e}")

    # Remove from in-memory session store
    if session_id in _sessions:
        del _sessions[session_id]
        _debug(f"[Cleanup] Session {session_id} removed from memory")

    return {
        "status":  "ok",
        "deleted": deleted,
        "errors":  errors,
        "message": f"Cleaned up {len(deleted)} file(s) for session {session_id[:8]}…",
    }


# =============================================================================
# SECTION 19 — Animation Selection & Compare
# =============================================================================

animation_state = {"model": "pretrained", "index": 0}
compare_state   = {"index": 1, "triggered": False}


@app.get("/set_animation", tags=["animation"], response_model=SetAnimationResponse)
def set_animation(
    model: str = Query(..., enum=["pretrained","retrained"]),
    anim:  int = Query(..., ge=1, le=5),
):
    animation_state["model"] = model
    animation_state["index"] = anim
    return {"status":"updated","server_status":server_status_value,**animation_state}


@app.get("/get_animation", tags=["animation"])
def get_animation():
    model    = animation_state["model"]
    index    = animation_state["index"]
    filename = f"{model}_{index}.zip"
    zip_path = os.path.join("pregen_animation", filename)
    if os.path.isfile(zip_path):
        return FileResponse(zip_path, media_type="application/octet-stream", filename=filename)
    raise HTTPException(status_code=404, detail=f"File '{filename}' not found")


@app.post("/trigger_compare/", tags=["animation"], response_model=CompareTriggerResponse)
async def trigger_compare(index: int = Query(..., ge=1, le=5)):
    compare_state["index"]     = index
    compare_state["triggered"] = True
    return {"status":"compare triggered","index":index}


@app.get("/check_compare/", tags=["animation"])
async def check_compare():
    if compare_state["triggered"]:
        compare_state["triggered"] = False
        return {"status":"compare","index":compare_state["index"]}
    return {"status":"idle"}


@app.get("/check_animation_state/", tags=["animation"], response_model=AnimationStateResponse)
def check_animation_state():
    return {"model": animation_state["model"]}


@app.get("/get_animation_selection", tags=["animation"], response_model=AnimationSelectionResponse)
def get_animation_selection():
    model = animation_state.get("model","pretrained")
    index = max(1, min(5, animation_state.get("index",1)))
    return {"model":model,"animation_id":index}


# =============================================================================
# SECTION 20 — Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    print("[Startup] Launching FastAPI on 0.0.0.0:8000 ...")
    uvicorn.run("app_server:app", host="0.0.0.0", port=8000, reload=True)