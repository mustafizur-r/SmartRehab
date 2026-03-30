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
import platform
import subprocess
import unicodedata
import re
import os
import uuid
import shutil

DetectorFactory.seed = 0
load_dotenv()


# =============================================================================
# SECTION 1 — Logging
# =============================================================================

def _debug(msg: str) -> None:
    print(msg, flush=True)


# =============================================================================
# SECTION 2 — Settings  (loaded from .env)
# =============================================================================

class Settings(BaseModel):
    # OpenAI
    openai_api_key: str        = Field("",            alias="OPENAI_API_KEY")
    openai_model_translate: str = Field("gpt-4o-mini", alias="OPENAI_MODEL_TRANSLATE")
    openai_model_rewrite: str  = Field("gpt-4o-mini", alias="OPENAI_MODEL_REWRITE")
    openai_base_url: str       = Field("",            alias="OPENAI_BASE_URL")
    openai_timeout_seconds: int = Field(60,           alias="OPENAI_TIMEOUT_SECONDS")
    # Ollama
    ollama_model_translate: str = Field("llama3",     alias="OLLAMA_MODEL_TRANSLATE")
    ollama_model_rewrite: str   = Field("llama3",     alias="OLLAMA_MODEL_REWRITE")
    # Vertex AI
    google_credentials: str    = Field("",            alias="GOOGLE_APPLICATION_CREDENTIALS")
    vertex_project_id: str     = Field("",            alias="VERTEX_PROJECT_ID")
    vertex_location: str       = Field("us-central1", alias="VERTEX_LOCATION")
    vertex_model: str          = Field("gemini-2.5-pro-preview-06-05", alias="VERTEX_MODEL")


settings = Settings.model_validate(os.environ)

_USE_OPENAI = bool(settings.openai_api_key and settings.openai_api_key.strip())
_USE_VERTEX = bool(settings.google_credentials and settings.vertex_project_id)

if _USE_VERTEX:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.google_credentials
    _debug(f"[Startup] Vertex AI enabled. project={settings.vertex_project_id} "
           f"model={settings.vertex_model} key={settings.google_credentials}")
else:
    _debug("[Startup] Vertex AI not configured.")

client = None
if _USE_OPENAI:
    client = OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url or None,
        timeout=settings.openai_timeout_seconds,
    )
    _debug(f"[Startup] OpenAI enabled. "
           f"responses={hasattr(client,'responses')} "
           f"base_url={settings.openai_base_url or '(default)'}")
else:
    _debug("[Startup] OPENAI_API_KEY not found. Using Ollama fallback.")
    _debug(f"[Startup] Ollama models: "
           f"translate={settings.ollama_model_translate}, rewrite={settings.ollama_model_rewrite}")


# =============================================================================
# SECTION 3 — LLM Callers  (Vertex AI / OpenAI / Ollama)
# =============================================================================

def _vertex_text(prompt: str, purpose: str) -> str:
    """Call Gemini via Vertex AI using the service account JSON key."""
    try:
        import vertexai
        from vertexai.generative_models import GenerativeModel, GenerationConfig
    except ImportError:
        raise RuntimeError(
            "google-cloud-aiplatform not installed. Run: pip install google-cloud-aiplatform"
        )

    _debug(f"[LLM] provider=VertexAI purpose={purpose} model={settings.vertex_model}")
    vertexai.init(project=settings.vertex_project_id, location=settings.vertex_location)

    model  = GenerativeModel(settings.vertex_model)
    config = GenerationConfig(temperature=0.2, max_output_tokens=2048, top_p=0.9)
    out    = (model.generate_content(prompt, generation_config=config).text or "").strip()

    _debug(f"[VertexAI] raw output: {out!r}")

    # Strip markdown fences only — preserve newlines so _parse_rewrite_output
    # can split description from "Length:" line correctly
    out = re.sub(r"^```(?:text)?\s*", "", out, flags=re.IGNORECASE)
    out = re.sub(r"\s*```$", "", out).strip()
    _debug(f"[VertexAI] success — {len(out.split())} words")
    return out


def _openai_text(prompt: str, model: str, max_output_tokens: int, purpose: str) -> str:
    """Call OpenAI Responses API, falling back to Chat Completions."""
    import re as _re
    if client is None:
        raise RuntimeError("OpenAI client is not configured.")
    _debug(f"[LLM] provider=OpenAI purpose={purpose} model={model}")

    system_msg = "You follow instructions exactly and return plain text only."

    def _norm(s):
        return _re.sub(r"\s{2,}", " ", (s or "").replace("\r", " ").strip()).strip(" \"\u2018\u2019\u201c\u201d`'")

    def _strip_fences(s):
        s = _re.sub(r"^```(?:text)?\s*", "", (s or "").strip(), flags=_re.IGNORECASE)
        return _re.sub(r"\s*```$", "", s).strip()

    def _looks_like_meta(s):
        low = (s or "").lower()
        return any(t in low for t in ["okay, let's", "let's tackle", "the user wants",
                                       "first, i need", "analysis", "thinking", "reasoning"])

    def _salvage(s):
        chunks = [c.strip() for c in _re.split(r"\n\s*\n", (s or "").strip()) if c.strip()]
        cands  = [c for c in chunks if _re.search(r"[.!?]", c)]
        return cands[-1] if cands else (chunks[-1] if chunks else s)

    # Responses API (preferred)
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
                return _norm(_strip_fences(_salvage(out))) if _looks_like_meta(out) else out
    except Exception as e:
        _debug(f"[LLM] Responses API failed: {e}")

    # Chat Completions fallback
    if hasattr(client, "chat") and hasattr(client.chat, "completions"):
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system_msg},
                      {"role": "user",   "content": prompt}],
            temperature=0.2,
            max_tokens=int(max_output_tokens),
        )
        out = _norm(_strip_fences((resp.choices[0].message.content or "").strip()))
        return _norm(_strip_fences(_salvage(out))) if _looks_like_meta(out) else out

    raise RuntimeError("OpenAI SDK does not support responses or chat.completions.")


def _pick_ollama_model(preferred: str) -> str:
    """Return best available Ollama model (prefers qwen3)."""
    try:
        import ollama
        resp   = ollama.list()
        models = resp.get("models", []) if isinstance(resp, dict) else getattr(resp, "models", []) or []
        have   = [(m.get("model", "") if isinstance(m, dict) else getattr(m, "model", "")) for m in models]
        have   = [x for x in have if x]
        _debug(f"[Ollama] available_models={have}")
        qwen3  = sorted([m for m in have if m.lower().startswith("qwen3")], reverse=True)
        chosen = qwen3[0] if qwen3 else (preferred or (have[0] if have else "llama3"))
        _debug(f"[Ollama] chosen_model={chosen}")
        return chosen
    except Exception as e:
        _debug(f"[Ollama] list failed: {e}")
        return preferred or "llama3"


def _ollama_text(prompt: str, model: str, num_predict: int, purpose: str) -> str:
    """Call Ollama (JSON mode first, plain text fallback)."""
    import re as _re, json, ollama
    _debug(f"[LLM] provider=Ollama purpose={purpose} model={model}")

    def _norm(s):
        return _re.sub(r"\s{2,}", " ",
                       (s or "").replace("\r", " ").replace("\n", " ")
                       ).strip(" \"\u201c\u201d\u2018\u2019")

    def _strip_fences(s):
        s = _re.sub(r"^```(?:json)?\s*", "", (s or "").strip(), flags=_re.IGNORECASE)
        return _re.sub(r"\s*```$", "", s).strip()

    def _best_para(s):
        chunks = [c.strip() for c in _re.split(r"\n\s*\n", (s or "").strip()) if c.strip()]
        cands  = [c for c in chunks if _re.search(r"[.!?]", c)]
        return cands[-1] if cands else (chunks[-1] if chunks else s)

    def _looks_like_reasoning(s):
        low = (s or "").lower()
        return any(t in low for t in ["okay, let's", "the user wants", "first, i need",
                                       "analysis", "thinking", "reasoning", "step by step"])

    def _call(sys_msg: str, use_json: bool) -> str:
        kw   = {"format": "json"} if use_json else {}
        resp = ollama.chat(
            model=model,
            messages=[{"role": "system", "content": sys_msg},
                      {"role": "user",   "content": prompt}],
            options={"temperature": 0.0, "top_p": 0.9,
                     "num_ctx": 4096, "num_predict": int(num_predict)},
            **kw,
        )
        msg = getattr(resp, "message", None)
        return (getattr(msg, "content", "") or "").strip() if msg else ""

    # JSON mode
    try:
        raw  = _strip_fences(_call(
            "Return ONLY a JSON object with key 'text' containing the final plain-English paragraph.",
            True,
        ))
        data = {}
        try:
            data = json.loads(raw)
        except Exception:
            m = _re.search(r"\{.*\}", raw, _re.DOTALL)
            if m:
                try: data = json.loads(m.group(0))
                except: pass
        out = _norm(data.get("text", "") if isinstance(data, dict) else "")
        if out:
            return _best_para(out) if _looks_like_reasoning(out) else out
    except Exception as e:
        _debug(f"[Ollama] json mode failed: {e}")

    # Plain text fallback
    try:
        out = _norm(_strip_fences(_call(
            "Return ONLY the final paragraph as plain text. Never return empty.", False
        )))
        return _best_para(out) if _looks_like_reasoning(out) else out
    except Exception as e:
        _debug(f"[Ollama] chat failed: {e}")
        return ""


# =============================================================================
# SECTION 4 — Translation   (Vertex AI → OpenAI → Ollama → unchanged)
# =============================================================================

def _is_english(text: str) -> bool:
    """
    Robust English detection — two-layer check.

    Layer 1: character-level check (fast, no model needed).
      If text contains CJK or Japanese/Korean characters → definitely not English.
      If text is pure ASCII (or common accented Latin) → almost certainly English.

    Layer 2: langdetect (only for ambiguous mixed-script text).
      langdetect is unreliable on short phrases (<5 words), so we trust the
      character check over it for short inputs.
    """
    if not text:
        return True

    # Contains CJK / Japanese / Korean → definitely needs translation
    if re.search(r"[぀-ヿ一-鿿가-힯㐀-䶿]", text):
        return False

    # Contains Arabic, Hebrew, Thai, Devanagari etc. → needs translation
    if re.search(r"[؀-ۿ֐-׿฀-๿ऀ-ॿ]", text):
        return False

    # Pure ASCII (letters, digits, punctuation, spaces) → English
    try:
        text.encode("ascii")
        return True
    except UnicodeEncodeError:
        pass

    # Mixed Latin (accented chars) — use langdetect but trust it only if confident
    try:
        lang = detect(text) or ""
        return lang.lower().startswith("en")
    except Exception:
        # langdetect failed → assume English for Latin-script text
        return True


def translate_to_english(raw_text: str) -> str:
    """Translate any language input to English. Returns original if already English."""
    cleaned  = unicodedata.normalize("NFKC", raw_text).strip()
    original = cleaned

    # Skip if already English — use robust two-layer check
    if _is_english(cleaned):
        _debug(f"[Translator] English input detected, skipping translation: {cleaned!r}")
        return original

    tmpl = (
        "You are a professional Japanese/Chinese/English medical translator. "
        "Translate the user's text into natural, idiomatic ENGLISH only.\n"
        "Preserve clinical/anatomical terms and laterality precisely. "
        "Do not add or omit information. No notes or headings.\n\n"
        f"Text:\n{cleaned}"
    )

    # Vertex AI
    if _USE_VERTEX:
        try:
            out = _vertex_text(tmpl, "translate")
            out = re.sub(r"\s{2,}", " ", out.replace("\n", " ")).strip(" '\"")
            if out and not re.search(r"[\u3040-\u30FF\u4E00-\u9FFF]", out):
                _debug("[Translator] VertexAI success")
                return out
        except Exception as e:
            _debug(f"[Translator] VertexAI failed: {e}")

    # OpenAI
    if _USE_OPENAI:
        try:
            out = _openai_text(tmpl, settings.openai_model_translate, 1024, "translate")
            out = re.sub(r"\s{2,}", " ", out.replace("\n", " ")).strip(" '\"\u201c\u201d\u2018\u2019`")
            if re.search(r"[\u3040-\u30FF\u4E00-\u9FFF]", out):
                strict = (
                    "Translate STRICTLY into ENGLISH ONLY. No headings, no notes, no quotes, no markdown. "
                    "Return only fluent English sentences.\n\n" + cleaned
                )
                out2 = _openai_text(strict, settings.openai_model_translate, 1024, "translate_strict")
                out2 = re.sub(r"\s{2,}", " ", out2.replace("\n", " ")).strip(" '\"\u201c\u201d\u2018\u2019`")
                if out2:
                    out = out2
            _debug("[Translator] provider=OpenAI success")
            return out or original
        except Exception as e:
            _debug(f"[Translator] provider=OpenAI failed, fallback to Ollama. error={e}")

    # Ollama
    try:
        model = _pick_ollama_model(settings.ollama_model_translate)
        out   = _ollama_text(tmpl, model, 1024, "translate")
        if re.search(r"[\u3040-\u30FF\u4E00-\u9FFF]", out):
            strict = (
                "Translate STRICTLY into ENGLISH ONLY. No headings, no notes, no quotes, no markdown. "
                "Return only fluent English sentences.\n\n" + cleaned
            )
            out2 = _ollama_text(strict, model, 1024, "translate_strict")
            if out2:
                out = out2
        _debug("[Translator] provider=Ollama success")
        return out or original
    except Exception as e:
        _debug(f"[Translator] provider=Ollama failed. error={e}")
        return original


# =============================================================================
# SECTION 5 — Prompt Rewriter   (SnapMoGen Table 7 style)
#             Priority: Vertex AI → OpenAI → Ollama → rule-based fallback
# =============================================================================

# System prompt taken word-for-word from SnapMoGen paper Table 7
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
    """Extract (description, seconds) from LLM output."""
    raw    = (raw or "").strip()

    # Extract length first
    len_m  = re.search(r"Length:\s*(\d+)", raw, re.IGNORECASE)
    secs   = int(len_m.group(1)) if len_m else 8
    secs   = max(4, min(12, secs))

    # Remove the Length line from raw before extracting description
    raw_no_len = re.sub(r"Length:\s*\d+\s*seconds?", "", raw, flags=re.IGNORECASE).strip()

    # Extract description — everything after "Rewrite:" or "Output:"
    desc_m = re.search(r"(?:Rewrite:|Output:)\s*(.+)", raw_no_len, re.IGNORECASE | re.DOTALL)
    desc   = desc_m.group(1).strip() if desc_m else ""

    # If no prefix found, use the whole cleaned text
    if not desc:
        for line in [l.strip() for l in raw_no_len.splitlines() if l.strip()]:
            if len(line) > 20 and not re.match(r"^(length|rewrite|input|output):", line, re.I):
                desc = line
                break
        if not desc:
            desc = raw_no_len

    # Clean up
    desc = re.sub(r"^(Rewrite:|Output:)\s*", "", desc, flags=re.IGNORECASE).strip()
    desc = re.sub(r"\s{2,}", " ", desc.replace("\n", " ")).strip('"\'')

    # Fix "The person" prefix — only add if description doesn't already
    # start with "The person", "A person", "He ", "She ", "They " etc.
    person_starters = re.compile(
        r"^(the person|a person|he |she |they |the patient|the individual)",
        re.IGNORECASE
    )
    if desc and not person_starters.match(desc):
        desc = "The person " + desc[0].lower() + desc[1:]

    return desc, secs


def _rewrite_fallback(text: str) -> tuple:
    """
    Rule-based fallback when all LLMs are unavailable.
    Written in SnapMoGen Table 7 annotation style.
    Returns (description, seconds).
    """
    t = (text or "").lower()

    if any(w in t for w in ["limp", "drag", "foot drop", "drop foot", "hemipleg", "hemiparesis"]):
        side = "right" if "right" in t else "left" if "left" in t else "right"
        opp  = "left" if side == "right" else "right"
        return (
            f"The person walks forward with an uneven, labored gait. "
            f"The {side} foot barely clears the ground during swing, scraping slightly before heel contact. "
            f"The {side} leg takes a shorter step than the {opp} and the torso leans subtly toward the {side} side. "
            f"The {side} arm swings with less amplitude, staying closer to the body throughout."
        ), 8

    if any(w in t for w in ["stiff knee", "stiff leg", "stiff right", "stiff left"]):
        side = "right" if "right" in t else "left"
        return (
            f"The person walks forward with the {side} knee held relatively extended throughout the stride. "
            f"The {side} leg swings outward in a slight arc during swing rather than bending normally. "
            f"The pelvis tilts upward on the {side} side to aid foot clearance."
        ), 7

    if any(w in t for w in ["trunk lean", "lean to the", "leans right", "leans left"]):
        side = "right" if "right" in t else "left"
        return (
            f"The person walks forward with a noticeable lateral lean of the torso toward the {side}. "
            f"Each step is deliberate with weight shifting visibly to the {side} side during stance. "
            f"The arms stay low at the sides and the head remains level throughout."
        ), 7

    if any(w in t for w in ["slow", "shuffle", "elderly", "careful"]):
        return (
            "The person walks forward at a slow, cautious pace with short, careful steps. "
            "The knees remain slightly flexed and the feet barely lift from the ground during swing. "
            "The arms swing minimally at the sides and the torso leans slightly forward."
        ), 6

    if any(w in t for w in ["fast", "quick", "brisk", "energetic"]):
        return (
            "The person strides forward with quick, energetic steps. "
            "The knees lift higher than normal and the arms pump actively at the sides. "
            "The torso leans slightly forward as momentum builds with each stride."
        ), 5

    return (
        "The person walks forward at a steady, natural pace. "
        "The arms swing alternately at the sides in rhythm with the opposite leg. "
        "Each foot lifts cleanly during swing and contacts the ground heel-first at stance. "
        "The torso remains upright with small lateral shifts as weight transfers smoothly between steps."
    ), 7


def rewrite_prompt_auto(raw_text: str) -> tuple:
    """
    Rewrite user prompt in SnapMoGen Table 7 style.
    Returns (expressive_description: str, length_seconds: int).
    """
    text        = unicodedata.normalize("NFKC", (raw_text or "")).strip()
    full_prompt = _SNAPMOGEN_SYSTEM + f"\n\nInput: {text}"

    # Vertex AI
    if _USE_VERTEX:
        try:
            raw  = _vertex_text(full_prompt, "snapmogen_rewrite")
            _debug(f"[Rewriter] VertexAI raw output: {raw[:120]!r}")
            desc, secs = _parse_rewrite_output(raw)
            if desc and len(desc.split()) >= 10:
                _debug(f"[Rewriter] VertexAI success — {len(desc.split())} words, {secs}s")
                return desc, secs
            _debug(f"[Rewriter] VertexAI output too short ({len(desc.split())} words) — trying fallback")
        except Exception as e:
            _debug(f"[Rewriter] VertexAI failed: {e}")

    # OpenAI
    if _USE_OPENAI:
        try:
            raw  = _openai_text(full_prompt, settings.openai_model_rewrite, 300, "snapmogen_rewrite")
            desc, secs = _parse_rewrite_output(raw)
            if desc and len(desc.split()) >= 10:
                _debug(f"[Rewriter] OpenAI success — {len(desc.split())} words, {secs}s")
                return desc, secs
        except Exception as e:
            _debug(f"[Rewriter] OpenAI failed: {e}")

    # Ollama
    try:
        model = _pick_ollama_model(settings.ollama_model_rewrite)
        raw   = _ollama_text(full_prompt, model, 300, "snapmogen_rewrite")
        desc, secs = _parse_rewrite_output(raw)
        if desc and len(desc.split()) >= 10:
            _debug(f"[Rewriter] Ollama success — {len(desc.split())} words, {secs}s")
            return desc, secs
    except Exception as e:
        _debug(f"[Rewriter] Ollama failed: {e}")

    # Rule-based fallback
    _debug("[Rewriter] Using rule-based fallback")
    return _rewrite_fallback(text)


# =============================================================================
# SECTION 6 — FastAPI App Setup
# =============================================================================

tags_metadata = [
    {"name": "root",          "description": "Landing page and basic service info."},
    {"name": "status",        "description": "Server state endpoints used by clients."},
    {"name": "prompts",       "description": "Prompt IO endpoints."},
    {"name": "downloads",     "description": "Download generated assets (FBX/ZIP/Video)."},
    {"name": "generation",    "description": "Text-to-motion generation pipeline."},
    {"name": "animation",     "description": "Pre-generated animation selection and compare mode."},
    {"name": "questionnaire", "description": "Questionnaire schema + saving responses (jsonl + sqlite)."},
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
        "A Dockerized FastAPI service for generating and retargeting patient-specific gait motions. "
        "This service exposes endpoints for prompt handling, motion generation, and asset downloads.\n\n"
        "Swagger UI: /docs\nReDoc: /redoc"
    ),
    version="1.0.0",
    openapi_tags=tags_metadata,
    lifespan=lifespan,
)

app.include_router(questionnaire_router, prefix="/questionnaire", tags=["questionnaire"])

text_prompt_global  = None
server_status_value = 4  # Default: Initializing


# =============================================================================
# SECTION 7 — Pydantic Request / Response Models
# =============================================================================

class ServerStatus(BaseModel):
    status: int = Field(..., description="0=not running, 1=running, 2=initial status, 3=compare")

class Prompt(BaseModel):
    prompt: str = Field(..., description="User input prompt (any language)")

class MessageResponse(BaseModel):
    message: str

class StatusResponse(BaseModel):
    status: str
    server_status: int

class ServerStatusMessageResponse(BaseModel):
    message: str

class GetPromptsResponse(BaseModel):
    status: str
    prompt: str

# GenText2MotionResponse moved below AnimationSelectionResponse

class CompareTriggerResponse(BaseModel):
    status: str
    index: int

class AnimationStateResponse(BaseModel):
    model: str

class SetAnimationResponse(BaseModel):
    status: str
    server_status: int
    model: str
    index: int

class UserIdPhaseSelection(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=64)
    phase: int   = Field(..., ge=1, le=2)  # 1=Single, 2=Compare

class AnimationSelectionResponse(BaseModel):
    model: str
    animation_id: int

class RefineMotionRequest(BaseModel):
    session_id: str = Field(..., description="Session ID returned by /gen_text2motion/")
    prompt: str     = Field(..., description="Natural language refinement prompt (any language)")

class RefineMotionResponse(BaseModel):
    status: str
    session_id: str
    prompt: str
    impairment_state: dict
    message: str

class GenText2MotionResponse(BaseModel):
    status: str
    message: str
    original_prompt: str
    english_prompt: str
    expressive_prompt: str
    session_id: str = ""

user_phase_state = {"user_id": None, "phase": None}

# In-memory session store: { session_id: { "base_bvh": str, "impairments": dict } }
_sessions: dict = {}

def _get_session(session_id: str) -> dict:
    if session_id not in _sessions:
        _sessions[session_id] = {"base_bvh": None, "impairments": {}}
    return _sessions[session_id]


# =============================================================================
# SECTION 8 — Root Endpoint
# =============================================================================

@app.get("/", response_class=HTMLResponse, tags=["root"], summary="Landing page")
async def landing():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>GaitSimPT-Codes (Dockerized)</title>
      <style>
        body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #f5f7fa; color: #333; }
        .container { max-width: 800px; margin: 4rem auto; background: #ffffff; padding: 2.5rem; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }
        h1 { margin-top: 0; font-size: 2rem; color: #222; }
        p  { line-height: 1.6; }
        .button { display: inline-block; margin: 1rem 0; padding: 0.6rem 1.2rem; background-color: #007ACC; color: #fff; text-decoration: none; font-weight: 500; border-radius: 4px; }
        .button:hover { background-color: #005A9E; }
        footer { margin-top: 3rem; font-size: 0.9rem; color: #666; border-top: 1px solid #e1e4e8; padding-top: 1rem; }
        footer a { color: #007ACC; text-decoration: none; }
        footer a:hover { text-decoration: underline; }
      </style>
    </head>
    <body>
      <div class="container">
        <h1>GaitSimPT-Codes (Dockerized)</h1>
        <p>
          A Dockerized FastAPI service for generating and retargeting patient-specific gait motions.
          Combines state-of-the-art text-to-motion models, Blender automation,
          and the KeeMap rig-retargeting addon.
        </p>
        <a class="button" href="https://mustafizur-r.github.io/SmartRehab/" target="_blank" rel="noopener">
          Project Homepage
        </a>
        <p>Start by calling the <code>/gen_text2motion/</code> endpoint.</p>
        <footer>
          <p><strong>Prepared by:</strong> Md Mustafizur Rahman</p>
          <p>Master's student, Interactive Media Design Laboratory,<br>Division of Information Science, NAIST</p>
          <p>Email: <a href="mailto:mustafizur.cd@gmail.com">mustafizur.cd@gmail.com</a></p>
        </footer>
      </div>
    </body>
    </html>
    """


# =============================================================================
# SECTION 9 — Status Endpoints
# =============================================================================

@app.get("/server_status/", tags=["status"], summary="Get server status message",
         response_model=ServerStatusMessageResponse)
async def get_server_status():
    msgs = {0: "Server is not running.", 1: "Server is running.",
            2: "Initial server status.",  3: "compare"}
    return {"message": msgs.get(server_status_value, "Initializing")}


@app.post("/set_server_status/", tags=["status"], summary="Set server status",
          response_model=ServerStatusMessageResponse)
async def set_server_status(server_status: ServerStatus):
    global server_status_value
    status = server_status.status
    print(f"Received status: {status}")
    if status in (0, 1, 2, 3):
        server_status_value = status
        labels = {0: "not running", 1: "running", 2: "initial status", 3: "compare"}
        return {"message": f"Server status set to '{labels[status]}'."}
    raise HTTPException(status_code=400, detail="Invalid status value. Please provide 0, 1, 2, or 3.")


@app.post("/userid_phase_selection", tags=["status"],
          summary="Set current user_id and phase (1=Single, 2=Compare)")
async def set_userid_phase_selection(payload: UserIdPhaseSelection):
    user_phase_state["user_id"] = payload.user_id
    user_phase_state["phase"]   = payload.phase
    return {"message": "updated successfully"}


@app.get("/userid_phase_selection_status", tags=["status"],
         summary="Get current user_id and phase status")
async def get_userid_phase_selection_status():
    return user_phase_state


@app.get("/set_status", tags=["status"], summary="Set server status (legacy)",
         response_model=StatusResponse)
def set_status(status: int = Query(..., ge=0, le=5, description="0..5 legacy status range")):
    global server_status_value
    server_status_value = status
    return {"status": "updated", "server_status": server_status_value}


# =============================================================================
# SECTION 10 — Prompt IO Endpoints
# =============================================================================

@app.get("/get_prompts/", tags=["prompts"], summary="Read latest input prompt from input.txt",
         response_model=GetPromptsResponse)
async def get_prompts():
    try:
        with open("input.txt", "r", encoding="utf-8") as f:
            return {"status": "success", "prompt": f.read()}
    except Exception as e:
        raise HTTPException(status_code=500, detail={"status": "failed", "error": str(e)})


@app.post("/input_prompts/", tags=["prompts"], summary="Write prompt to input.txt",
          response_model=MessageResponse)
async def input_prompts(prompt: Prompt):
    try:
        with open("input.txt", "w", encoding="utf-8") as f:
            f.write(prompt.prompt)
        return {"message": "Prompt saved successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving prompt: {e}")


# =============================================================================
# SECTION 11 — Download Endpoints
# =============================================================================

@app.get("/download_fbx/", tags=["downloads"], summary="Download an FBX file from fbx_folder")
async def download_bvh(filename: str = Query(..., description="FBX filename inside fbx_folder")):
    filepath = os.path.join("fbx_folder", filename)
    if os.path.exists(filepath):
        return FileResponse(filepath, media_type="application/octet-stream", filename=filename)
    raise HTTPException(status_code=404, detail="File not found")


@app.get("/download_video", tags=["downloads"], summary="Download rendered MP4 video")
async def download_video():
    p = "./video_result/Final_Fbx_Mesh_Animation.mp4"
    if not os.path.exists(p):
        return {"error": "Video not found."}
    return FileResponse(path=p, filename="Final_Fbx_Mesh_Animation.mp4", media_type="video/mp4")


@app.get("/download_zip/", tags=["downloads"], summary="Download a ZIP file from fbx_zip_folder")
async def download_fbx(filename: str = Query(..., description="ZIP filename inside fbx_zip_folder")):
    filepath = os.path.join("fbx_zip_folder", filename)
    if os.path.exists(filepath):
        return FileResponse(filepath, media_type="application/octet-stream", filename=filename)
    raise HTTPException(status_code=404, detail="File not found")


# =============================================================================
# SECTION 12 — Generation Endpoint  (translate → rewrite → MoMask → Blender)
# =============================================================================

def _blender_cmd(video_render: str) -> str:
    """Return the platform-appropriate Blender command."""
    flag = f"--video_render={video_render.lower()}"
    sys  = platform.system()
    if sys == "Windows":
        return (f'blender --background --addons KeeMapAnimRetarget '
                f'--python "./bvh2fbx/bvh2fbx.py" -- {flag}')
    elif sys == "Darwin":
        return (f'"/Applications/Blender.app/Contents/MacOS/Blender" --background '
                f'--addons KeeMapAnimRetarget --python "./bvh2fbx/bvh2fbx.py" -- {flag}')
    elif sys == "Linux":
        return (f'xvfb-run blender --background '
                f'--addons KeeMapAnimRetarget --python ./bvh2fbx/bvh2fbx.py -- {flag}')
    return (f'blender --background --addons KeeMapAnimRetarget '
            f'--python "./bvh2fbx/bvh2fbx.py" -- {flag}')


@app.get(
    "/gen_text2motion/",
    tags=["generation"],
    summary="Generate motion from text prompt (translation + rewrite + MoMask + Blender)",
    response_model=GenText2MotionResponse,
)
async def gen_text2motion(
    text_prompt:  str = Query(...,     description="Text prompt (any language)"),
    video_render: str = Query("false", description="Render video (true/false)"),
    session_id:   str = Query("",      description="Optional existing session ID. New one created if empty."),
):
    base_prompt = text_prompt.strip()

    # Create or reuse session
    sid     = session_id.strip() if session_id.strip() else str(uuid.uuid4())
    session = _get_session(sid)
    _debug(f"[Session] {sid} — new generation")

    with open("input.txt", "w", encoding="utf-8") as f:
        f.write(base_prompt)
    print("[INFO] Saved base prompt -> input.txt")

    # Write video title for Blender — on base generation use the original prompt
    with open("video_title.txt", "w", encoding="utf-8") as f:
        f.write(base_prompt)
    print("[INFO] Saved video title -> video_title.txt")

    # Step 1 — Translate to English
    english_prompt = translate_to_english(base_prompt)
    with open("input_en.txt", "w", encoding="utf-8") as f:
        f.write(english_prompt)
    print("[INFO] Saved English prompt -> input_en.txt")

    # Step 2 — Rewrite in SnapMoGen style
    expressive_prompt, length_seconds = rewrite_prompt_auto(english_prompt)
    frames       = length_seconds * 30  # SnapMoGen runs at 30 fps
    clean_base   = re.sub(r"\s+#.*", "", english_prompt).strip()
    final_prompt = f"{clean_base} # {expressive_prompt} #{frames}"
    with open("rewrite_input.txt", "w", encoding="utf-8") as f:
        f.write(final_prompt)
    print(f"[INFO] Saved rewritten prompt -> rewrite_input.txt ({length_seconds}s, {frames} frames)")
    print(f"[INFO] Prompt: {final_prompt[:120]}...")

    # Step 3 — Run MoMask + Blender
    try:
        def run_pipeline():
            subprocess.run("python gen_momask_plus.py", shell=True, check=True)
            cmd = _blender_cmd(video_render)
            print("Launching Blender:", cmd)
            result = subprocess.run(cmd, shell=True)

            # Tolerate Blender EXCEPTION_ACCESS_VIOLATION crash-after-render
            if result.returncode != 0:
                fbx_ok   = os.path.exists("./fbx_folder/bvh_0_out.fbx")
                video_ok = os.path.exists("./video_result/Final_Fbx_Mesh_Animation.mp4")
                if fbx_ok or video_ok:
                    print(f"[INFO] Blender exited {result.returncode} but outputs exist "
                          f"(fbx={fbx_ok}, video={video_ok}) — treating as success")
                else:
                    raise subprocess.CalledProcessError(result.returncode, cmd)

        await run_in_threadpool(run_pipeline)

        # Store base BVH — gen_momask_plus.py saves to bvh_folder/bvh_0_out.bvh
        base_bvh    = "bvh_folder/bvh_0_out.bvh"
        base_backup = f"bvh_folder/bvh_0_out_base_{sid}.bvh"
        os.makedirs("bvh_folder", exist_ok=True)
        if os.path.exists(base_bvh):
            shutil.copy(base_bvh, base_backup)  # keep clean copy for iterative refinement
        session["base_bvh"]    = base_backup    # always apply impairments to this clean copy
        session["impairments"] = {}
        _debug(f"[Session] {sid} — clean base BVH backed up: {base_backup}")

        return {
            "status":            "success",
            "message":           f"Base motion generated. Use session_id='{sid}' to refine.",
            "original_prompt":   base_prompt,
            "english_prompt":    english_prompt,
            "expressive_prompt": expressive_prompt,
            "session_id":        sid,
        }

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Command execution failed: {e}")


# =============================================================================
# SECTION 13 — Iterative Refinement Endpoints  (/refine_motion + /session_state)
# =============================================================================

# LLM prompt for extracting impairment parameters from natural language
_IMPAIRMENT_PARSE_PROMPT = """You are a clinical gait impairment parameter extractor for a biomechanics simulation system.

The user describes a gait modification. Extract parameters and return ONLY a valid JSON object.

AVAILABLE PARAMETERS (all float 0.0 to 1.0):

Primitive parameters:
  ankle_drop_right / ankle_drop_left            : foot drop, dragging foot, equinus foot
  knee_stiffness_right / knee_stiffness_left    : stiff knee, rectus femoris spasticity
  stride_asymmetry_right / stride_asymmetry_left : shorter step on one side, uneven stride
  trunk_lean_right / trunk_lean_left            : lateral trunk lean, Trendelenburg sign
  arm_swing_reduction_right / arm_swing_reduction_left : reduced/absent arm swing, spastic arm
  cadence_reduction                             : slow walking, shuffling pace
  hip_hike_right / hip_hike_left                : pelvic hike to clear foot during swing

Compound clinical syndromes (one key applies the full pattern):
  hemiplegic_right / hemiplegic_left            : post-stroke hemiplegia (foot drop + stiff knee + arm posture + hip hike + trunk lean)
  parkinsonian_shuffle                          : Parkinson's gait (small steps + no arm swing + flexed posture + slow)
  crouch_gait                                   : cerebral palsy crouch (persistent knee/hip flexion)
  scissor_gait                                  : adductor spasticity (legs cross midline)
  antalgic_right / antalgic_left                : pain avoidance gait (hip/knee OA, short stance on painful side)

SEVERITY MAPPING:
  slight / mild / a little = 0.3
  moderate / some / noticeable = 0.6
  severe / strong / significant / marked = 0.9
  (no qualifier) = 0.6

SPECIAL ACTIONS:
  "more severe" / "increase" / "worse" / "stronger" → {"action": "increase_all"}
  "reset" / "normal" / "remove all" / "clear"       → {}
  "remove X" / "without X"                           → set that parameter to 0.0

EXAMPLES:
Input: "add moderate right leg limp"
Output: {"ankle_drop_right": 0.6, "stride_asymmetry_right": 0.54, "arm_swing_reduction_right": 0.42}

Input: "hemiplegic gait on the right side"
Output: {"hemiplegic_right": 0.7}

Input: "Parkinson disease walking"
Output: {"parkinsonian_shuffle": 0.7}

Input: "crouch gait like cerebral palsy"
Output: {"crouch_gait": 0.7}

Input: "walking like scissors, legs crossing"
Output: {"scissor_gait": 0.6}

Input: "stiff right knee with hip hike to clear foot"
Output: {"knee_stiffness_right": 0.7, "hip_hike_right": 0.6}

Input: "pain in left hip, avoids putting weight on it"
Output: {"antalgic_left": 0.6}

Input: "severe right foot drop with circumduction"
Output: {"ankle_drop_right": 0.9, "hip_hike_right": 0.7}

Input: "make it more severe"
Output: {"action": "increase_all"}

Input: "remove the trunk lean"
Output: {"trunk_lean_right": 0.0, "trunk_lean_left": 0.0}

Input: "reset to normal walking"
Output: {}

Now extract parameters for:
Input: "{prompt}"
Output:"""


def _rule_based_impairment_parse(text: str) -> dict:
    """Keyword-based fallback when LLM is unavailable. Covers all v3 parameters."""
    t        = text.lower()
    params   = {}
    severity = (0.3 if any(w in t for w in ["slight", "mild", "little", "minor"]) else
                0.9 if any(w in t for w in ["severe", "strong", "significant", "heavy", "marked"]) else 0.6)
    side = "right" if "right" in t else "left" if "left" in t else None

    # ── Compound syndromes (check first — highest specificity) ───────────────
    if "hemipleg" in t or "hemiparesis" in t or "stroke" in t:
        s = side or "right"
        params[f"hemiplegic_{s}"] = severity
        return params

    if "parkinson" in t or "festination" in t or ("shuffle" in t and "parkin" in t):
        params["parkinsonian_shuffle"] = severity
        return params

    if "crouch" in t or ("cerebral palsy" in t and "crouch" in t):
        params["crouch_gait"] = severity
        return params

    if "scissor" in t or "scissors" in t or "adductor spasticity" in t:
        params["scissor_gait"] = severity
        return params

    if "antalgic" in t or "pain avoidance" in t or ("pain" in t and ("hip" in t or "knee" in t)):
        s = side or "right"
        params[f"antalgic_{s}"] = severity
        return params

    # ── Primitive parameters ─────────────────────────────────────────────────
    if any(w in t for w in ["foot drop", "drop foot", "footdrop", "limp", "drag", "equinus"]):
        s = side or "right"
        params[f"ankle_drop_{s}"]          = severity
        params[f"stride_asymmetry_{s}"]    = round(severity * 0.8, 2)
        params[f"arm_swing_reduction_{s}"] = round(severity * 0.6, 2)

    if any(w in t for w in ["stiff knee", "knee stiff", "stiff leg", "rigid knee", "rectus femoris"]):
        s = side or "right"
        params[f"knee_stiffness_{s}"]   = severity
        params[f"hip_hike_{s}"]         = round(severity * 0.7, 2)

    if any(w in t for w in ["hip hike", "pelvic hike", "circumduction", "hitch"]):
        s = side or "right"
        params[f"hip_hike_{s}"]         = severity

    if any(w in t for w in ["trunk lean", "trendelenburg", "leans to", "tilts to"]) and side:
        params[f"trunk_lean_{side}"]    = severity
    elif any(w in t for w in ["trunk lean", "lateral lean"]):
        params["trunk_lean_right"]      = round(severity * 0.7, 2)

    if any(w in t for w in ["arm swing", "reduced arm", "no arm", "spastic arm"]):
        s = side or "right"
        params[f"arm_swing_reduction_{s}"] = severity

    if any(w in t for w in ["slow", "shuffle", "shuffling", "cadence", "bradykinesia"]):
        params["cadence_reduction"]     = min(0.7, round(severity * 0.8, 2))

    if any(w in t for w in ["asymmetric", "uneven step", "shorter step"]):
        s = side or "right"
        params[f"stride_asymmetry_{s}"] = severity

    # ── Remove operations ────────────────────────────────────────────────────
    if any(w in t for w in ["remove", "without", "no more", "clear", "reset"]):
        if "trunk lean" in t or "lean" in t:
            params.update({"trunk_lean_right": 0.0, "trunk_lean_left": 0.0})
        if "limp" in t or "drop" in t:
            s = side or "right"
            params[f"ankle_drop_{s}"]       = 0.0
            params[f"stride_asymmetry_{s}"] = 0.0
        if "arm" in t:
            s = side or "right"
            params[f"arm_swing_reduction_{s}"] = 0.0
        if not params:
            return {}  # full reset

    return params


def _parse_impairment_prompt(user_prompt: str, current_state: dict) -> dict:
    """
    Use LLM (Vertex → OpenAI → Ollama) or rules to extract impairment parameters.
    Merges result with current session state. Returns new merged state.
    """
    import json as _json

    llm_prompt = _IMPAIRMENT_PARSE_PROMPT.replace("{prompt}", user_prompt)
    raw = None

    if _USE_VERTEX:
        try:
            raw = _vertex_text(llm_prompt, "impairment_parse")
        except Exception as e:
            _debug(f"[ImpairmentParser] VertexAI failed: {e}")

    if not raw and _USE_OPENAI:
        try:
            raw = _openai_text(llm_prompt, settings.openai_model_rewrite, 300, "impairment_parse")
        except Exception as e:
            _debug(f"[ImpairmentParser] OpenAI failed: {e}")

    if not raw:
        try:
            model = _pick_ollama_model(settings.ollama_model_rewrite)
            raw   = _ollama_text(llm_prompt, model, 300, "impairment_parse")
        except Exception as e:
            _debug(f"[ImpairmentParser] Ollama failed: {e}")

    # Parse JSON from LLM output
    new_params = {}
    if raw:
        m = re.search(r"\{.*\}", (raw or "").strip(), re.DOTALL)
        if m:
            try:
                new_params = _json.loads(m.group(0))
            except Exception:
                pass

    # If LLM failed or returned nothing usable — use rule-based
    if not new_params and not raw:
        _debug("[ImpairmentParser] Using rule-based fallback")
        new_params = _rule_based_impairment_parse(user_prompt)

    _debug(f"[ImpairmentParser] extracted: {new_params}")

    # Handle special actions
    action = new_params.pop("action", None)
    merged = dict(current_state)

    if action == "increase_all":
        for k, v in merged.items():
            merged[k] = round(min(1.0, float(v) * 1.3), 2)
        _debug(f"[ImpairmentParser] increased all by 30%: {merged}")
    elif not new_params and action is None:
        # Empty dict = reset
        merged = {}
        _debug("[ImpairmentParser] reset to normal")
    else:
        for k, v in new_params.items():
            if float(v) == 0.0:
                merged.pop(k, None)
            else:
                merged[k] = round(max(0.0, min(1.0, float(v))), 2)

    _debug(f"[ImpairmentParser] final state: {merged}")
    return merged


# _blender_cmd_with_bvh removed — using _blender_cmd directly


@app.post(
    "/refine_motion/",
    tags=["generation"],
    summary="Iteratively refine motion by adding/modifying clinical gait impairments",
    response_model=RefineMotionResponse,
)
async def refine_motion(
    request:      RefineMotionRequest,
    video_render: str = Query("false", description="Render video (true/false)"),
):
    sid     = request.session_id.strip()
    session = _get_session(sid)

    # Validate base BVH exists
    base_bvh = session.get("base_bvh")
    if not base_bvh or not os.path.exists(base_bvh):
        raise HTTPException(
            status_code=400,
            detail=f"No base motion for session '{sid}'. Call /gen_text2motion/ first."
        )

    # Translate prompt if needed
    english_prompt = translate_to_english(request.prompt)
    _debug(f"[Refine] session={sid} prompt={english_prompt!r}")

    # Write video_title.txt for Blender — shows original prompt + active impairments
    def _write_video_title(orig_prompt: str, imp_state: dict) -> None:
        if not imp_state:
            title = orig_prompt
        else:
            labels = {
                "ankle_drop_right":          "R ankle drop",
                "ankle_drop_left":           "L ankle drop",
                "knee_stiffness_right":      "R knee stiff",
                "knee_stiffness_left":       "L knee stiff",
                "stride_asymmetry_right":    "R stride asym",
                "stride_asymmetry_left":     "L stride asym",
                "trunk_lean_right":          "R trunk lean",
                "trunk_lean_left":           "L trunk lean",
                "arm_swing_reduction_right": "R arm swing↓",
                "arm_swing_reduction_left":  "L arm swing↓",
                "cadence_reduction":         "Cadence↓",
            }
            parts = [f"{labels.get(k,k)}={v:.1f}" for k,v in imp_state.items()]
            title = f"{orig_prompt} [{', '.join(parts)}]"
        try:
            with open("video_title.txt", "w", encoding="utf-8") as f:
                f.write(title)
            _debug(f"[Refine] Wrote video title: {title[:80]}")
        except Exception as e:
            _debug(f"[Refine] Could not write video_title.txt: {e}")

    # Parse + merge impairment parameters with current session state
    current     = session.get("impairments", {})
    new_state   = _parse_impairment_prompt(english_prompt, current)
    session["impairments"] = new_state

    # Update video title for Blender to show current impairment state
    _write_video_title(request.prompt, new_state)

    # Save impaired BVH to the standard location so Blender video render works
    # bvh_folder/bvh_0_out.bvh is what bvh2fbx.py always reads
    os.makedirs("impaired_bvh_folder", exist_ok=True)
    os.makedirs("bvh_folder", exist_ok=True)
    archive_bvh = f"impaired_bvh_folder/session_{sid}_impaired.bvh"  # keep a copy
    output_bvh  = "bvh_folder/bvh_0_out.bvh"                         # standard path for Blender

    try:
        def run_refinement():
            if new_state:
                # Apply clinical modifications to the ORIGINAL clean base BVH
                from bvh_impairment_engine import apply_impairment
                apply_impairment(base_bvh, new_state, output_bvh)
                shutil.copy(output_bvh, archive_bvh)  # keep session archive copy
                _debug(f"[Refine] Applied {len(new_state)} impairment parameters -> {output_bvh}")
            else:
                # No impairments — restore the original base BVH
                shutil.copy(base_bvh, output_bvh)
                shutil.copy(base_bvh, archive_bvh)
                _debug("[Refine] No impairments — restored base BVH")

            # Retarget using standard Blender command — impaired BVH is already at bvh_folder/bvh_0_out.bvh
            cmd = _blender_cmd(video_render)
            _debug(f"[Refine] Launching Blender: {cmd}")
            result = subprocess.run(cmd, shell=True)

            # Blender sometimes crashes with EXCEPTION_ACCESS_VIOLATION in python311.dll
            # during interpreter shutdown AFTER the render completes successfully.
            # Check if output files exist rather than relying on return code.
            fbx_ok   = os.path.exists("./fbx_folder/bvh_0_out.fbx")
            video_ok = os.path.exists("./video_result/Final_Fbx_Mesh_Animation.mp4")

            if result.returncode != 0:
                if fbx_ok or video_ok:
                    _debug(f"[Refine] Blender exited with code {result.returncode} "
                           f"but outputs exist (fbx={fbx_ok}, video={video_ok}) — treating as success")
                else:
                    raise subprocess.CalledProcessError(result.returncode, cmd)

        await run_in_threadpool(run_refinement)

        n = len(new_state)
        return {
            "status":          "success",
            "session_id":      sid,
            "prompt":          request.prompt,
            "impairment_state": new_state,
            "message":         (f"Applied {n} impairment parameter(s). "
                                f"Download updated FBX from /download_fbx/" if n
                                else "Reset to normal walking. Download FBX from /download_fbx/"),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Refinement failed: {e}")


@app.get(
    "/session_state/{session_id}",
    tags=["generation"],
    summary="Get current impairment state and base BVH info for a session",
)
async def get_session_state(session_id: str):
    session = _get_session(session_id)
    return {
        "session_id":      session_id,
        "has_base_motion": bool(session.get("base_bvh")),
        "base_bvh":        session.get("base_bvh"),
        "impairments":     session.get("impairments", {}),
        "impairment_count": len(session.get("impairments", {})),
    }


# =============================================================================
# SECTION 14 — Animation Selection & Compare Endpoints
# =============================================================================

animation_state = {"model": "pretrained", "index": 0}
compare_state   = {"index": 1, "triggered": False}


@app.get("/set_animation", tags=["animation"],
         summary="Set which pre-generated animation should be served next",
         response_model=SetAnimationResponse)
def set_animation(
    model: str = Query(..., enum=["pretrained", "retrained"]),
    anim:  int = Query(..., ge=1, le=5),
):
    animation_state["model"] = model
    animation_state["index"] = anim
    return {"status": "updated", "server_status": server_status_value, **animation_state}


@app.get("/get_animation", tags=["animation"],
         summary="Download the selected pre-generated animation ZIP")
def get_animation():
    model    = animation_state["model"]
    index    = animation_state["index"]
    filename = f"{model}_{index}.zip"
    zip_path = os.path.join("pregen_animation", filename)
    if os.path.isfile(zip_path):
        return FileResponse(zip_path, media_type="application/octet-stream", filename=filename)
    raise HTTPException(status_code=404, detail=f"File '{filename}' not found")


@app.post("/trigger_compare/", tags=["animation"],
          summary="Trigger compare mode with an index (one-shot)",
          response_model=CompareTriggerResponse)
async def trigger_compare(index: int = Query(..., ge=1, le=5)):
    compare_state["index"]   = index
    compare_state["triggered"] = True
    return {"status": "compare triggered", "index": index}


@app.get("/check_compare/", tags=["animation"],
         summary="Check compare trigger state (returns compare once, then resets to idle)")
async def check_compare():
    if compare_state["triggered"]:
        compare_state["triggered"] = False
        return {"status": "compare", "index": compare_state["index"]}
    return {"status": "idle"}


@app.get("/check_animation_state/", tags=["animation"],
         summary="Get current animation model state",
         response_model=AnimationStateResponse)
def check_animation_state():
    return {"model": animation_state["model"]}


@app.get("/get_animation_selection", tags=["animation"],
         summary="Get current selected model name and animation index",
         response_model=AnimationSelectionResponse)
def get_animation_selection():
    model = animation_state.get("model", "pretrained")
    index = max(1, min(5, animation_state.get("index", 1)))
    return {"model": model, "animation_id": index}


# =============================================================================
# SECTION 14 — Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    print("[Startup] Launching FastAPI on 0.0.0.0:8000 ...")
    uvicorn.run("app_server:app", host="0.0.0.0", port=8000, reload=True)