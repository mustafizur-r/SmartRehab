from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field
import platform
import subprocess
import re
import os
import unicodedata
from starlette.concurrency import run_in_threadpool
from langdetect import detect, DetectorFactory

from dotenv import load_dotenv
from openai import OpenAI

DetectorFactory.seed = 0  # makes detection deterministic

# ---------------------------------------------------------------------
# Load .env and OpenAI client
# If OPENAI_API_KEY is missing, fallback to Ollama local models.
# ---------------------------------------------------------------------
load_dotenv()


def _debug(msg: str) -> None:
    print(msg, flush=True)


class Settings(BaseModel):
    # Optional: if missing, we will use Ollama fallback
    openai_api_key: str = Field("", alias="OPENAI_API_KEY")

    openai_model_translate: str = Field("gpt-4o-mini", alias="OPENAI_MODEL_TRANSLATE")
    openai_model_rewrite: str = Field("gpt-4o-mini", alias="OPENAI_MODEL_REWRITE")
    openai_base_url: str = Field("", alias="OPENAI_BASE_URL")
    openai_timeout_seconds: int = Field(60, alias="OPENAI_TIMEOUT_SECONDS")

    # Ollama preferences for fallback
    ollama_model_translate: str = Field("llama3", alias="OLLAMA_MODEL_TRANSLATE")
    ollama_model_rewrite: str = Field("llama3", alias="OLLAMA_MODEL_REWRITE")


settings = Settings.model_validate(os.environ)

_USE_OPENAI = bool(settings.openai_api_key and settings.openai_api_key.strip())

client = None
if _USE_OPENAI:
    # FIX 1: Create client first, then check capabilities
    client = OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url or None,
        timeout=settings.openai_timeout_seconds,
    )

    has_responses = hasattr(client, "responses")
    has_chat = hasattr(client, "chat") and hasattr(client.chat, "completions")
    _debug(f"[Startup] OpenAI capabilities: responses={has_responses}, chat.completions={has_chat}")
    _debug(f"[Startup] OPENAI_API_KEY present=True len={len(settings.openai_api_key.strip())}")
    _debug(f"[Startup] OpenAI enabled (OPENAI_API_KEY found). base_url={settings.openai_base_url or '(default)'}")
    _debug(
        f"[Startup] OpenAI models: translate={settings.openai_model_translate}, rewrite={settings.openai_model_rewrite}"
    )
else:
    _debug("[Startup] OPENAI_API_KEY not found. Using Ollama fallback.")
    _debug(f"[Startup] Ollama preferred models: translate={settings.ollama_model_translate}, rewrite={settings.ollama_model_rewrite}")


def _openai_text(prompt: str, model: str, max_output_tokens: int, purpose: str) -> str:
    """
    Calls OpenAI using Responses API when available, otherwise falls back to Chat Completions.

    Returns:
        Plain text (best-effort). Strips common wrappers and tries to avoid returning accidental
        non-answer content if the model leaks meta text.
    """
    import re as _re

    if client is None:
        raise RuntimeError("OpenAI client is not configured.")

    _debug(f"[LLM] provider=OpenAI purpose={purpose} model={model} max_output_tokens={max_output_tokens}")

    system_msg = "You follow instructions exactly and return plain text only."

    def _normalize(s: str) -> str:
        s = (s or "").replace("\r", " ").strip()
        s = _re.sub(r"\s{2,}", " ", s)
        return s.strip(" \"'“”‘’`")

    def _strip_fences(s: str) -> str:
        s = (s or "").strip()
        # Remove ```...``` wrappers if the model returns them
        s = _re.sub(r"^```(?:text)?\s*", "", s, flags=_re.IGNORECASE)
        s = _re.sub(r"\s*```$", "", s)
        return s.strip()

    def _looks_like_meta(s: str) -> bool:
        low = (s or "").lower()
        triggers = [
            "okay, let's", "let's tackle", "the user wants", "first, i need",
            "analysis", "thinking", "reasoning", "strict format contract",
        ]
        return any(t in low for t in triggers)

    def _salvage(s: str) -> str:
        s = (s or "").strip()
        if not s:
            return ""
        # If meta leaked, prefer the last paragraph-like chunk with punctuation.
        chunks = [c.strip() for c in _re.split(r"\n\s*\n", s) if c.strip()]
        if chunks:
            candidates = [c for c in chunks if _re.search(r"[.!?]", c)]
            if candidates:
                return candidates[-1]
            return chunks[-1]
        return s

    # -------------------------
    # Responses API (preferred)
    # -------------------------
    try:
        if hasattr(client, "responses"):
            resp = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_output_tokens=int(max_output_tokens),
            )

            out = getattr(resp, "output_text", "") or ""
            out = _normalize(_strip_fences(out))
            if out:
                if _looks_like_meta(out):
                    out = _normalize(_strip_fences(_salvage(out)))
                return out

            # Fallback parse if output_text is empty
            chunks = []
            for item in (getattr(resp, "output", None) or []):
                for c in (getattr(item, "content", None) or []):
                    if getattr(c, "type", "") == "output_text":
                        chunks.append(getattr(c, "text", "") or "")
            out2 = _normalize(_strip_fences(" ".join(chunks)))
            if _looks_like_meta(out2):
                out2 = _normalize(_strip_fences(_salvage(out2)))
            return out2
    except Exception as e:
        _debug(f"[LLM] Responses API failed, trying chat.completions. error={e}")

    # -------------------------
    # Chat Completions fallback
    # -------------------------
    if hasattr(client, "chat") and hasattr(client.chat, "completions"):
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=int(max_output_tokens),
        )
        out = (resp.choices[0].message.content or "").strip()
        out = _normalize(_strip_fences(out))
        if _looks_like_meta(out):
            out = _normalize(_strip_fences(_salvage(out)))
        return out

    raise RuntimeError("OpenAI SDK does not support responses or chat.completions in this environment.")




# ---------------------------------------------------------------------
# Swagger / OpenAPI metadata
# ---------------------------------------------------------------------
tags_metadata = [
    {"name": "root", "description": "Landing page and basic service info."},
    {"name": "status", "description": "Server state endpoints used by clients."},
    {"name": "prompts", "description": "Prompt IO endpoints."},
    {"name": "downloads", "description": "Download generated assets (FBX/ZIP/Video)."},
    {"name": "generation", "description": "Text-to-motion generation pipeline."},
    {"name": "animation", "description": "Pre-generated animation selection and compare mode."},
]

app = FastAPI(
    title="GaitSimPT-Codes (Dockerized)",
    description=(
        "A Dockerized FastAPI service for generating and retargeting patient-specific gait motions. "
        "This service exposes endpoints for prompt handling, motion generation, and asset downloads.\n\n"
        "Swagger UI: /docs\n"
        "ReDoc: /redoc"
    ),
    version="1.0.0",
    openapi_tags=tags_metadata,
)

text_prompt_global = None
server_status_value = 4  # Default to initial status


# ---------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------
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


class GenText2MotionResponse(BaseModel):
    status: str
    message: str
    original_prompt: str
    english_prompt: str
    expressive_prompt: str


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


# ---------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------
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
        body {
          margin: 0;
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
          background: #f5f7fa;
          color: #333;
        }
        .container {
          max-width: 800px;
          margin: 4rem auto;
          background: #ffffff;
          padding: 2.5rem;
          border-radius: 8px;
          box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        }
        h1 {
          margin-top: 0;
          font-size: 2rem;
          color: #222;
        }
        p {
          line-height: 1.6;
        }
        .button {
          display: inline-block;
          margin: 1rem 0;
          padding: 0.6rem 1.2rem;
          background-color: #007ACC;
          color: #fff;
          text-decoration: none;
          font-weight: 500;
          border-radius: 4px;
        }
        .button:hover {
          background-color: #005A9E;
        }
        footer {
          margin-top: 3rem;
          font-size: 0.9rem;
          color: #666;
          border-top: 1px solid #e1e4e8;
          padding-top: 1rem;
        }
        footer a {
          color: #007ACC;
          text-decoration: none;
        }
        footer a:hover {
          text-decoration: underline;
        }
      </style>
    </head>
    <body>
      <div class="container">
        <h1>GaitSimPT-Codes (Dockerized)</h1>
        <p>
          A Dockerized FastAPI service for generating and retargeting
          patient-specific gait motions. Combines state-of-the-art text-to-motion
          models, Blender automation, and the KeeMap rig-retargeting addon.
        </p>
        <a class="button"
           href="https://mustafizur-r.github.io/SmartRehab/"
           target="_blank" rel="noopener">
          Project Homepage
        </a>
        <p>Start by calling the <code>/gen_text2motion/</code> endpoint.</p>

        <footer>
          <p><strong>Prepared by:</strong> Md Mustafizur Rahman</p>
          <p>
            Master's student, Interactive Media Design Laboratory,<br>
            Division of Information Science, NAIST
          </p>
          <p>
            Email:
            <a href="mailto:mustafizur.cd@gmail.com">
              mustafizur.cd@gmail.com
            </a>
          </p>
        </footer>
      </div>
    </body>
    </html>
    """


# ---------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------
@app.get(
    "/server_status/",
    tags=["status"],
    summary="Get server status message",
    response_model=ServerStatusMessageResponse,
)
async def get_server_status():
    if server_status_value == 0:
        return {"message": "Server is not running."}
    elif server_status_value == 1:
        return {"message": "Server is running."}
    elif server_status_value == 2:
        return {"message": "Initial server status."}
    elif server_status_value == 3:
        return {"message": "compare"}
    else:
        return {"message": "Initializing"}


@app.post(
    "/set_server_status/",
    tags=["status"],
    summary="Set server status",
    response_model=ServerStatusMessageResponse,
)
async def set_server_status(server_status: ServerStatus):
    status = server_status.status
    print(f"Received status: {status}")
    global server_status_value
    if status in (0, 1, 2, 3):
        server_status_value = status
        if status == 0:
            return {"message": "Server status set to 'not running'."}
        elif status == 1:
            return {"message": "Server status set to 'running'."}
        elif status == 2:
            return {"message": "Server status set to 'initial status'."}
        elif status == 3:
            return {"message": "Server status set to 'compare'."}
    raise HTTPException(status_code=400, detail="Invalid status value. Please provide 0, 1, or 2,3.")


@app.get(
    "/set_status",
    tags=["status"],
    summary="Set server status (legacy)",
    response_model=StatusResponse,
)
def set_status(status: int = Query(..., ge=0, le=5, description="0..5 legacy status range")):
    global server_status_value
    server_status_value = status
    return {"status": "updated", "server_status": server_status_value}


# ---------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------
@app.get(
    "/get_prompts/",
    tags=["prompts"],
    summary="Read latest input prompt from input.txt",
    response_model=GetPromptsResponse,
)
async def get_prompts():
    try:
        file_path = "input.txt"
        with open(file_path, "r", encoding="utf-8") as file:
            prompt = file.read()
            return {"status": "success", "prompt": prompt}
    except Exception as e:
        raise HTTPException(status_code=500, detail={"status": "failed", "error": str(e)})


@app.post(
    "/input_prompts/",
    tags=["prompts"],
    summary="Write prompt to input.txt",
    response_model=MessageResponse,
)
async def input_prompts(prompt: Prompt):
    try:
        file_path = "input.txt"
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(prompt.prompt)
        return {"message": "Prompt saved successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving prompt: {e}")


# ---------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------
@app.get(
    "/download_fbx/",
    tags=["downloads"],
    summary="Download an FBX file from fbx_folder",
)
async def download_bvh(filename: str = Query(..., description="FBX filename inside fbx_folder")):
    bvh_folder = "fbx_folder"
    filepath = os.path.join(bvh_folder, filename)

    if os.path.exists(filepath):
        return FileResponse(filepath, media_type="application/octet-stream", filename=filename)
    raise HTTPException(status_code=404, detail="File not found")


VIDEO_PATH = "./video_result/Final_Fbx_Mesh_Animation.mp4"


@app.get(
    "/download_video",
    tags=["downloads"],
    summary="Download rendered MP4 video",
)
async def download_video():
    if not os.path.exists(VIDEO_PATH):
        return {"error": "Video not found."}
    return FileResponse(
        path=VIDEO_PATH,
        filename="Final_Fbx_Mesh_Animation.mp4",
        media_type="video/mp4",
    )


@app.get(
    "/download_zip/",
    tags=["downloads"],
    summary="Download a ZIP file from fbx_zip_folder",
)
async def download_fbx(filename: str = Query(..., description="ZIP filename inside fbx_zip_folder")):
    fbx_zip_folder = "fbx_zip_folder"
    zip_filepath = os.path.join(fbx_zip_folder, filename)
    if os.path.exists(zip_filepath):
        return FileResponse(zip_filepath, media_type="application/octet-stream", filename=filename)
    raise HTTPException(status_code=404, detail="File not found")


# =====================================================
# Ollama helpers (used only when OPENAI_API_KEY is missing)
# =====================================================
def _pick_ollama_model(preferred: str) -> str:
    try:
        import ollama
        resp = ollama.list()

        # Support both dict and object return types
        models = []
        if isinstance(resp, dict):
            models = resp.get("models", []) or []
        elif hasattr(resp, "models"):
            models = getattr(resp, "models") or []

        have = []
        for m in models:
            if isinstance(m, dict):
                have.append(m.get("model", ""))
            else:
                have.append(getattr(m, "model", "") or "")

        have = [x for x in have if x]
        _debug(f"[Ollama] available_models={have}")

        qwen3 = [m for m in have if m.lower().startswith("qwen3")]
        if qwen3:
            qwen3.sort(reverse=True)
            chosen = qwen3[0]
            _debug(f"[Ollama] chosen_model={chosen} (from qwen3 list)")
            return chosen

        chosen = preferred or (have[0] if have else "llama3")
        _debug(f"[Ollama] chosen_model={chosen} (fallback)")
        return chosen
    except Exception as e:
        _debug(f"[Ollama] list failed: {e}")
        return preferred or "llama3"


def _ollama_text(prompt: str, model: str, num_predict: int, purpose: str) -> str:
    import re as _re
    import json
    import ollama

    _debug(f"[LLM] provider=Ollama purpose={purpose} model={model} num_predict={num_predict}")

    # Prefer JSON mode so the model cannot easily spill reasoning into the main text.
    system_msg_json = (
        "Return ONLY a valid JSON object with exactly one key: text. "
        "Do not include any other keys. "
        "Do not include analysis or thinking. "
        "The value of text must be the final rewritten paragraph as plain English text."
    )

    system_msg_text = (
        "Return ONLY the final rewritten paragraph as plain text. "
        "Do not output analysis or thinking. "
        "Never return empty output."
    )

    def _normalize(s: str) -> str:
        s = (s or "").replace("\r", " ").replace("\n", " ")
        s = _re.sub(r"\s{2,}", " ", s).strip(" \"'“”‘’")
        return s

    def _safe_json_loads(s: str) -> dict:
        """
        Qwen/Ollama sometimes returns extra whitespace or accidental surrounding text.
        Try strict JSON first; if it fails, attempt to extract the first {...} block.
        """
        s = (s or "").strip()
        if not s:
            return {}
        try:
            return json.loads(s)
        except Exception:
            m = _re.search(r"\{.*\}", s, flags=_re.DOTALL)
            if not m:
                return {}
            try:
                return json.loads(m.group(0))
            except Exception:
                return {}

    def _strip_common_wrappers(s: str) -> str:
        # Remove common wrappers like ```json ... ``` or ``` ... ```
        s = (s or "").strip()
        s = _re.sub(r"^```(?:json)?\s*", "", s, flags=_re.IGNORECASE)
        s = _re.sub(r"\s*```$", "", s)
        return s.strip()

    def _extract_best_paragraph(s: str) -> str:
        s = (s or "").strip()
        if not s:
            return ""

        # Split by blank lines, take last chunk that looks like real sentences.
        chunks = [c.strip() for c in _re.split(r"\n\s*\n", s) if c.strip()]
        if chunks:
            candidates = [c for c in chunks if _re.search(r"[.!?]", c)]
            if candidates:
                return candidates[-1].strip()

        # Fallback: last few non-empty lines
        s2 = s.replace("\r", "\n")
        lines = [ln.strip() for ln in s2.split("\n") if ln.strip()]
        tail = " ".join(lines[-10:]).strip()
        return tail

    def _looks_like_reasoning(s: str) -> bool:
        low = (s or "").lower()
        triggers = [
            "okay, let's", "let's tackle", "the user wants", "first, i need", "hard rules",
            "i need to parse", "analysis", "thinking", "reasoning", "step by step",
            "here's how", "i will", "we should",
        ]
        return any(t in low for t in triggers)

    def _remove_reasoning_headers(s: str) -> str:
        """
        If reasoning leaks, often the final answer is after markers like 'Final:'.
        Try to cut to the part after these markers; otherwise use best paragraph.
        """
        s = (s or "").strip()
        if not s:
            return ""
        markers = [
            r"\bfinal\s*:\s*",
            r"\banswer\s*:\s*",
            r"\boutput\s*:\s*",
        ]
        for pat in markers:
            m = _re.search(pat, s, flags=_re.IGNORECASE)
            if m:
                return s[m.end():].strip()
        return _extract_best_paragraph(s)

    def _call_ollama(system_msg: str, use_json: bool) -> str:
        kwargs = {}
        if use_json:
            kwargs["format"] = "json"

        resp = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            options={
                "temperature": 0.0,
                "top_p": 0.9,
                "num_ctx": 4096,
                "num_predict": int(num_predict),
                # Optional: if your Ollama build supports it, you can add:
                # "stop": ["\n\n\n", "</final>", "<analysis>", "Final:"],
            },
            **kwargs,
        )

        msg = getattr(resp, "message", None)
        if msg is None:
            return ""
        content = (getattr(msg, "content", "") or "").strip()
        thinking = (getattr(msg, "thinking", "") or "").strip()
        return content or thinking or ""

    # -------------------------
    # 1) JSON mode first
    # -------------------------
    try:
        raw = _call_ollama(system_msg_json, use_json=True)
        raw = _strip_common_wrappers(raw)
        data = _safe_json_loads(raw)
        out = _normalize(data.get("text", "") if isinstance(data, dict) else "")
        if out:
            # Guard: if the model mistakenly put reasoning into "text", salvage.
            if _looks_like_reasoning(out):
                out = _normalize(_remove_reasoning_headers(out))
            return out
    except Exception as e:
        _debug(f"[Ollama] json mode failed, fallback to text mode. error={e}")

    # -------------------------
    # 2) Plain text fallback
    # -------------------------
    try:
        raw = _call_ollama(system_msg_text, use_json=False)
    except Exception as e:
        _debug(f"[Ollama] chat failed: {e}")
        return ""

    raw = _strip_common_wrappers(raw)
    out = raw

    # If reasoning leaked, try to cut it down.
    if _looks_like_reasoning(out):
        out = _remove_reasoning_headers(out)

    out = _normalize(out)
    return out








# =====================================================
# 1) TRANSLATION FUNCTION (Skip if input is English; OpenAI, else Ollama fallback)
# =====================================================
def translate_to_english(raw_text: str) -> str:
    import re as _re
    import unicodedata

    cleaned = unicodedata.normalize("NFKC", raw_text).strip()
    original_unchanged = cleaned

    try:
        lang = detect(cleaned) or ""
        if lang.lower().startswith("en"):
            _debug("[Translator] Detected English input, skipping translation.")
            return original_unchanged
    except Exception:
        _debug("[Translator] Language detection failed; attempting translation.")

    if _USE_OPENAI:
        try:
            tmpl = (
                "You are a professional Japanese/Chinese/English medical translator. "
                "Translate the user's text into natural, idiomatic ENGLISH only.\n"
                "Preserve clinical/anatomical terms and laterality precisely. "
                "Do not add or omit information. No notes or headings.\n\n"
                f"Text:\n{cleaned}"
            )
            out = _openai_text(
                prompt=tmpl,
                model=settings.openai_model_translate,
                max_output_tokens=1024,
                purpose="translate",
            )
            out = _re.sub(r"\s{2,}", " ", out.replace("\n", " ")).strip(" '\"“”‘’`")

            if _re.search(r"[\u3040-\u30FF\u4E00-\u9FFF]", out):
                strict = (
                    "Translate STRICTLY into ENGLISH ONLY. No headings, no notes, no quotes, no markdown. "
                    "Return only fluent English sentences.\n\n" + cleaned
                )
                out2 = _openai_text(
                    prompt=strict,
                    model=settings.openai_model_translate,
                    max_output_tokens=1024,
                    purpose="translate_strict",
                )
                out2 = _re.sub(r"\s{2,}", " ", out2.replace("\n", " ")).strip(" '\"“”‘’`")
                if out2:
                    out = out2

            _debug("[Translator] provider=OpenAI success")
            return out or original_unchanged
        except Exception as e:
            _debug(f"[Translator] provider=OpenAI failed, fallback to Ollama. error={e}")

    try:
        model = _pick_ollama_model(settings.ollama_model_translate)
        tmpl = (
            "You are a professional Japanese/Chinese/English medical translator. "
            "Translate the user's text into natural, idiomatic ENGLISH only.\n"
            "Preserve clinical/anatomical terms and laterality precisely. "
            "Do not add or omit information. No notes or headings.\n\n"
            f"Text:\n{cleaned}"
        )
        out = _ollama_text(tmpl, model=model, num_predict=1024, purpose="translate")

        if _re.search(r"[\u3040-\u30FF\u4E00-\u9FFF]", out):
            strict = (
                "Translate STRICTLY into ENGLISH ONLY. No headings, no notes, no quotes, no markdown. "
                "Return only fluent English sentences.\n\n" + cleaned
            )
            out2 = _ollama_text(strict, model=model, num_predict=1024, purpose="translate_strict")
            if out2:
                out = out2

        _debug("[Translator] provider=Ollama success")
        return out or original_unchanged
    except Exception as e:
        _debug(f"[Translator] provider=Ollama failed. error={e}")
        return original_unchanged

#  =====================================================
# 2) PROMPT REWRITE FUNCTION (Therapist-style, asymmetry-aware; OpenAI, else Ollama fallback)
# # =====================================================
def rewrite_prompt_auto(raw_text: str) -> str:
    import re
    import unicodedata
    from typing import Optional, Tuple

    text = unicodedata.normalize("NFKC", (raw_text or "")).strip()
    if not text:
        return (
            "The person walks forward with a steady posture. "
            "The legs alternate as the feet lift and place with consistent timing. "
            "The arms move near the sides as weight shifts from step to step. "
            "The torso stays mostly upright with small balance adjustments. "
            "This pattern repeats for several steps."
        )

    def _has_word(s: str, w: str) -> bool:
        return bool(re.search(rf"\b{re.escape(w)}\b", s, flags=re.IGNORECASE))

    def _is_step_motion(s: str) -> bool:
        return bool(
            re.search(
                r"\b(walk|walking|step|steps|stepping|stride|gait|march|shuffle|pace|stroll)\b",
                s,
                flags=re.IGNORECASE,
            )
        )

    def _split_sentences(par: str):
        return [p.strip() for p in re.split(r"(?<=[.!?])\s+", (par or "").strip()) if p.strip()]

    def _count_sentences(par: str) -> int:
        return len(_split_sentences(par))

    def _contains_numbers_or_units(par: str) -> bool:
        if re.search(r"\d", par or ""):
            return True
        if re.search(r"\b(seconds?|meters?|metres?|degrees?|percent)\b", par or "", flags=re.IGNORECASE):
            return True
        return False

    def _ends_with_required_sentence(par: str, required: str) -> bool:
        return (par or "").strip().endswith(required)

    def _mentions_opposite_side(par: str, in_has_right: bool, in_has_left: bool) -> bool:
        p = (par or "").lower()
        if in_has_right and not in_has_left:
            return _has_word(p, "left")
        if in_has_left and not in_has_right:
            return _has_word(p, "right")
        return False

    banned_style_words = [
        "fluid", "smooth", "smoothly", "grace", "graceful", "effortless", "effortlessly",
        "elegant", "elegantly", "rhythmic", "rhythmically", "seamless", "seamlessly",
    ]
    banned_intention_words = [
        "focus", "focused", "intends", "intention", "decides", "decision", "tries to", "attempts to",
        "deliberate", "deliberately",
    ]

    def _contains_banned(par: str) -> bool:
        low = (par or "").lower()
        for w in banned_style_words:
            if re.search(rf"\b{re.escape(w)}\b", low):
                return True
        for w in banned_intention_words:
            if w in low:
                return True
        return False

    step_motion = _is_step_motion(text)
    repetition_line = "This pattern repeats for several steps." if step_motion else "The motion repeats for several cycles."

    in_has_right = _has_word(text, "right")
    in_has_left = _has_word(text, "left")

    text_l = text.lower()
    mentions_arc_like = bool(re.search(r"\b(arc|arcing|circular|circle|circles|curved|curve|around)\b", text_l))
    explicitly_walk_in_circles = bool(
        re.search(r"\bwalk(?:s|ing)?\s+in\s+circles\b", text_l)
        or re.search(r"\bwalk(?:s|ing)?\s+around\s+a\s+circle\b", text_l)
        or re.search(r"\bwalk(?:s|ing)?\s+in\s+a\s+circle\b", text_l)
    )

    force_straight = bool(step_motion and mentions_arc_like and not explicitly_walk_in_circles)

    forward_constraint = ""
    limb_wording_hint = ""
    if force_straight:
        forward_constraint = (
            "- Include an explicit sentence that the person walks forward in a straight line with a stable heading.\n"
            "- Do NOT describe or imply turning, circling, rotating, spinning, or moving around a loop.\n"
            "- If arc/circle/curved is mentioned, apply it ONLY to the leg or foot swing path, not the travel direction.\n"
        )
        limb_wording_hint = (
            "- Prefer wording like \"wide arc\" or \"outward arc\" for the limb path instead of \"circle\".\n"
        )

    def _has_circular_travel_risk(par: str) -> bool:
        p = (par or "").lower()
        if re.search(r"\b(turns?|turning|rotates?|rotating|spins?|spinning|circles?\s+around|moves?\s+around)\b", p):
            return True
        if force_straight:
            has_forward = bool(re.search(r"\b(forward|straight\s+line|in\s+a\s+straight\s+line|straight)\b", p))
            if not has_forward:
                return True
        return False

    def _enforce_straight_and_limb_arc(par: str) -> str:
        """
        Post-fix: make the output safer for generation.
        - If force_straight: guarantee explicit straight-line forward travel.
        - If force_straight: replace circle-phrases with arc-phrases for limb path.
        - Keep 4..7 sentences and preserve required final sentence.
        """
        par = (par or "").strip()
        if not par:
            return par

        # Remove accidental code fences
        par = re.sub(r"^```(?:text)?\s*", "", par, flags=re.IGNORECASE).strip()
        par = re.sub(r"\s*```$", "", par).strip()

        # Ensure final sentence exact
        if not par.endswith(repetition_line):
            # If model added something after repetition line, truncate after it
            idx = par.rfind(repetition_line)
            if idx != -1:
                par = par[: idx + len(repetition_line)].strip()
            else:
                par = par.rstrip(". ") + ". " + repetition_line

        if force_straight:
            sents = _split_sentences(par)
            low = par.lower()

            # Add explicit straight-line forward sentence if missing
            if not re.search(r"\b(forward|straight\s+line|in\s+a\s+straight\s+line|straight)\b", low):
                sents.insert(0, "The person walks forward in a straight line with a stable heading.")

            # Replace "circle/circular" phrases for limb path (do not change laterality)
            joined = " ".join(sents)

            joined = re.sub(
                r"\b(around\s+in\s+a\s+circle|in\s+a\s+circle|around\s+a\s+circle)\b",
                "outward in a wide arc",
                joined,
                flags=re.IGNORECASE,
            )
            joined = re.sub(
                r"\bcircular\s+motion\b",
                "wide arcing path",
                joined,
                flags=re.IGNORECASE,
            )
            joined = re.sub(
                r"\bswinging\s+([a-z\s]*?)\s+in\s+a\s+circle\b",
                r"swinging \1 outward in a wide arc",
                joined,
                flags=re.IGNORECASE,
            )

            # Restore sentences
            sents = _split_sentences(joined)

            # Keep last sentence as repetition_line
            if sents and sents[-1] != repetition_line:
                # drop any trailing sentence and enforce final
                sents = [x for x in sents if x != repetition_line]
                sents.append(repetition_line)

            # Enforce 4..7 sentences: keep first N-1 + final
            if len(sents) > 7:
                core = sents[:-1]
                core = core[:6]  # keep 6 body sentences max
                sents = core + [repetition_line]

            # If too short, pad without adding new actions
            if len(sents) < 4:
                core = sents[:-1]
                core.append("The torso stays mostly upright with small balance adjustments.")
                sents = core + [repetition_line]

            par = " ".join(sents).strip()

        # Ensure starts with "The person"
        if not par.startswith("The person"):
            par = "The person " + par.lstrip()

        return par

    def _is_valid(par: str) -> Tuple[bool, str]:
        par = (par or "").strip()
        if not par:
            return False, "empty"
        if not par.startswith("The person"):
            return False, "must_start"
        n_sent = _count_sentences(par)
        if n_sent < 4 or n_sent > 7:
            return False, "sentence_count"
        if _contains_numbers_or_units(par):
            return False, "numbers"
        if _contains_banned(par):
            return False, "banned_words"
        if _mentions_opposite_side(par, in_has_right, in_has_left):
            return False, "opposite_side"
        if not _ends_with_required_sentence(par, repetition_line):
            return False, "bad_final_sentence"
        if _has_circular_travel_risk(par):
            return False, "circular_travel_risk"
        return True, "ok"

    base_prompt = f"""
You are a motion-caption writer for a text-to-motion dataset (SnapMoGen-style).

Return ONLY the final motion description paragraph.
Do NOT output analysis, reasoning, planning, or any extra text.

Strict format contract:
- Output ENGLISH ONLY.
- Output exactly ONE paragraph.
- 4 to 7 complete sentences (no fragments).
- Present tense, third-person. The paragraph MUST start with "The person".
- The FINAL sentence MUST be EXACTLY this text: {repetition_line}
- Do NOT add any sentence after the final sentence.
- Do NOT paraphrase the final sentence.
- Do NOT include any numbers or measurements.

Anti-hallucination rules:
- Describe ONLY what is stated or directly implied by the user input.
- Do NOT add new actions, props, devices, diagnosis, or explanations.
- If the input mentions only one side (left or right), do NOT mention the other side in any form.

Travel-direction rules:
{forward_constraint}{limb_wording_hint}
Banned content:
- No style words like: {", ".join(banned_style_words)}
- No intention/mental words like: {", ".join(banned_intention_words)}

User input:
{text}
""".strip()

    retry_prompt = f"""
Correct your previous output. Return ONLY the final paragraph.

You MUST follow these constraints:
- Start with "The person"
- 4 to 7 complete sentences (no fragments)
- If the input mentions only one side, do NOT mention the opposite side
- Last sentence MUST be EXACTLY: {repetition_line}
- No numbers or measurements
- Do NOT use style words: {", ".join(banned_style_words)}
- Do NOT use intention/mental words: {", ".join(banned_intention_words)}

Travel-direction rules:
{forward_constraint}{limb_wording_hint}
User input:
{text}
""".strip()

    def _call_llm(p: str, purpose_suffix: str) -> str:
        out_local: Optional[str] = ""

        if _USE_OPENAI:
            try:
                out_local = _openai_text(
                    prompt=p,
                    model=settings.openai_model_rewrite,
                    max_output_tokens=1024,
                    purpose=f"rewrite_snapmogen_{purpose_suffix}",
                )
            except Exception as e:
                _debug(f"[Rewriter] OpenAI failed, fallback to Ollama. error={e}")
                out_local = ""

        if not out_local:
            try:
                model = _pick_ollama_model(settings.ollama_model_rewrite)
                out_local = _ollama_text(
                    p,
                    model=model,
                    num_predict=320,
                    purpose=f"rewrite_snapmogen_{purpose_suffix}",
                )
            except Exception as e:
                _debug(f"[Rewriter] Ollama failed. error={e}")
                out_local = ""

        out_local = (out_local or "").strip()
        out_local = _enforce_straight_and_limb_arc(out_local)
        return out_local

    out = _call_llm(base_prompt, "try1")
    ok, _ = _is_valid(out)
    if ok:
        return out

    out2 = _call_llm(retry_prompt, "try2")
    ok2, _ = _is_valid(out2)
    if ok2:
        return out2

    # Final guaranteed fallback
    if force_straight:
        return (
            "The person walks forward in a straight line with a stable heading. "
            "The person keeps the described arm posture while stepping forward. "
            "The leg follows an outward arcing path during the swing. "
            "The toes lightly brush the floor before foot placement. "
            f"{repetition_line}"
        )

    return (
        "The person maintains a steady posture while performing the described movement. "
        "The limbs coordinate through the action with controlled joint motion. "
        "Small balance adjustments occur through the torso as the movement continues. "
        f"{repetition_line}"
    )










# ---------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------
@app.get(
    "/gen_text2motion/",
    tags=["generation"],
    summary="Generate motion from text prompt (translation + rewrite + MoMask + Blender)",
    response_model=GenText2MotionResponse,
)
async def gen_text2motion(
    text_prompt: str = Query(..., description="Text prompt (any language)"),
    video_render: str = Query("false", description="Render video (true/false)"),
    # high_res: str = Query("false", description="Use high resolution video (true/false)"),
):
    base_prompt = text_prompt.strip()
    with open("input.txt", "w", encoding="utf-8") as f:
        f.write(base_prompt)
    print("[INFO] Saved base prompt -> input.txt")

    english_prompt = translate_to_english(base_prompt)
    with open("input_en.txt", "w", encoding="utf-8") as f:
        f.write(english_prompt)
    print("[INFO] Saved English prompt -> input_en.txt")

    expressive_prompt = rewrite_prompt_auto(english_prompt)

    clean_base = re.sub(r"\s+#.*", "", english_prompt).strip()
    modified_prompt = f"{clean_base} # {expressive_prompt}"
    if not modified_prompt.endswith("#268"):
        modified_prompt = modified_prompt.rstrip(". ") + ". #268"

    with open("rewrite_input.txt", "w", encoding="utf-8") as f:
        f.write(modified_prompt)
    print("[INFO] Saved rewritten prompt -> rewrite_input.txt")

    try:
        def run_evaluation():
            subprocess.run("python gen_momask_plus.py", shell=True, check=True)

            sys_platform = platform.system()
            if sys_platform == "Windows":
                blender_cmd = (
                    'blender --background --addons KeeMapAnimRetarget '
                    '--python "./bvh2fbx/bvh2fbx.py" '
                    f'-- --video_render={video_render.lower()}'
                )
            elif sys_platform == "Darwin":
                blender_cmd = (
                    '"/Applications/Blender.app/Contents/MacOS/Blender" --background '
                    "--addons KeeMapAnimRetarget "
                    '--python "./bvh2fbx/bvh2fbx.py" '
                    f'-- --video_render={video_render.lower()}'
                )
            elif sys_platform == "Linux":
                blender_cmd = (
                    "xvfb-run blender --background "
                    "--addons KeeMapAnimRetarget "
                    "--python ./bvh2fbx/bvh2fbx.py "
                    f"-- --video_render={video_render.lower()}"
                )
            else:
                blender_cmd = (
                    'blender --background --addons KeeMapAnimRetarget '
                    '--python "./bvh2fbx/bvh2fbx.py" '
                    f'-- --video_render={video_render.lower()}'
                )

            print("Launching Blender:", blender_cmd)
            subprocess.run(blender_cmd, shell=True, check=True)

        await run_in_threadpool(run_evaluation)

        return {
            "status": "success",
            "message": "Evaluation completed successfully.",
            "original_prompt": base_prompt,
            "english_prompt": english_prompt,
            "expressive_prompt": expressive_prompt,
        }

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Command execution failed: {e}")


# ---------------------------------------------------------------------
# Animation selection / compare (pre-generated zips)
# ---------------------------------------------------------------------
animation_state = {
    "model": "pretrained",
    "index": 0,
}

compare_state = {
    "index": 1,
    "triggered": False,
}


@app.get(
    "/set_animation",
    tags=["animation"],
    summary="Set which pre-generated animation should be served next",
    response_model=SetAnimationResponse,
)
def set_animation(
    model: str = Query(..., enum=["pretrained", "retrained"]),
    anim: int = Query(..., ge=1, le=5),
):
    animation_state["model"] = model
    animation_state["index"] = anim
    return {"status": "updated", "server_status": server_status_value, **animation_state}


@app.get(
    "/get_animation",
    tags=["animation"],
    summary="Download the selected pre-generated animation ZIP",
)
def get_animation():
    model = animation_state["model"]
    index = animation_state["index"]

    filename = f"{model}_{index}.zip"
    zip_folder = "pregen_animation"
    zip_path = os.path.join(zip_folder, filename)

    if os.path.isfile(zip_path):
        return FileResponse(
            zip_path,
            media_type="application/octet-stream",
            filename=filename,
        )
    raise HTTPException(status_code=404, detail=f"File '{filename}' not found")


@app.post(
    "/trigger_compare/",
    tags=["animation"],
    summary="Trigger compare mode with an index (one-shot)",
    response_model=CompareTriggerResponse,
)
async def trigger_compare(index: int = Query(..., ge=1, le=5)):
    compare_state["index"] = index
    compare_state["triggered"] = True
    return {"status": "compare triggered", "index": index}


@app.get(
    "/check_compare/",
    tags=["animation"],
    summary="Check compare trigger state (returns compare once, then resets to idle)",
)
async def check_compare():
    if compare_state["triggered"]:
        compare_state["triggered"] = False
        return {"status": "compare", "index": compare_state["index"]}
    return {"status": "idle"}


@app.get(
    "/check_animation_state/",
    tags=["animation"],
    summary="Get current animation model state",
    response_model=AnimationStateResponse,
)
def check_animation_state():
    return {"model": animation_state["model"]}


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    print("[Startup] Launching FastAPI on 0.0.0.0:8000 ...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
