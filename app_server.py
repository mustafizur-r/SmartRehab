from cffi.model import global_lock
from fastapi import FastAPI, File, HTTPException, Query, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import torch
import zipfile
import platform
from typing import List, Optional
from starlette.concurrency import run_in_threadpool
import subprocess, platform, re, os
from langdetect import detect

app = FastAPI()
text_prompt_global = None
server_status_value = 4  # Default to initial status


class ServerStatus(BaseModel):
    status: int


class Prompt(BaseModel):
    prompt: str


@app.get("/", response_class=HTMLResponse, tags=["root"])
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
           href="https://mustafizur-r.github.io/text2gaitsim/"
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


@app.get("/server_status/")
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


@app.post("/set_server_status/")
async def set_server_status(server_status: ServerStatus):
    status = server_status.status
    print(f"Received status: {status}")
    global server_status_value
    if status in (0, 1, 2, 3):
        server_status_value = status
        # Return a success message with status message if 0, 1, or 2
        if status == 0:
            return {"message": "Server status set to 'not running'."}
        elif status == 1:
            return {"message": "Server status set to 'running'."}
        elif status == 2:
            return {"message": "Server status set to 'initial status'."}
        elif status == 3:
            return {"message": "Server status set to 'compare'."}
    else:
        # Return an error message if the status is invalid
        raise HTTPException(status_code=400, detail="Invalid status value. Please provide 0, 1, or 2,3.")


# get input prompt
@app.get("/get_prompts/")
async def get_prompts():
    try:
        file_path = "input.txt"

        with open(file_path, "r") as file:
            prompt = file.read()
            # Return structured JSON response
            return {"status": "success", "prompt": prompt}
    except Exception as e:
        # Return an error message in JSON format
        raise HTTPException(status_code=500, detail={"status": "failed", "error": str(e)})


@app.get("/download_fbx/")
async def download_bvh(filename: str):
    bvh_folder = 'fbx_folder'
    filepath = os.path.join(bvh_folder, filename)

    if os.path.exists(filepath):
        return FileResponse(filepath, media_type='application/octet-stream', filename=filename)
    else:
        raise HTTPException(status_code=404, detail="File not found")


VIDEO_PATH = "./video_result/Final_Fbx_Mesh_Animation.mp4"


@app.get("/download_video")
async def download_video():
    if not os.path.exists(VIDEO_PATH):
        return {"error": "Video not found."}
    return FileResponse(
        path=VIDEO_PATH,
        filename="Final_Fbx_Mesh_Animation.mp4",
        media_type="video/mp4"
    )


@app.get("/download_zip/")
async def download_fbx(filename: str):
    fbx_zip_folder = 'fbx_zip_folder'  # Folder where the ZIP files are stored
    zip_filepath = os.path.join(fbx_zip_folder, filename)
    if os.path.exists(zip_filepath):
        return FileResponse(zip_filepath, media_type='application/octet-stream', filename=filename)
    else:
        raise HTTPException(status_code=404, detail="File not found")





# =====================================================
# 1) TRANSLATION FUNCTION (offline, deterministic, CJK-aware)
# =====================================================
def translate_to_english(raw_text: str) -> str:
    """
    Translate any-language input to English (offline).
    Priority:
      1) Ollama (local) with English-only instruction
      2) Fallback: return original text
    Uses qwen2.5 if available, else llama3.
    """
    import re, unicodedata
    cleaned = unicodedata.normalize("NFKC", raw_text).strip()

    def _pick_model():
        try:
            import ollama
            have = {m.get("model","") for m in ollama.list().get("models", [])}
            # prefer qwen2.5 if present, else llama3
            for name in ("qwen2.5:7b-instruct-q4_K_M", "qwen2.5:7b-instruct", "llama3"):
                if any(x.startswith(name) for x in have):
                    return name
        except Exception:
            pass
        return "llama3"

    model = _pick_model()

    try:
        import ollama
        tmpl = (
            "You are a professional Japanese/Chinese/English medical translator. "
            "Translate the user's text into NATURAL, idiomatic ENGLISH.\n"
            "Output ONLY the translation — no headings, no notes, no quotes, no markdown.\n"
            "Preserve clinical/anatomical terms and laterality precisely. Do not add information.\n\n"
            f"Text:\n{cleaned}"
        )
        print(f"[Ollama] Translating to English with model={model} ...")
        resp = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": tmpl}],
            options={"temperature": 0, "num_ctx": 4096, "num_predict": 1024},
        )
        out = (resp.get("message", {}) or {}).get("content", "") or ""
        # strip wrappers/meta
        out = out.replace("\n", " ").strip(" '\"“”‘’`")
        out = re.sub(r"\s{2,}", " ", out)
        # if any leftover CJK slipped through, make one stricter retry
        if re.search(r"[\u3040-\u30FF\u4E00-\u9FFF]", out):
            strict = (
                "Translate STRICTLY into ENGLISH ONLY. No headings/notes/quotes/markdown. "
                "Return only fluent English sentences.\n\n" + cleaned
            )
            resp2 = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": strict}],
                options={"temperature": 0, "num_ctx": 4096, "num_predict": 1024},
            )
            out2 = (resp2.get("message", {}) or {}).get("content", "") or ""
            out2 = out2.replace("\n", " ").strip(" '\"“”‘’`")
            out2 = re.sub(r"\s{2,}", " ", out2)
            if out2:
                out = out2

        return out or cleaned
    except Exception as e:
        print(f"[Ollama] Translation unavailable: {e}")
        return cleaned



# =====================================================
# 2) PROMPT REWRITE FUNCTION (no meta text, single paragraph, facts preserved)
# =====================================================
def rewrite_prompt_auto(raw_text: str) -> str:
    """
    Rewrites clinical gait descriptions into ONE SnapMoGen-style paragraph (70–100 words)
    while preserving original meaning (laterality, diagnosis, device use, negations)
    and guaranteeing forward locomotion suitable for animation synthesis.
    """

    import re

    # ---- Locomotion defaults ----
    LOCOMOTION_DISTANCE = "2–4 meters"
    LOCOMOTION_DIRECTION = "straight line"
    LOCOMOTION_PACE = "slow, uneven pace"

    # ---- Detect walking context ----
    has_walk = re.search(r"\b(walk|walking|gait|stride|step|limp|pace|stumble|shuffle|hobble|advance)\b",
                         raw_text.lower())
    if not has_walk:
        raw_text = raw_text.strip() + (
            " The person attempts to walk forward, showing imbalance and effort consistent with an abnormal gait."
        )

    # ---- Identify protected clinical info ----
    protected = []
    for word in ["right", "left", "bilateral", "hemiplegia", "hemiparesis", "stroke",
                 "cane", "walker", "crutch", "no pain", "without assistance", "unsteady",
                 "ataxic", "spastic", "foot drop", "trendelenburg"]:
        if re.search(rf"\b{re.escape(word)}\b", raw_text.lower()):
            protected.append(word)
    protected_text = "Preserve exactly these clinical facts: " + ", ".join(protected) + "." if protected else ""

    # ---- Prompt to model ----
    tmpl = f"""
    Rewrite the following description into ONE continuous, natural ENGLISH paragraph (70–100 words)
    suitable for SnapMoGen motion synthesis.
    Requirements:
    - Must depict a person WALKING with an ABNORMAL GAIT (not static).
    - Include clear FORWARD LOCOMOTION over {LOCOMOTION_DISTANCE} in a {LOCOMOTION_DIRECTION} at a {LOCOMOTION_PACE}.
    - Keep the clinical meaning identical: no change in laterality, diagnosis, assistive device, or severity.
    - Maintain a smooth chronological flow (start → progression → continuation).
    - Include limb, balance, and compensatory movement details.
    - Respond ONLY with the paragraph text — no headings, no introductions, and no phrases like "Here is the rewritten paragraph:".
    - Do not add explanations or conclusions.
    {protected_text}

    Original: {raw_text}
    """.strip()

    try:
        import ollama
        print("[Ollama] Generating locomotion-consistent rewrite...")
        resp = ollama.chat(
            model="llama3",  # llama3 is fine for rewriting; keep as you used
            messages=[{"role": "user", "content": tmpl}],
            options={"temperature": 0, "num_ctx": 4096, "num_predict": 512},
        )
        out = (resp.get("message", {}) or {}).get("content", "") or ""

        # ---- Clean unwanted text (kill meta-intros and boilerplate) ----
        # Specific meta phrases
        out = re.sub(
            r"(?is)^\s*(?:here\s*(?:is|’s)\s*(?:the\s*)?(?:rewritten|final)\s*(?:paragraph|version)\s*:|"
            r"this\s*(?:is)?\s*(?:the\s*)?(?:rewritten|final)\s*(?:paragraph|version)\s*:|"
            r"(?:rewritten|final)\s*(?:paragraph|version)\s*:)\s*",
            "",
            out.strip(),
        )
        # Generic: strip a short leading label + colon if any slipped through
        out = re.sub(r"(?is)^\s*[^.\n]{0,80}:\s+", "", out)

        # Housekeeping
        out = out.replace("\n", " ")
        out = re.sub(r"\s{2,}", " ", out).strip(" \"'“”‘’")
        out = re.sub(r"(?is)\b(let me know|hope this helps|feel free to ask|would you like).*", "", out).strip()

        # ---- Ensure forward movement phrasing ----
        if not re.search(r"\b(move|walk|advance|proceed|travel)\b", out.lower()):
            out = f"The person walks forward over {LOCOMOTION_DISTANCE} in a {LOCOMOTION_DIRECTION}. " + out

        # ---- Enforce 70–100 words softly (trim if wildly long) ----
        words = out.split()
        if len(words) > 110:
            # Trim to last full stop before ~100 words; if none, hard-trim
            cut = 100
            # search backwards for a sentence end within ±15 words
            for i in range(min(len(words), 110), 75, -1):
                if words[i-1].endswith(('.', '!', '?')):
                    cut = i
                    break
            out = " ".join(words[:cut]).rstrip(",;: ") + "."

        # ---- Re-assert missing protected facts (append, not invent) ----
        out_lc = out.lower()
        missing = [p for p in protected if p not in out_lc]
        if missing:
            out += " (Preserved facts: " + ", ".join(missing) + ".)"

        return out if out else raw_text

    except Exception as e:
        print(f"[Ollama] Failed or unavailable: {e}")
        return (raw_text.strip() +
                f" The person walks forward {LOCOMOTION_DISTANCE} in a {LOCOMOTION_DIRECTION}, "
                "displaying imbalance, asymmetry, and compensatory movements consistent with abnormal gait.")


# =====================================================
# 3) MAIN ENDPOINT
# =====================================================
@app.get("/gen_text2motion/")
async def gen_text2motion(
    text_prompt: str = Query(..., description="Text prompt (any language)"),
    video_render: str = Query("false", description="Render video (true/false)"),
    high_res: str = Query("false", description="Use high resolution video (true/false)")
):
    # -----------------------------
    # Step 1: Save raw prompt
    # -----------------------------
    base_prompt = text_prompt.strip()
    with open("input.txt", "w", encoding="utf-8") as f:
        f.write(base_prompt)
    print("[INFO] Saved base prompt → input.txt")

    # -----------------------------
    # Step 2: Translate + rewrite
    # -----------------------------
    english_prompt = translate_to_english(base_prompt)
    with open("input_en.txt", "w", encoding="utf-8") as f:
        f.write(english_prompt)
    print("[INFO] Saved English prompt → input_en.txt")

    expressive_prompt = rewrite_prompt_auto(english_prompt)

    # -----------------------------
    # Step 3: Combine final format
    # -----------------------------
    clean_base = re.sub(r"\s+#.*", "", english_prompt).strip()
    modified_prompt = f"{clean_base} # {expressive_prompt}"
    if not modified_prompt.endswith("#268"):
        modified_prompt = modified_prompt.rstrip(". ") + ". #268"

    with open("rewrite_input.txt", "w", encoding="utf-8") as f:
        f.write(modified_prompt)
    print("[INFO] Saved rewritten prompt → rewrite_input.txt")

    # -----------------------------
    # Step 4: Run MoMask + Blender
    # -----------------------------
    try:
        def run_evaluation():
            # 1) Text-to-motion generation
            subprocess.run("python gen_momask_plus.py", shell=True, check=True)

            # 2) BVH → FBX via Blender
            sys_platform = platform.system()
            if sys_platform == "Windows":
                blender_cmd = (
                    'blender --background --addons KeeMapAnimRetarget '
                    '--python "./bvh2fbx/bvh2fbx.py" '
                    f'-- --video_render={video_render.lower()} --high_res={high_res.lower()}'
                )
            elif sys_platform == "Darwin":
                blender_cmd = (
                    '"/Applications/Blender.app/Contents/MacOS/Blender" --background '
                    '--addons KeeMapAnimRetarget '
                    '--python "./bvh2fbx/bvh2fbx.py" '
                    f'-- --video_render={video_render.lower()} --high_res={high_res.lower()}'
                )
            elif sys_platform == "Linux":
                blender_cmd = (
                    "xvfb-run blender --background "
                    "--addons KeeMapAnimRetarget "
                    "--python ./bvh2fbx/bvh2fbx.py "
                    f"-- --video_render={video_render.lower()} --high_res={high_res.lower()}"
                )
            else:
                blender_cmd = (
                    'blender --background --addons KeeMapAnimRetarget '
                    '--python "./bvh2fbx/bvh2fbx.py" '
                    f'-- --video_render={video_render.lower()} --high_res={high_res.lower()}'
                )

            print("Launching Blender:", blender_cmd)
            subprocess.run(blender_cmd, shell=True, check=True)

        await run_in_threadpool(run_evaluation)

        return {
            "status": "success",
            "message": "Evaluation completed successfully.",
            "original_prompt": base_prompt,
            "english_prompt": english_prompt,
            "expressive_prompt": expressive_prompt
        }

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Command execution failed: {e}")




# @app.get("/gen_text2motion/")
# async def gen_text2motion(
#     text_prompt: str = Query(..., description="Text prompt"),
#     video_render: str = Query("false", description="Render video (true/false)"),
#     high_res: str = Query("false", description="Use high resolution video (true/false)")
# ):
#
#     # 1) Save the prompt for downstream scripts to read
#     # with open("input.txt", "w", encoding="utf-8") as f:
#     #     f.write(text_prompt)
#     # 1) Modify the text_prompt
#     modified_prompt = text_prompt
#
#     # Insert '#' after the first period
#     if "." in modified_prompt:
#         parts = modified_prompt.split(".", 1)  # split only on first period
#         modified_prompt = parts[0] + ". # " + parts[1].strip()
#     else:
#         modified_prompt = modified_prompt.strip()
#
#     # Ensure it ends with #268
#     if modified_prompt.endswith("."):
#         modified_prompt = modified_prompt[:-1] + ". #268"
#     elif not modified_prompt.endswith("#268"):
#         modified_prompt = modified_prompt + ". #268"
#
#     # 2) Save the modified prompt
#     with open("input.txt", "w", encoding="utf-8") as f:
#         f.write(modified_prompt)
#
#     try:
#         def run_evaluation():
#             # 2) Run the text-to-motion Python script
#             python_cmd = f'python gen_momask_plus.py'
#             subprocess.run(python_cmd, shell=True, check=True)
#
#             # 3) Build a Blender command string depending on the OS
#             sys_platform = platform.system()  # "Windows", "Linux", "Darwin", etc.
#
#             if sys_platform == "Windows":
#                 # ── WINDOWS ───────────────────────────────────────────────────────────────────
#                 # If blender.exe is on your PATH, leave as "blender". Otherwise use the full path.
#                 BLENDER_EXE_PATH = "blender"
#                 blender_cmd = (
#                     f'"{BLENDER_EXE_PATH}" --background '
#                     '--addons KeeMapAnimRetarget '
#                     '--python "./bvh2fbx/bvh2fbx.py" '
#                     f'-- --video_render={video_render.lower()} --high_res={high_res.lower()}'
#                 )
#
#             elif sys_platform == "Darwin":
#                 # ── macOS ───────────────────────────────────────────────────────────────────
#                 # Use the Blender executable inside the .app bundle.
#                 BLENDER_EXE_PATH = "/Applications/Blender.app/Contents/MacOS/Blender"
#                 blender_cmd = (
#                     f'"{BLENDER_EXE_PATH}" --background '
#                     '--addons KeeMapAnimRetarget '
#                     '--python "./bvh2fbx/bvh2fbx.py" '
#                     f'-- --video_render={video_render.lower()} --high_res={high_res.lower()}'
#                 )
#
#             elif sys_platform == "Linux":
#                 # ── LINUX ───────────────────────────────────────────────────────────────────
#                 # Use xvfb-run for headless execution if no display is available.
#                 blender_cmd = (
#                     "xvfb-run blender --background "
#                     "--addons KeeMapAnimRetarget "
#                     "--python ./bvh2fbx/bvh2fbx.py "
#                     f"-- --video_render={video_render.lower()} --high_res={high_res.lower()}"
#                 )
#
#             else:
#                 # ── ANY OTHER PLATFORM ───────────────────────────────────────────────────────
#                 # Fallback: call blender directly
#                 BLENDER_EXE_PATH = "blender"
#                 blender_cmd = (
#                     f'"{BLENDER_EXE_PATH}" --background '
#                     '--addons KeeMapAnimRetarget '
#                     '--python "./bvh2fbx/bvh2fbx.py" '
#                     f'-- --video_render={video_render.lower()} --high_res={high_res.lower()}'
#                 )
#
#             print("Launching Blender step:", blender_cmd)
#             subprocess.run(blender_cmd, shell=True, check=True)
#
#         # 4) Execute text2motion + Blender in a background thread
#         await run_in_threadpool(run_evaluation)
#
#         return {"status": "success", "message": "Evaluation completed successfully."}
#
#     except subprocess.CalledProcessError as e:
#         raise HTTPException(status_code=500, detail=f"Command execution failed: {e}")

@app.post("/input_prompts/")
async def input_prompts(prompt: Prompt):
    try:
        file_path = "input.txt"

        with open(file_path, "w") as file:
            file.write(prompt.prompt)

        return {"message": "Prompt saved successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving prompt: {e}")


# Store current animation state
animation_state = {
    "model": "pretrained",  # "pretrained" or "retrained"
    "index": 0  # 0 to 4
}

compare_state = {
    "index": 1,
    "triggered": False
}


@app.get("/set_status")
def set_status(status: int = Query(..., ge=0, le=5)):
    global server_status_value

    server_status_value = status
    return {"status": "updated", "server_status": server_status_value}


@app.get("/set_animation")
def set_animation(
        model: str = Query(..., enum=["pretrained", "retrained"]),
        anim: int = Query(..., ge=1, le=5)
):
    animation_state["model"] = model
    animation_state["index"] = anim

    return {"status": "updated", "server_status": server_status_value, **animation_state}


@app.get("/get_animation")
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
            filename=filename
        )
    else:
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found")


@app.post("/trigger_compare/")
async def trigger_compare(index: int = Query(..., ge=1, le=5)):
    compare_state["index"] = index
    compare_state["triggered"] = True
    return {"status": "compare triggered", "index": index}


@app.get("/check_compare/")
async def check_compare():
    if compare_state["triggered"]:
        compare_state["triggered"] = False
        return {
            "status": "compare",
            "index": compare_state["index"]
        }
    else:
        return {"status": "idle"}


@app.get("/check_animation_state/")
def check_animation_state():
    return {
        "model": animation_state["model"]
    }


# if __name__ == "__main__":
#     import uvicorn
#
#     # Run the FastAPI application with uvicorn
#     # uvicorn.run(app, host="localhost", port=8000)
#     # for ip address access
#     uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    import uvicorn, subprocess, time, requests, os


    def is_ollama_running() -> bool:
        """Check if Ollama API service is available."""
        try:
            r = requests.get("http://127.0.0.1:11434/api/version", timeout=2)
            return r.status_code == 200
        except Exception:
            return False


    # -----------------------------------------------------
    # 1 Auto-start Ollama service if not running
    # -----------------------------------------------------
    if not is_ollama_running():
        print("[Startup] Ollama not running — starting service...")
        try:
            if os.name == "nt":  # Windows
                subprocess.Popen(
                    [
                        "powershell",
                        "-Command",
                        "Start-Process ollama -ArgumentList 'serve' -WindowStyle Hidden"
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            else:  # Linux / macOS / Docker
                subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            print("[Startup] Waiting 5 seconds for Ollama to initialize...")
            time.sleep(5)
        except Exception as e:
            print(f"[Startup] Failed to start Ollama: {e}")
    else:
        print("[Startup] Ollama service already running.")

    # -----------------------------------------------------
    # 2 Launch FastAPI
    # -----------------------------------------------------
    print("[Startup] Launching FastAPI on 0.0.0.0:8000 ...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
