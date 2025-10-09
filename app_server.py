from cffi.model import global_lock
from fastapi import FastAPI, File, HTTPException, Query, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import os
import torch
import subprocess
import zipfile
import platform
from typing import List, Optional
from starlette.concurrency import run_in_threadpool

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


#get input prompt
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

@app.get("/gen_text2motion/")
async def gen_text2motion(
    text_prompt: str = Query(..., description="Text prompt"),
    video_render: str = Query("false", description="Render video (true/false)"),
    high_res: str = Query("false", description="Use high resolution video (true/false)")
):

    # 1) Save the prompt for downstream scripts to read
    # with open("input.txt", "w", encoding="utf-8") as f:
    #     f.write(text_prompt)
    # 1) Modify the text_prompt
    modified_prompt = text_prompt

    # Insert '#' after the first period
    if "." in modified_prompt:
        parts = modified_prompt.split(".", 1)  # split only on first period
        modified_prompt = parts[0] + ". # " + parts[1].strip()
    else:
        modified_prompt = modified_prompt.strip()

    # Ensure it ends with #268
    if modified_prompt.endswith("."):
        modified_prompt = modified_prompt[:-1] + ". #268"
    elif not modified_prompt.endswith("#268"):
        modified_prompt = modified_prompt + ". #268"

    # 2) Save the modified prompt
    with open("input.txt", "w", encoding="utf-8") as f:
        f.write(modified_prompt)

    try:
        def run_evaluation():
            # 2) Run the text-to-motion Python script
            python_cmd = f'python gen_momask_plus.py'
            subprocess.run(python_cmd, shell=True, check=True)

            # 3) Build a Blender command string depending on the OS
            sys_platform = platform.system()  # "Windows", "Linux", "Darwin", etc.

            if sys_platform == "Windows":
                # ── WINDOWS ───────────────────────────────────────────────────────────────────
                # If blender.exe is on your PATH, leave as "blender". Otherwise use the full path.
                BLENDER_EXE_PATH = "blender"
                blender_cmd = (
                    f'"{BLENDER_EXE_PATH}" --background '
                    '--addons KeeMapAnimRetarget '
                    '--python "./bvh2fbx/bvh2fbx.py" '
                    f'-- --video_render={video_render.lower()} --high_res={high_res.lower()}'
                )

            elif sys_platform == "Darwin":
                # ── macOS ───────────────────────────────────────────────────────────────────
                # Use the Blender executable inside the .app bundle.
                BLENDER_EXE_PATH = "/Applications/Blender.app/Contents/MacOS/Blender"
                blender_cmd = (
                    f'"{BLENDER_EXE_PATH}" --background '
                    '--addons KeeMapAnimRetarget '
                    '--python "./bvh2fbx/bvh2fbx.py" '
                    f'-- --video_render={video_render.lower()} --high_res={high_res.lower()}'
                )

            elif sys_platform == "Linux":
                # ── LINUX ───────────────────────────────────────────────────────────────────
                # Use xvfb-run for headless execution if no display is available.
                blender_cmd = (
                    "xvfb-run blender --background "
                    "--addons KeeMapAnimRetarget "
                    "--python ./bvh2fbx/bvh2fbx.py "
                    f"-- --video_render={video_render.lower()} --high_res={high_res.lower()}"
                )

            else:
                # ── ANY OTHER PLATFORM ───────────────────────────────────────────────────────
                # Fallback: call blender directly
                BLENDER_EXE_PATH = "blender"
                blender_cmd = (
                    f'"{BLENDER_EXE_PATH}" --background '
                    '--addons KeeMapAnimRetarget '
                    '--python "./bvh2fbx/bvh2fbx.py" '
                    f'-- --video_render={video_render.lower()} --high_res={high_res.lower()}'
                )

            print("Launching Blender step:", blender_cmd)
            subprocess.run(blender_cmd, shell=True, check=True)

        # 4) Execute text2motion + Blender in a background thread
        await run_in_threadpool(run_evaluation)

        return {"status": "success", "message": "Evaluation completed successfully."}

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Command execution failed: {e}")

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
    "index": 0              # 0 to 4
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

if __name__ == "__main__":
    import uvicorn

    # Run the FastAPI application with uvicorn
    # uvicorn.run(app, host="localhost", port=8000)
    # for ip address access
    uvicorn.run(app, host="0.0.0.0", port=8000)
