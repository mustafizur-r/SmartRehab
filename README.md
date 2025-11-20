# SmartRehab-Codes Project (Dockerized)

A Dockerized FastAPI server for generating and retargeting patient-specific gait motions using text-to-motion models, Blender, and the KeeMap rig-retargeting addon.

---

## Features

* **Text-to-Motion**: Generate impaired gait motions from text prompts via `gen_t2m.py` and MoMask models.
* **Animation Retargeting**: Convert `.bvh` output to `.fbx` using Blender + KeeMap addon.
* **REST API**: Expose endpoints for status, prompts, motion generation, and file download.
* **Fully Dockerized**: One-command build & run, with Conda, Blender, CLIP, and model downloads baked in.

---

## Landing Page

Check out the demo and documentation at: [https://mustafizur-r.github.io/SmartRehab/](https://mustafizur-r.github.io/text2gaitsim/)

---

## Prerequisites

* [Docker](https://www.docker.com/get-started)
* NVIDIA GPU
* NVIDIA Docker Runtime (`nvidia-docker2`)
* Docker >= 20.10 with GPU support
* Driver >= CUDA 12.1
* "/Applications/Blender.app/Contents/MacOS/Blender"

### ✅ Verify GPU Access

**Windows CMD / PowerShell**

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('torch version:', torch.__version__); print('CUDA version PyTorch was built for:', torch.version.cuda); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

**Linux / WSL / macOS**

```bash
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('torch version:', torch.__version__); print('CUDA version PyTorch was built for:', torch.version.cuda); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

👉 [Download CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

---

## Quickstart

<details>
  <summary><strong>👉 Windows (Local Setup)</strong></summary>

  1. **Clone this repo**
     ```bash
     git clone https://github.com/mustafizur-r/SmartRehab.git
     cd SmartRehab
     ```

  2. **Create and activate Conda environment**
     ```bash
     conda env create -f environment.yml
     conda activate smartrehab
     ```

  3. **Run the project**
     ```bash
     python main.py
     ```
</details>

<details>
  <summary><strong>👉 Docker (Cross-Platform)</strong></summary>

  1. **Clone this repo**
     ```bash
     git clone https://github.com/mustafizur-r/SmartRehab.git
     cd SmartRehab
     ```

  2. **Build the Docker image**
     ```bash
     docker build -t smartrehab .
     ```

3. **Run the container**
   **For CPU**

   ```bash
   docker run --name smartrehab -p 8000:8000 smartrehab
   ```

   **For GPU**

   ```bash
   docker run --name smartrehab --gpus all -p 8000:8000 smartrehab
   ```

   **(Optional) Note: Run in Background use -d tag...**</br>
   **For CPU**

   ```bash
   docker run -d --name smartrehab -p 8000:8000 smartrehab
   ```

   **For GPU**

   ```bash
   docker run -d --name smartrehab --gpus all -p 8000:8000 smartrehab
   ```

   **(Optional) Enter the container**

   ```bash
   docker exec -it smartrehab bash
   ```

---

## Manual Animation Generation & Output Access

**Note: Follow Run the container Step 3 First**

1. **Start a shell in the container**

   ```bash
   docker exec -it smartrehab bash
   ```

2. **Activate your conda environment**

   ```bash
   conda activate smartrehab
   ```

3. **Generate text→motion**

     ```bash
     python gen_momask_plus.py
     ```

4. **Locate the raw BVH output**

   ```bash
   cd bvh_folder
   ls
   # e.g. bvh_0_out.bvh
   ```

6. **Run Blender to retarget BVH→FBX**
   *Run full pipeline without rendering (default)*
   ```bash
   xvfb-run blender --background --addons KeeMapAnimRetarget --python ./bvh2fbx/my_bvh2fbx.py
   ```
   *Run and render video in low-res*
   ```bash
   xvfb-run blender --background --addons KeeMapAnimRetarget --python ./bvh2fbx/my_bvh2fbx.py -- --video_render=true
   ```
   *Run and render video in high-res*
   ```bash
   xvfb-run blender --background --addons KeeMapAnimRetarget --python ./bvh2fbx/my_bvh2fbx.py -- --video_render=true --high_res=true
   ```

7. **Grab the retargeted FBX,FBX Animated mp4 or ZIP**

   ```bash
   ls fbx_folder
   ls fbx_zip_folder
   ls video_result
   ```

---



## Stop & Remove Container

```bash
# Stop the container
docker stop smartrehab

# Remove it (if not using --rm)
docker rm smartrehab
```

---
</details>

## Accessing Generated Files & Folders

By default, outputs are written inside the container under `/workspace`:

* **`bvh_folder/`**: `.bvh` motion files from `gen_t2m.py`
* **`fbx_folder/`**: Retargeted `.fbx` files from Blender
* **`fbx_zip_folder/`**: Zipped archives of the `.fbx` files
* **`video_result/`**: Video archives of the `.mp4` files

### Mounting to Host

Persist these on your host machine by mounting volumes:

```bash
docker run -d --name smartrehab   -p 8000:8000   -v $(pwd)/bvh_folder:/workspace/bvh_folder   -v $(pwd)/fbx_folder:/workspace/fbx_folder   -v $(pwd)/fbx_zip_folder:/workspace/fbx_zip_folder -v $(pwd)/videos:/workspace/videos  smartrehab
```

### Inspecting Inside the Container

```bash
docker exec -it smartrehab bash
ls /workspace/bvh_folder
ls /workspace/fbx_folder
ls /workspace/fbx_zip_folder
ls /workspace/video_result
```

### Copying Out Without Mounts

```bash
docker cp smartrehab:/workspace/fbx_folder/bvh_0_out.fbx ./fbx_folder/
docker cp smartrehab:/workspace/fbx_zip_folder/bvh_0_out.zip ./fbx_zip_folder/
docker cp smartrehab:/workspace/video_result/Final_Fbx_Mesh_Animation.mp4 ./videos/
```

---

## API Endpoints

### 1. Health & Status

* **GET** `/server_status/`

  ```json
  { "message": "Server is running." }
  ```

* **POST** `/set_server_status/`
  *Body*:

  ```json
  { "status": 0 }
  ```

### 2. Prompt Management

* **GET** `/get_prompts/`
  Reads `input.txt` and returns:

  ```json
  { "status": "success", "prompt": "<your text>" }
  ```

* **POST** `/input_prompts/`
  *Body*:

  ```json
  { "prompt": "A person walks with a limp." }
  ```

### 3. Text-to-Motion
**For Tags:
  - --video_render=true (System Render video with low resolution)**
  - --video_render=true&high_res=true (System Render video with high resolution)**
* **GET** `/gen_text2motion/?ext=<config>&text_prompt=<your+text>[&gpu_id=<id>&video_render=true&high_res=true]`
    ### Generate motion only (no video) 
* **GET** `/gen_text2motion/?text_prompt=A person walks with a limp`
    ### Generate motion with low-res video

* **GET** ` /gen_text2motion/?video_render=true&text_prompt=A person walks with a limp`

    ### Generate motion with high-res video
* **GET** ` /gen_text2motion/?video_render=true&high_res=true&text_prompt=A person walks with a limp`
  1. Runs `gen_momask_plus.py` (GPU if requested and available).
  2. Launches headless Blender with KeeMap for retargeting.
     **Returns**:

  ```json
  { "status": "success", "message": "Evaluation completed successfully." }
  ```
### 4. Render MP4 video separately
The rendered video will be saved to `video_result/Final_Fbx_Mesh_Animation.mp4` and can be downloaded via:
```bash
GET /download_video/
```
---
### 5. File Downloads

* **GET** `/download_fbx/?filename=<name>.fbx`
  Streams the `.fbx` from `fbx_folder/`.
* **GET** `/download_zip/?filename=<name>.zip`
  Streams the zipped archive from `fbx_zip_folder/`.
* **GET** `/download_video/`
  Streams the video archive from `download_video/`.

---

## Usage Examples

```bash
# Generate motion & retarget with gpu
http://localhost:8000/gen_text2motion/?text_prompt=Walk forward with a limp.
# Generate motion & retarget with gpu -low res video
http://localhost:8000/gen_text2motion/?video_render=true&text_prompt=Walk forward with a limp.
# Generate motion & retarget with gpu -high res video
http://localhost:8000/gen_text2motion/?video_render=true&high_res=true&text_prompt=Walk forward with a limp.

# Download results fbx
http://localhost:8000/download_fbx/?filename=bvh_0_out.fbx
# Download results
http://localhost:8000/download_zip/?filename=bvh_0_out.zip

# Download video results
http://localhost:8000/download_video

```

---



## Project Structure

```
project_root/
├── Dockerfile
├── environment.yml
├── app_server.py                         # FastAPI server with /download_video
├── gen_momask_plus.py
├── bvh2fbx/                        # Blender automation + FBX input
│   ├── bvh2fbx.py
│   ├── log.txt
│   └── test.fbx                    # Input FBX model for retargeting
├── prepare/
│   ├── download_models.sh
│   ├── download_evaluator.sh
│   └── download_glove.sh
├── bone_mapping_asset/
│   └── mapping_mixamo_brianmodel.json      # Bone mapping for retargeting
├── fbx_folder/                     # Output: exported FBX animations
├── fbx_zip_folder/                 # Output: zipped FBX files
├── bvh_folder/                     # Input: BVH motion files
├── video_result/                         # Final rendered video(s)
│   └── Final_Fbx_Mesh_Animation.mp4
└── input.txt                       # Input Prompts and Title text for rendering
                 
```

---

## License & Credits

© 2025 Md Mustafizur Rahman et al.
Licensed under CC BY-4.0 — see [LICENSE](LICENSE).

Prepared by Md Mustafizur Rahman
Master's student, Interactive Media Design Laboratory, NAIST
[Portfolio & Docs](https://mustafizur-r.github.io/text2gaitsim/)

