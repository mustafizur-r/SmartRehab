# 0) Base image
FROM continuumio/miniconda3
LABEL authors="mdmustafizur-r"

# 1) Set working directory
WORKDIR /workspace

# 2) Copy env specs early for better Docker cache
COPY environment.yml ./environment.yml
COPY requirements.txt ./requirements.txt

# 3) Create Conda environment (make sure environment.yml name == "momask-plus")
RUN conda env create -f environment.yml

# 4) Add Conda env bin to PATH
ENV PATH=/opt/conda/envs/momask-plus/bin:$PATH

# 5) Use Conda environment shell from this point forward
SHELL ["conda", "run", "-n", "momask-plus", "/bin/bash", "-c"]

# 6) Install Python deps from requirements.txt (inside the env)
RUN pip install -r requirements.txt

# 7) (Optional) Pin/Install PyTorch CUDA build explicitly (if not in requirements.txt)
#    Comment out if your requirements.txt already covers torch/torchvision/torchaudio
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 8) Install a few extra Python utilities (if not in requirements.txt)
RUN pip install ftfy regex tqdm gdown

# 9) Now copy the rest of your project and KeeMap add-on
COPY . .
COPY KeeMapAnimRetarget /opt/KeeMapAnimRetarget

# 10) Install Blender and system dependencies
RUN apt-get update && \
    apt-get install -y \
      blender \
      xvfb \
      libglu1-mesa \
      libegl1 \
      libgl1-mesa-glx \
      libxrender1 \
      libxrandr2 \
      libxinerama1 \
      libxcursor1 \
      libxi6 \
      git \
      wget \
      curl \
      unzip \
      ffmpeg \
      python3-numpy \
      dos2unix && \
    apt-get clean && \
    sed -i "s/open(file_path, 'rU')/open(file_path, 'r')/g" \
      /usr/share/blender/scripts/addons/io_anim_bvh/import_bvh.py

# 11) Move KeeMap addon into Blender user addons directory
RUN BLVER=$(blender --version | head -n1 | awk '{print $2}' | cut -d'.' -f1,2) && \
    ADDON_DIR="/root/.config/blender/${BLVER}/scripts/addons/KeeMapAnimRetarget" && \
    mkdir -p "$ADDON_DIR" && \
    cp -r /opt/KeeMapAnimRetarget/* "$ADDON_DIR"

# 12) Normalize and run prepare scripts
RUN dos2unix prepare/*.sh && \
    chmod +x prepare/*.sh && \
    bash prepare/download_models.sh && \
    bash prepare/download_evaluators.sh && \
    bash prepare/download_glove.sh

# 13) Expose FastAPI port and run server
EXPOSE 8000
CMD ["conda","run","-n","momask-plus","--no-capture-output","uvicorn","app_server:app","--host","0.0.0.0","--port","8000"]
