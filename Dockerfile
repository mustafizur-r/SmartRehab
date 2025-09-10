# 0) Base image
FROM continuumio/miniconda3
LABEL authors="mdmustafizur-r"

# 1) Set working directory
WORKDIR /workspace

# 2) Copy env specs early for better Docker cache
COPY environment.yml ./environment.yml
COPY requirements.txt ./requirements.txt

# 3) Create Conda environment (make sure environment.yml is Linux-safe)
RUN conda config --set channel_priority strict \
 && conda env create -f environment.yml \
 && conda clean -afy

# 4) Add Conda env bin to PATH
ENV PATH=/opt/conda/envs/momask-plus/bin:$PATH
ENV CONDA_DEFAULT_ENV=momask-plus

# Keep shell as bash for system packages first
SHELL ["/bin/bash", "-lc"]

# 5) Install Blender and system dependencies BEFORE switching to conda shell
# 6) Install Blender and system dependencies
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

# 6) Now switch to conda env shell for Python installs
SHELL ["conda", "run", "-n", "momask-plus", "/bin/bash", "-c"]

RUN conda install -y -c conda-forge "ffmpeg>=4.3,<6" "openh264=2.1.*" imageio-ffmpeg

# 7) (Optional) Install PyTorch CUDA build explicitly (uncomment if not in requirements.txt)
# RUN python -m pip install --upgrade pip \
#  && python -m pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
#       torch torchvision torchaudio

# 8) Install Python deps from requirements.txt (inside the env) — uncomment if you use it
# RUN pip install -r requirements.txt

# 9) Copy project (this already includes KeeMapAnimRetarget if it’s in the repo)
COPY . .

# 10) Install KeeMap add-on into Blender user addons directory
RUN BLVER=$(blender --version | head -n1 | awk '{print $2}' | cut -d'.' -f1,2) && \
    ADDON_DIR="/root/.config/blender/${BLVER}/scripts/addons/KeeMapAnimRetarget" && \
    mkdir -p "$ADDON_DIR" && \
    cp -r KeeMapAnimRetarget/* "$ADDON_DIR"

# 11) Normalize and run prepare scripts
RUN dos2unix prepare/*.sh && \
    chmod +x prepare/*.sh && \
    bash prepare/download_models.sh && \
    bash prepare/download_evaluators.sh && \
    bash prepare/download_glove.sh && \
    bash prepare/download_preparedata.sh

#    bash prepare/download_humanml3d_dataset.sh && \
#    bash prepare/download_snapmogen_dataset.sh


# 12) Expose FastAPI port and run server
EXPOSE 8000
CMD ["conda","run","-n","momask-plus","--no-capture-output","uvicorn","app_server:app","--host","0.0.0.0","--port","8000"]
