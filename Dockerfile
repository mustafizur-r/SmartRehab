# 0) Base image
FROM continuumio/miniconda3
LABEL authors="mdmustafizur-r"

# 1) Set working directory
WORKDIR /workspace

# 2) Copy env specs early for better Docker cache
COPY environment.yml ./environment.yml
COPY requirements.txt ./requirements.txt

# 3) Create Conda environment (Linux-safe minimal env)
RUN conda env create -f environment.yml \
 && conda clean -afy

# 4) Add Conda env bin to PATH
ENV PATH=/opt/conda/envs/smartrehab/bin:$PATH
ENV CONDA_DEFAULT_ENV=smartrehab

# 5) Install Blender and system dependencies
SHELL ["/bin/bash", "-lc"]
RUN apt-get update && \
    apt-get install -y \
      blender \
      xvfb \
      libglu1-mesa \
      libegl1 \
      libgl1 \
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
    find /usr/share/blender -name "import_bvh.py" -exec \
      sed -i "s/open(file_path, 'rU')/open(file_path, 'r')/g" {} \; || true

# 5b) Install numpy into Blender's own Python (python3.13)
RUN /opt/conda/bin/python3.13 -m pip install numpy

# 6) Switch to conda env shell for Python installs
SHELL ["conda", "run", "-n", "smartrehab", "/bin/bash", "-c"]

# 7) Install ffmpeg codecs via conda
RUN conda install -y -c conda-forge "ffmpeg>=4.3,<6" "openh264=2.1.*" imageio-ffmpeg

# 8) Install all Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# 9) Install PyTorch with CUDA 12.1
RUN pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu121

# 10) Copy project
COPY . .

# 11) Install KeeMap add-on into Blender
RUN BLVER=$(blender --background --version 2>/dev/null | head -n1 | awk '{print $2}' | cut -d'.' -f1,2) && \
    echo "Blender version: $BLVER" && \
    ADDON_DIR="/root/.config/blender/${BLVER}/scripts/addons/KeeMapAnimRetarget" && \
    mkdir -p "$ADDON_DIR" && \
    cp -r KeeMapAnimRetarget/* "$ADDON_DIR"

# 12) Normalize and run prepare scripts
RUN dos2unix prepare/*.sh && \
    chmod +x prepare/*.sh && \
    conda run -n smartrehab bash prepare/download_avatar_model_fbx.sh && \
    conda run -n smartrehab bash prepare/download_models.sh && \
    conda run -n smartrehab bash prepare/download_evaluators.sh && \
    conda run -n smartrehab bash prepare/download_glove.sh && \
    conda run -n smartrehab bash prepare/download_preparedata.sh

# 13) Expose FastAPI port and run server
EXPOSE 8000
CMD ["conda", "run", "-n", "smartrehab", "--no-capture-output", \
     "uvicorn", "app_server:app", "--host", "0.0.0.0", "--port", "8000"]