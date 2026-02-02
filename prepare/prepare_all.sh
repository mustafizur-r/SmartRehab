#!/usr/bin/env bash
set -e  # stop if any command fails

bash prepare/download_avatar_model_fbx.sh
bash prepare/download_models.sh
bash prepare/download_evaluators.sh
bash prepare/download_glove.sh
bash prepare/download_preparedata.sh

echo "All installation steps completed successfully."