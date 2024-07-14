FROM nvcr.io/nvidia/tritonserver:24.05-py3 AS runtime

# Install PyTorch and other dependencies
RUN pip install torch torchvision torchaudio open_clip_torch

COPY ./models /models
COPY ./model_cache /model_cache
