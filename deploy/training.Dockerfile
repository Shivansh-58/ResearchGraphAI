FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
WORKDIR /project
COPY training/requirements-qlora.txt training/requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements-qlora.txt
COPY training/ ./training/
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_DISABLED=true
