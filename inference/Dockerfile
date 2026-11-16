# Inference image: RTMDet-x person detection followed by RTMW3D-x whole-body 3D pose.
# Library versions and weights are the ones the reference results were produced with, so that
# fp32 output matches the reference to the last decimal. Nothing is downloaded at run time.
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip install "numpy<2" "opencv-python-headless<4.11" mmengine==0.10.7 \
    && pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html \
    && pip install mmdet==3.3.0 mmpose==1.3.2 aiohttp==3.9.5 boto3==1.34.162

# RTMPose3D lives in the MMPose repository's projects/ directory, not in the pip package.
ARG MMPOSE_COMMIT=759b39c13fea6ba094afc1fa932f51dc1b11cbf9
RUN git clone https://github.com/open-mmlab/mmpose.git /opt/mmpose \
    && cd /opt/mmpose && git checkout -q ${MMPOSE_COMMIT} && rm -rf .git
ENV PYTHONPATH=/opt/mmpose/projects/rtmpose3d

# Public weights, verified against the published checksums.
RUN mkdir -p /models && cd /models \
    && wget -q https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_x_8xb32-300e_coco/rtmdet_x_8xb32-300e_coco_20220715_230555-cc79b9ae.pth \
    && wget -q https://download.openmmlab.com/mmpose/v1/wholebody_3d_keypoint/rtmw3d/rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.pth \
    && echo "cc79b9ae0970f7f597e2aa5f81663129f3aa24e5612549b4f1a383da7fff825d  rtmdet_x_8xb32-300e_coco_20220715_230555-cc79b9ae.pth" | sha256sum -c - \
    && echo "b0a0eab7ca501e2c88028fe1cfd78b50c7130e220d39541da7f669e5de25e6ca  rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.pth" | sha256sum -c -

COPY engine.py admission.py server.py /app/

# The container runs as a non-root user on a read-only root filesystem; only /tmp is writable.
ENV HOME=/tmp XDG_CACHE_HOME=/tmp MPLCONFIGDIR=/tmp TORCH_HOME=/tmp
USER 10001
WORKDIR /app
EXPOSE 8080
CMD ["python", "/app/server.py"]
