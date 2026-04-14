FROM vae_working_uid1000:lfs

USER root
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    python3-pip \
  && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir \
    "numpy<2" \
    "pandas<3" \
    "openpyxl<4" \
    "matplotlib<3.9" \
    "pytest<10"

USER 1000:1000
WORKDIR /workspace
