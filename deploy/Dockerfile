# spatio-flux worker image
# Built by scripts/build-and-push-image.sh; pushed to GovCloud ECR.
# Pulled by Ray workers in the cluster (configured in
# ray-govcloud-cluster.yaml docker: block).
#
# Build context expects:
#   src/        spatio-flux source tree (everything except .venv, out, .git)
#   comets/     COMETS install dir (the jar + scripts)

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps. One RUN to keep the layer count down. The whole point of
# baking is to do this slow apt+download step ON A FAST INTERNET MACHINE
# (your laptop), not on every cluster bring-up.
RUN apt-get update -y && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        openjdk-17-jre-headless \
        build-essential \
        curl \
        ca-certificates \
        awscli \
        rsync \
        openssh-client \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# uv — single static binary, no venv install needed
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv \
    && mv /root/.local/bin/uvx /usr/local/bin/uvx

# COMETS Java jar + supporting files
RUN mkdir -p /opt/comets
COPY comets/ /opt/comets/
ENV COMETS_HOME=/opt/comets

# spatio-flux source. Build venv + install everything via uv into a
# stable path that's first on PATH for the entrypoint.
RUN mkdir -p /app
COPY src/ /app/spatio-flux/
WORKDIR /app/spatio-flux
RUN uv venv --python 3.11 /opt/spatio-flux-venv \
    && uv pip install --python /opt/spatio-flux-venv/bin/python \
       -e . "ray[default]>=2.10" cometspy

# Put the venv on PATH so `ray`, `python`, etc. just work.
ENV PATH="/opt/spatio-flux-venv/bin:${PATH}" \
    PYTHONPATH="/app/spatio-flux"

# Ray's autoscaler runs `ray start --head ...` / `ray start --address=...`
# inside the container via its head_start/worker_start commands, so we
# don't set a CMD here.
