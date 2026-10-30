# Builder image for llama.cpp CUDA builds. Mirrors the build stage of
# .devops/cuda.Dockerfile at llama.cpp b12004 (same base image, same apt packages),
# with the base pinned by digest and ccache added.
# Base: docker.io/nvidia/cuda:12.8.1-devel-ubuntu24.04
# The digest below is the multi-arch index. The linux/amd64 manifest is
# sha256:4b9ed5fa8361736996499f64ecebf25d4ec37ff56e4d11323ccde10aa36e0c43.
FROM docker.io/nvidia/cuda@sha256:520292dbb4f755fd360766059e62956e9379485d9e073bbd2f6e3c20c270ed66

ARG GCC_VERSION=14
LABEL dmk.base.index="sha256:520292dbb4f755fd360766059e62956e9379485d9e073bbd2f6e3c20c270ed66" \
      dmk.base.amd64="sha256:4b9ed5fa8361736996499f64ecebf25d4ec37ff56e4d11323ccde10aa36e0c43" \
      dmk.base.tag="docker.io/nvidia/cuda:12.8.1-devel-ubuntu24.04"

RUN apt-get update && \
    apt-get install -y gcc-${GCC_VERSION} g++-${GCC_VERSION} build-essential cmake python3 python3-pip git libssl-dev libgomp1 ccache && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    git config --system --add safe.directory '*'

ENV CC=gcc-${GCC_VERSION} CXX=g++-${GCC_VERSION} CUDAHOSTCXX=g++-${GCC_VERSION}
WORKDIR /out
