# To check what CUDA and cuDNN version your host uses: https://gist.github.com/fauxneticien/343b1dd7b68a30cb6f8983dacac28721
# Then, check what base image you need (e.g. 21.05) from https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html
#
# Container 21.05 = CUDA 11.3.0 and cuDNN = 8.2.0.51 insider the container
# These need to match the CUDA and cuDNN versions of the host

FROM nvcr.io/nvidia/pytorch:21.05-py3

# Update container PyTorch from 1.9 to 1.10
RUN conda install pytorch==1.10.1 \
    torchvision==0.11.2 \
    torchaudio==0.10.1 \
    cudatoolkit=11.3 \
    -c pytorch \
    -c conda-forge

# Make existing packages play nice with
# those to be installed below
RUN pip uninstall -y torchtext && \
    # ruamel.yaml required by pyannote
    conda install -y ruamel.yaml

# Dependencies for KenLM
# Adapted from https://github.com/mpenagar/kenlm-docker/blob/master/Dockerfile.debian
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y build-essential=12.8ubuntu1.1 \
    cmake=3.16.3-1ubuntu1 \
    libboost-system-dev=1.71.0.0ubuntu2 \
    libboost-thread-dev=1.71.0.0ubuntu2 \
    libboost-program-options-dev=1.71.0.0ubuntu2 \
    libboost-test-dev=1.71.0.0ubuntu2 \
    libeigen3-dev=3.3.7-2 \
    zlib1g-dev=1:1.2.11.dfsg-2ubuntu1.3 \
    libbz2-dev=1.0.8-2 \
    liblzma-dev=5.2.4-1ubuntu1.1

# Make it easy to fetch large files from cloud storage
RUN apt-get install -y rclone

WORKDIR /tmp

RUN git clone https://github.com/kpu/kenlm

# Build bindings for KenLM
RUN cd kenlm && \
    git checkout 0760f4c4df76f3286656e7232dc3ad6495248bc2 && \
    mkdir -p build && \
    cd build && \
    cmake .. && \
    make -j `nproc` && \
    cp bin/lmplz /usr/bin

RUN pip install datasets==2.0.0 \
    pandas==1.4.2 \
    pympi-ling==1.70.2 \
    speechbrain==0.5.11 \
    jiwer==2.3.0 \
    # Keep torch requirements here to stop pyannote from upgrading to latest versions \
    torch==1.10.1 \
    torchaudio==0.10.1 \
    tqdm==4.62.3 \
    transformers==4.18.0 \
    scikit-learn==1.0.2 \
    # Install specific version of pyannote (which plays nice with SpeechBrain and Transformers) \
    git+https://github.com/pyannote/pyannote-audio@b84d5431a1ac494e60e5d74a4bd8a45f44698f1f \
    # (unlisted) pyannote dependencies \
    numba==0.55.1 \
    torch-pitch-shift==1.2.2 \
    packaging>=21.3 \ 
    # KenLM for language models
    git+https://github.com/kpu/kenlm@0760f4c4df76f3286656e7232dc3ad6495248bc2 \
    pyctcdecode==0.3.0

WORKDIR /workspace
