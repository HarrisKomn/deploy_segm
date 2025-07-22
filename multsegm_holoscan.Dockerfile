# python 3.8
#FROM nvcr.io/nvidia/pytorch:23.02-py3 
# python 3.10
FROM nvcr.io/nvidia/pytorch:23.05-py3 AS pytorch

# Install dependencies
RUN apt-get update && apt-get install -y \
    git \   
    git-lfs \ 
    x11-apps \
    && rm -rf /var/lib/apt/lists/*

# Set up Git LFS
RUN git lfs install

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive TZ=Europe/London apt-get install -y --no-install-recommends tzdata

#RUN DEBIAN_FRONTEND=noninteractive TZ=Europe/London apt-get -y install tzdata
RUN apt-get install -y unzip git python3-setuptools 

# set the working directory
WORKDIR /cotracker


# Check the architecture and download the CUDA keyring
RUN if [ $(uname -m) = "aarch64" ]; then ARCH=arm64 \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/sbsa/cuda-keyring_1.1-1_all.deb \
    ; elif [ $(uname -m) = "x86_64" ]; then ARCH=x86_64 \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb \
    ; else echo "Unsupported architecture"; fi
RUN dpkg -i cuda-keyring_1.1-1_all.deb \
    && apt-get update



# Setup Docker & NVIDIA Container Toolkit's apt repositories to enable DooD
# for packaging & running applications with the CLI
# Ref: Docker installation: https://docs.docker.com/engine/install/ubuntu/
# DooD (Docker-out-of-Docker): use the Docker (or Moby) CLI in your dev container to connect to
#  your host's Docker daemon by bind mounting the Docker Unix socket.
RUN install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg \
    && chmod a+r /etc/apt/keyrings/docker.gpg \
    && echo "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
        "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
        tee /etc/apt/sources.list.d/docker.list > /dev/null


RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        valgrind="1:3.18.1-*" \
        xvfb="2:21.1.4-*" \
        libx11-dev="2:1.7.5-*" \
        libxcb-glx0="1.14-*" \
        libxcursor-dev="1:1.2.0-*" \
        libxi-dev="2:1.8-*" \
        libxinerama-dev="2:1.1.4-*" \
        libxrandr-dev="2:1.5.2-*" \
        libvulkan-dev \
        glslang-tools="11.8.0+1.3.204.0-*" \
        vulkan-validationlayers \
        libwayland-dev="1.20.0-*" \
        libxkbcommon-dev="1.4.0-*" \
        pkg-config="0.29.2-*" \
        libdecor-0-plugin-1-cairo="0.1.0-*" \
        libegl1="1.4.0-*" \
        libopenblas0="0.3.20+ds-*" \
        libv4l-dev="1.22.1-*" \
        v4l-utils="1.22.1-*" \
        libpng-dev="1.6.37-*" \
        libjpeg-turbo8-dev="2.1.2-*" \
        docker-ce-cli="5:25.0.3-*" \
        docker-buildx-plugin="0.12.1-*" \
    && rm -rf /var/lib/apt/lists/*


RUN apt-get update && apt-get install -y vulkan-tools

RUN pip3 install segmentation-models-pytorch

# Install Holoscan
RUN apt-get update \
    && apt-get -y install holoscan

RUN pip3 install holoscan==2.2.0.post1


RUN apt-get install -y libgtk2.0-dev libgtk-3-dev

# # RUN export OPENCV_TAG=4.6.0 && \
RUN export OPENCV_TAG=4.10.0 && \
    git clone https://github.com/opencv/opencv.git -b ${OPENCV_TAG} && \
    git clone https://github.com/opencv/opencv_contrib.git -b ${OPENCV_TAG} && \
    cd opencv && mkdir build && cd build && \
    cmake \
        -DBUILD_PERF_TESTS=OFF \
        -DBUILD_TESTS=OFF \
        -DWITH_QT=OFF \
        -DWITH_GTK=ON \
        -DWITH_GTK_2_X=ON \
        -DBUILD_opencv_aruco=OFF \
        -DBUILD_opencv_barcode=OFF \
        -DBUILD_opencv_bgsegm=OFF \
        -DBUILD_opencv_bioinspired=OFF \
        -DBUILD_opencv_calib3d=OFF \
        -DBUILD_opencv_ccalib=OFF \
        -DBUILD_opencv_datasets=OFF \
        -DBUILD_opencv_dnn=OFF \
        -DBUILD_opencv_dnn_objdetect=OFF \
        -DBUILD_opencv_dnn_superres=OFF \
        -DBUILD_opencv_dpm=OFF \
        -DBUILD_opencv_face=OFF \
        -DBUILD_opencv_features2d=OFF \
        -DBUILD_opencv_flann=OFF \
        -DBUILD_opencv_fuzzy=OFF \
        -DBUILD_opencv_gapi=OFF \
        -DBUILD_opencv_hdf=OFF \
        -DBUILD_opencv_hfs=OFF \
        -DBUILD_opencv_intensity_transform=OFF \
        -DBUILD_opencv_java_bindings_generator=OFF \
        -DBUILD_opencv_js=OFF \
        -DBUILD_opencv_js_bindings_generator=OFF \
        -DBUILD_opencv_mcc=OFF \
        -DBUILD_opencv_ml=OFF \
        -DBUILD_opencv_objc_bindings_generator=OFF \
        -DBUILD_opencv_objdetect=OFF \
        -DBUILD_opencv_optflow=OFF \
        -DBUILD_opencv_phase_unwrapping=OFF \
        -DBUILD_opencv_photo=OFF \
        -DBUILD_opencv_plot=OFF \
        -DBUILD_opencv_python_tests=OFF \
        -DBUILD_opencv_quality=OFF \
        -DBUILD_opencv_rapid=OFF \
        -DBUILD_opencv_reg=OFF \
        -DBUILD_opencv_rgbd=OFF \
        -DBUILD_opencv_saliency=OFF \
        -DBUILD_opencv_shape=OFF \
        -DBUILD_opencv_stereo=OFF \
        -DBUILD_opencv_stitching=OFF \
        -DBUILD_opencv_structured_light=OFF \
        -DBUILD_opencv_superres=OFF \
        -DBUILD_opencv_surface_matching=OFF \
        -DBUILD_opencv_text=OFF \
        -DBUILD_opencv_tracking=OFF \
        -DBUILD_opencv_video=ON \
        -DBUILD_opencv_videoio=ON \
        -DBUILD_opencv_videostab=OFF \
        -DBUILD_opencv_wechat_qrcode=OFF \
        -DBUILD_opencv_xfeatures2d=OFF \
        -DBUILD_opencv_ximgproc=OFF \
        -DBUILD_opencv_xobjdetect=OFF \
        -DBUILD_opencv_xphoto=OFF \
        -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules .. && \
    make -j5 && \
    make -j5 install



## TODO: someshow get to install opencv correctly to python OR extend the path
# RUN echo "export PYTHONPATH=$PYTHONPATH:/cotracker/opencv/build/lib/python3" >> $HOME/.bashrc
ENV PYTHONPATH=$PYTHONPATH:/cotracker/opencv/build/lib/python3

# RUN apt-get update && apt-get install -y git

RUN pip3 install scipy
RUN pip3 install matplotlib
RUN pip3 install mss



