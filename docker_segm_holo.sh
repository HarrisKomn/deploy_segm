
export DISPLAY=:1

xhost +local:docker

sudo apt-get install -y libcanberra-gtk-module
pip install mss

# new container for screen visualization
docker run \
    --gpus all \
    -it --rm \
    --network host \
    --runtime=nvidia \
    --ipc=host --cap-add=CAP_SYS_PTRACE --ulimit memlock=-1 --ulimit stack=67108864 \
    -v /media/**/DATA:/cotracker/DATA \
    -v /home/**/co-tracker:/cotracker/co-tracker \
    -v /media/**/onnx_tensorrt:/cotracker/onnx_tensorrt \
    -v "/tmp/.X11-unix:/tmp.X11-unix:rw" \
    -e "DISPLAY=$DISPLAY" \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e __NV_PRIME_RENDER_OFFLOAD=1 \
    -e __GLX_VENDOR_LIBRARY_NAME=nvidia \
    --device /dev/video0 \
    multsegm_image_holo:latest

