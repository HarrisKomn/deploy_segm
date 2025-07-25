# Deployment of deep learning algorithms

## Intro
This repo provides code for deploying deep learning algorithms in a clinical setup using NVIDIA Clara AGX device.
It uses a single docker image and container for the inference. Pytorch and holoscan  are installed in the same container.

## Hardware connections
To provide input to Clara AGX for streaming and processing, connect the HDMI input of your source to Clara's bottom HDMI port.
In our hardware setup we used the output of an ophthalmic surgical microscope 
as input to Clara AGX.


## Installation
Create a docker image which contains the dependencies of segmentation models (pytorch, opencv, smp) and holoscan processing:

```docker build --progress=plain -t multsegm_image_holo . -f multsegm_holoscan.Dockerfile```

Before running the docker container, edit the ```docker_segm_holo.sh```. Several arguments need to be pass as shown below.
To create the image, you need to replace the ```-v``` paths with yours. We provide examples. 

``` 
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
    
```

In the terminal, run: 
```
sh docker_segm_holo.sh
```



## Run the application of multiclass segmentation
In the terminal, run:
```
sh docker_segm_holo.sh
```

Inside the container, run:
```
cd co-tracker
python multclas_segm_VR_holo.py
```




## Acknowledgments
This repo heavily relies on the following codebases: \
NVIDIA HOLOSCAN (https://github.com/nvidia-holoscan/holoscan-sdk) \
https://github.com/facebookresearch/co-tracker \
https://github.com/qubvel-org/segmentation_models.pytorch