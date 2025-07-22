"""
This script has been obtained by https://github.com/nvidia-holoscan/holoscan-sdk
and modified for our application (research purposes).
"""

import os
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.operators import HolovizOp, VideoStreamReplayerOp, FormatConverterOp
from holoscan.operators import HolovizOp, V4L2VideoCaptureOp
from holoscan.resources import BlockMemoryPool, UnboundedAllocator
try:
    import cupy as cp
    import cupyx.scipy.ndimage as ndi
except ImportError:
    raise ImportError(
        "CuPy must be installed to run this example. See "
        "https://docs.cupy.dev/en/stable/install.html"
    ) from None

import numpy as np
import cv2

# segmentation imports
import torch
import segmentation_models_pytorch as smp
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image


def overlay_mask_mult(image: torch.Tensor, mask: torch.Tensor, alpha: float = 0.1):
    """
    Overlay a multi-class mask on an image.

    :param image: Tensor of shape (1, H, W, 3) representing the image.
    :param mask: Tensor of shape (1, 4, H1, W1) representing the multi-class mask.
    :param alpha: Transparency factor for overlay.
    :return: Overlayed image as a NumPy array.
    """

    # Get the height and width of the image
    h, w = image.shape[1], image.shape[2]
    print(image.max(), image.min())

    # Step 1: Get the class with the highest score in the mask (argmax across the class dimension)
    combined_mask = torch.argmax(mask, dim=1).unsqueeze(0)  # Shape (H1, W1)

    # Step 2: Resize the mask to match the image size using torch interpolation
    combined_mask = F.interpolate(combined_mask.float(), size=(h, w), mode='nearest').squeeze(0).long()

    # Step 3: Normalize mask values and apply the colormap (viridis)
    # Instead of using cm.viridis (from Matplotlib), we will use PyTorch to apply the colormap.
    # colormap = torch.tensor([
    #     [0, 0, 255],  # Class 0 - Red
    #     [0, 255, 0],  # Class 1 - Green
    #     [255, 0, 0],  # Class 2 - Blue
    #     [255, 255, 0]  # Class 3 - Yellow
    # ], dtype=torch.uint8, device='cuda')

    colormap = torch.tensor([
        [70, 70, 70],  # Class 0 - outer retina
        [0, 180, 180],  # Class 1 - retina
        [255, 0, 255],  # Class 2 - right tool
        [255, 0, 0],  # Class 3 - left tool
    ], dtype=torch.uint8, device='cuda')

    # Map each pixel in the combined mask to its corresponding color
    color_mask = colormap[combined_mask]  # Shape (H, W, 3)

    # Step 4: Overlay the mask onto the image
    # The mask is overlayed onto the image using alpha blending.
    overlayed_image = (alpha * color_mask.float() + (1 - alpha) * 255.*image.squeeze(0)).byte()

    return overlayed_image

def get_vis_transform(resize_img_w, resize_img_h):
    list_trans=[]

    list_trans.extend([
                        transforms.Resize((resize_img_w, resize_img_h)),
                        transforms.ToTensor(), 
                    ])

    list_trans=transforms.Compose(list_trans)
    return list_trans

def get_test_transform(resize_img_w, resize_img_h, mean, std):
    list_trans=[]

    list_trans.extend([
                        transforms.Resize((resize_img_w, resize_img_h)),
                        transforms.ToTensor(), 
                        transforms.Normalize(mean=mean, std=std),
                    ])

    list_trans=transforms.Compose(list_trans)
    return list_trans

class Options:
    def __init__(self):
        self.model = "Segformer" # Unet, DeepLabV3Plus, UPerNet
        self.encoder = "mit_b0"  # resnet34, resnet50, resnet101,  mobilenet_v2

        self.batch_size = 1

        self.resize_img_w = 768
        self.resize_img_h = 768
        self.phase = 'test' # val, test, test_man
        self.global_threshold = 0.7
        self.type = "mult" # mult, bin

def preprocess_img(img_np, transorm):

    img_tns = transorm(img_np)
    return img_tns

class MultClassSegmOp(Operator):
    """
    Multiclass segmentation operator processing input tensors.

    This operator has:
        inputs:  "input_tensor"
        outputs: "output_tensor"

    The data from each input pass through a segmentation model and
    the result is sent to the output.
    """

    def __init__(self, fragment, *args, **kwargs):
        self.count = 1

        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)


    def visual_frame(self, torch_tensor, resize_img_w, resize_img_h):
        torch_tensor = (torch_tensor / 255.0)
        torch_tensor = torch_tensor.permute(2,0,1)
        torch_tensor = torch_tensor[:3, :, :]           # (1080, 1920, 3)
        torch_tensor = torch_tensor[[2, 1, 0], :, :] # bgr to rgb
        torch_tensor = torch_tensor.unsqueeze(0)        # (1, 3, 1080, 1920)
        tensor_resized = F.interpolate(torch_tensor, size=(resize_img_w, resize_img_h), 
                                        mode='bilinear', align_corners=False).squeeze(0)


        out_tensor = tensor_resized.unsqueeze(0).permute(0,2,3,1)
        print(torch_tensor.shape)        # torch.Size([3, 224, 224])
        print(torch_tensor.device)       # cuda:0
        print(torch_tensor.dtype)        # torch.float32

        return out_tensor


    def preprocess_frame(self, torch_tensor, resize_img_w, resize_img_h):
            
        torch_tensor = torch_tensor.permute(2, 0, 1)    # (4, 1080, 1920)
        torch_tensor = torch_tensor / 255.0
        torch_tensor = torch_tensor[:3, :, :]           # (3, 1080, 1920)
        torch_tensor = torch_tensor[[2, 1, 0], :, :] # bgr to rgb

        torch_tensor = torch_tensor.unsqueeze(0)        # (1, 3, 1080, 1920)
        tensor_resized = F.interpolate(torch_tensor, size=(resize_img_w, resize_img_h), 
                                        mode='bilinear', align_corners=False).squeeze(0)

        print(torch_tensor.shape)        # torch.Size([3, 224, 224])
        print(torch_tensor.device)       # cuda:0
        print(torch_tensor.dtype)        # torch.float32


        # Step 4: Normalize the tensor (mean, std are on GPU)
        mean = torch.tensor([0.485, 0.456, 0.406], device=tensor_resized.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=tensor_resized.device).view(1, 3, 1, 1)
        normalized_tensor = (tensor_resized - mean) / std

        return normalized_tensor
    
    def postprocess_frame(self, input_tensor):

        input_tensor = input_tensor.float() / 255.0  # Now float32 in range [0, 1]
        input_tensor = input_tensor.permute(0, 3, 1, 2)  # (1, 3, 512, 512)
        resized_tensor = F.interpolate(input_tensor, size=(1080, 1920), mode='bilinear', align_corners=False)
        resized_tensor = (resized_tensor * 255.0).clamp(0, 255).to(torch.uint8)
        resized_tensor = resized_tensor.permute(0, 2, 3, 1)  # (1, 1080, 1920, 3), uint8

        return resized_tensor
    


    def setup(self, spec: OperatorSpec):
        spec.input("input_tensor")
        spec.output("output_tensor")

        self.opt = Options()

        #self.vis_transform = get_vis_transform(self.opt.resize_img_w, self.opt.resize_img_h)

        self.device = torch.device("cuda")

        opt = Options()

        ####################################################################################
        checkpoint_folder = "DGX_SIMPLE_2025-05-08_20-25-39" # [mit_b0]
        ####################################################################################

        ckpt_path = f"./checkpoints/{checkpoint_folder}/segm_model.pth"

        device = torch.device("cuda")

        if opt.model == 'Unet':
            model = smp.Unet(opt.encoder, encoder_weights=None, classes=4, activation=None)

        elif opt.model == 'DeepLabV3Plus':
            model = smp.DeepLabV3Plus(opt.encoder, encoder_weights=None, classes=4, activation=None)

        elif opt.model == 'Segformer':
            model = smp.Segformer(opt.encoder, encoder_weights=None, classes=4, activation=None)

        model.to(device)
        model.eval()
        state = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(state["state_dict"])

        self.model = model




    def compute(self, op_input, op_output, context):

        in_message = op_input.receive("input_tensor")

        # out_message will be a dict of tensors
        out_message = dict()

        for key, value in in_message.items():
            self.count += 1

            cp_array_np = cp.asarray(value)

            # cp_array_np_cpu = cp.asnumpy(cp_array_np)
            # cp_array_np_cpu = cp_array_np_cpu[:, :, :3]
            # cp_array_np_PIL = Image.fromarray(cp_array_np_cpu, mode='RGB')
            
            # image_tns = preprocess_img(cp_array_np_PIL, self.test_transform)

            # image_tns = image_tns.unsqueeze(0).to(self.device)

            torch_tensor = torch.utils.dlpack.from_dlpack(cp_array_np.toDlpack()) # (1080, 1920, 4)
            
            # Apply transforms for preprocessing
            normalized_tensor = self.preprocess_frame(torch_tensor, self.opt.resize_img_w, 
                                                              self.opt.resize_img_h)

            # Apply transforms for visualisation
            input_img_tns = self.visual_frame(torch_tensor, self.opt.resize_img_w,  self.opt.resize_img_h)


            pred_masks = self.model(normalized_tensor)
            pred_mask = torch.softmax(pred_masks, dim=1)


            result = overlay_mask_mult(input_img_tns, pred_mask, 0.4)

            result = self.postprocess_frame(result)
            overlayed_image_np = result.cpu().numpy()[0]

            # cv2.namedWindow('prediction')
            cv2.imshow("Prediction", overlayed_image_np)
            cv2.waitKey(1)
            
            
            
            #cp_array_np = (cp_array*255).astype(np.uint8)

            # THis is how we visualise the image
            # cv2.imshow('BGRA img', cv2.cvtColor(cp_array_np.get(),cv2.COLOR_BGRA2RGB))
            # cv2.waitKey(1)

            print(self.count) 

            out_message[key] = cp_array_np

        
        op_output.emit(out_message, "output_tensor")





class MultClassSegmApp(Application):
    """ Segment multiple classes in the current frame of a video stream during 
    vitreoretinal (VR) surgery.
    
    This application has the following operators:

    - V4L2VideoCaptureOp
    - MultClassSegmOp
    - HolovizOp

    The V4L2VideoCaptureOp reads streams and send the frames to MultClassSegmOp for processing.
    The MultClassSegmOp performs multiclass segmentaton.
    The HolovizOp displays the processed frames.
    """

    def compose(self):

        # V4L2VideoCaptureOp 
        source = V4L2VideoCaptureOp(
            self,
            name="source",
            allocator=UnboundedAllocator(self, name="pool"),
            **self.kwargs("source"),
        )

        # HolovizOp
        visualizer = HolovizOp(
            self,
            name="visualizer",
            **self.kwargs("visualizer"),
        )

        # FormatConverterOp
        converter = FormatConverterOp(
            self, 
            name="converter", 
            pool=UnboundedAllocator(self, name="pool"), 
            out_dtype="uint8")
        

        # MultClassSegmOp
        image_processing = MultClassSegmOp(
            self, 
            name="image_processing", 
            pool=UnboundedAllocator(self, name="pool"), 
            **self.kwargs("image_processing")
        )

        self.add_flow(source, converter)
        self.add_flow(converter, image_processing)
        self.add_flow(image_processing, visualizer, {("output_tensor", "receivers")})


    

def main(config_file):

    app = MultClassSegmApp()

    app.config(config_file)
    app.run()
    print("Application has finished running.")



if __name__ == "__main__":

    config_file = os.path.join(os.path.dirname(__file__), "./v4l2_camera.yaml")
    print(config_file)
    main(config_file=config_file)

        
