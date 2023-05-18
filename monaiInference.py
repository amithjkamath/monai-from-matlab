from PIL import Image
#import numpy as np # this mysteriously errors when importing lib.

import torch
import monai

from monai.data import (
    decollate_batch,
)
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
)


def create_model(model_path):
    config = {
        "unet_model_params": dict(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2, 2),
            num_res_units=0,
            norm=Norm.BATCH,
            act="ReLU",
            bias=False,
        ),
    }
    model = UNet(**config["unet_model_params"])
    model.load_state_dict(torch.load(model_path))
    return model


def inference(model, img):
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    model.eval()

    with torch.no_grad():
        test_outputs = model(img)
        test_outputs = [post_trans(i) for i in decollate_batch(test_outputs)]

    return test_outputs[0]


def test_model():
    # To run this, please uncomment the numpy import line above.
    # to run in MATLAB, comment it back again: the library doesn't import otherwise.
    
    model = create_model(
        "./heart-2d-model.pth"
    )
    img = Image.open(
        "./data/images/image_0000.png"
    )
    img_array = np.array(img)
    img_array = img_array.reshape([1, 1, 256, 256])
    img_tensor = torch.Tensor(img_array)
    seg = inference(model, img_tensor)
    seg_array = np.array(seg[0]).squeeze()
    seg_array = 255 * seg_array  # Now scale by 255
    seg_array = seg_array.astype(np.uint8)
    seg_img = Image.fromarray(seg_array)
    seg_img.save("seg.png")


if __name__ == "__main__":
    test_model()
