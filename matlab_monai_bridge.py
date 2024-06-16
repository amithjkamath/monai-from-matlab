import os
import re

import torch

from monai.data import NibabelWriter
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.transforms import (
    Activations,
    AsDiscrete,
    EnsureChannelFirst,
    Compose,
    CropForeground,
    LoadImage,
    Orientation,
    ScaleIntensity,
    Spacing,
)
from monai.inferers import sliding_window_inference


def create_model(model_path):
    model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2, 2),
            num_res_units=0,
            norm=Norm.BATCH,
            act="ReLU",
            bias=False,
    )
    loaded_data = torch.load(model_path)

    # This section needed because we save model within a _model property
    # in the Net class, but we don't want to create the Net class here.
    # This removes _model. from the ordered dictionary keys.
    old_keys = list(loaded_data["state_dict"].keys())
    for key in old_keys:
        updated_key = re.sub('_model.', '', key)
        loaded_data["state_dict"][updated_key] = loaded_data["state_dict"][key]
        del loaded_data["state_dict"][key]

    model.load_state_dict(loaded_data["state_dict"])
    return model


def inference(model, image_path):

    test_transform = Compose(
        [
            LoadImage(),
            EnsureChannelFirst(),
            Orientation(axcodes="RAS"),
            Spacing(pixdim=(2.0, 2.0, 2.0), mode="bilinear"),
            ScaleIntensity(minv=0.0, maxv=1.0),
            CropForeground(),
        ]
    )
    input = test_transform(image_path)

    model.eval()

    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    with torch.no_grad():
        roi_size = (96, 96, 32)
        sw_batch_size = 4
        # add a batch dim, hence unsqueeze at 0.
        output = sliding_window_inference(input.unsqueeze(0), roi_size, sw_batch_size, model)
        test_output = post_trans(output)

    return test_output


def save_model(model_path, output_path):
    model = create_model(model_path)
    example_forward_input = torch.rand(1, 1, 96, 96, 96)

    # Trace a module
    module = torch.jit.trace(model, example_forward_input)

    # Save to file
    torch.jit.save(module, os.path.join(output_path, 'msd_heart_model.pt'))


def test_model(model_path, image_path, output_path):
    
    model = create_model(model_path)
    seg = inference(model, image_path)

    writer = NibabelWriter()
    writer.set_data_array(seg.squeeze(0), channel_dim=0) # remove batch dim.
    # writer.set_metadata({"affine": torch.eye(4), "original_affine": torch.eye(4)})
    writer.write(os.path.join(output_path, "data", "output_la_030.nii.gz"), verbose=True)


if __name__ == "__main__":
    repo_root = "/Users/amithkamath/repo/monai-from-matlab/"

    # Save model to load with MATLAB.
    model_path = os.path.join(repo_root, "logs-202406-1610-5919/model-epoch=99-val_loss=0.0607-val_dice=0.8865.ckpt")
    save_model(model_path, repo_root)

    # Run inference on test data and save results.
    test_model(model_path, os.path.join(repo_root, "data/la_030.nii.gz"), repo_root)
