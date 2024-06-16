
import pytorch_lightning

from monai.utils import set_determinism
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    ScaleIntensityd,
    Spacingd,
    EnsureType,
)
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, list_data_collate, decollate_batch, DataLoader

import torch

import os
import glob


PATCH_SIZE = 96
DEVICE = "cpu" #"cuda"


class Net(pytorch_lightning.LightningModule):
    def __init__(self):
        super().__init__()
        self._model = UNet(
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
        self.loss_function = DiceLoss(to_onehot_y=True, softmax=True)
        self.post_pred = Compose([EnsureType("tensor", device=DEVICE), AsDiscrete(argmax=True, to_onehot=2)])
        self.post_label = Compose([EnsureType("tensor", device=DEVICE), AsDiscrete(to_onehot=2)])
        self.dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        self.test_step_outputs = []

    def forward(self, x):
        return self._model(x)

    def test_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        roi_size = (PATCH_SIZE, PATCH_SIZE, 32)
        sw_batch_size = 4
        outputs = sliding_window_inference(images, roi_size, sw_batch_size, self.forward)
        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]
        dice = self.dice_metric(y_pred=outputs, y=labels)
        d = {"test_dice": dice, "test_number": len(outputs)}
        self.test_step_outputs.append(d)
        return d


def inference(saved_path, data_root):

    model = Net.load_from_checkpoint(saved_path)

    # set up the correct data path
    images = sorted(glob.glob(os.path.join(data_root, "imagesTr", "*.nii.gz")))
    labels = sorted(glob.glob(os.path.join(data_root, "labelsTr", "*.nii.gz")))
    data_dicts = [
        {"image": image_name, "label": label_name} for image_name, label_name in zip(images, labels)
    ]
    # For Spleen:
    test_files = data_dicts[15:]

    # set deterministic training for reproducibility
    set_determinism(seed=0)

    test_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(2.0, 2.0, 2.0),
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityd(
                keys=["image"],
                minv=0.0,
                maxv=1.0,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
        ]
    )

    test_ds = CacheDataset(
        data=test_files,
        transform=test_transforms,
        cache_rate=1.0,
        num_workers=4,
    )

    test_loader = DataLoader(test_ds, batch_size=1, num_workers=4)

    trainer = pytorch_lightning.Trainer(
        accelerator=DEVICE,
    )

    trainer.test(model=model, dataloaders=test_loader)
    print(model.test_step_outputs)


if __name__ == "__main__":

    repo_root = "/Users/amithkamath/repo/monai-from-matlab/"
    data_root = "/Users/amithkamath/data/MSD/Task02_Heart"
    saved_path = os.path.join(repo_root, "logs-202406-1610-5919/model-epoch=99-val_loss=0.0607-val_dice=0.8865.ckpt")

    inference(saved_path, data_root)
    # [{'test_dice': metatensor([[0.9011]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.9001]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.9100]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.9072]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.9093]]), 'test_number': 1}
