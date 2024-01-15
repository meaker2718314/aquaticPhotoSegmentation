import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from albumentations.pytorch import ToTensorV2
import albumentations as A


def filter_by_aspect(image_dir, mask_dir, sub_dirs):

    result = []

    for directory in sub_dirs:
        img_path = os.path.join(image_dir, directory)

        mask_path = os.path.join(mask_dir, directory)
        mask_path = mask_path.replace('.jpg', '.bmp')

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("RGB"), dtype=np.float32)

        if image.shape == mask.shape:
            result.append(directory)

    return result


class AquaticDataset(Dataset):
    def __init__(self, image_dir, mask_dir, color_dict, transform=None, img_dup=1):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.color_dict = color_dict

        self.images = []

        # Omit image/mask pairs with mismatched sizes
        filtered_img_dir = filter_by_aspect(self.image_dir, self.mask_dir, os.listdir(self.image_dir))

        # Dataset size will after be scaled using image augmentations, keep track of original size
        self.original_len = len(filtered_img_dir)

        for i in range(img_dup):
            self.images.extend(filtered_img_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])

        mask_path = os.path.join(self.mask_dir, self.images[index % self.original_len])
        mask_path = mask_path.replace('.jpg', '.bmp')

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("RGB"), dtype=np.int16)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        from utils import rgb_to_label_encoder
        mask = rgb_to_label_encoder(mask, self.color_dict)

        # Cast NumPy arrays to PyTorch tensors
        tensor_augment = A.Compose([ToTensorV2()], is_check_shapes=False)(image=image, mask=mask)
        image = tensor_augment["image"]
        mask = tensor_augment["mask"]

        return image, mask
