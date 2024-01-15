import torch
import torch.nn as nn
import torch.optim.lr_scheduler as scheduler
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import albumentations as A
from tqdm import tqdm
import torch.optim as optim
import segmentation_models_pytorch as smp
from utils import (
    get_loaders,
    get_accuracy,
    save_predictions_as_imgs,
    save_evaluation_metrics,
    pixel_counts
)
import warnings
warnings.filterwarnings("ignore")
torch.manual_seed(-1)

"""
Dataset name: SUIM
Nature: Dataset for semantic segmentation of underwater imagery
Source: https://irvlab.cs.umn.edu/resources/usr-248-dataset
Paper: https://arxiv.org/pdf/2004.01241.pdf


RGB color code and object categories:
-------------------------------------
000 BW: Background water body
001 HD: Human divers
010 PF: Plants/sea-grass
011 WR: Wrecks/ruins
100 RO: Robots/instruments
101 RI: Reefs and invertebrates
110 FV: Fish and vertebrates
111 SR: Sand/sea-floor (& rocks)


Statistics
-----------------------------------------------------------------------------
train_val/ contains 1525 images for training/validation
test/ contains 110 images for benchmark evaluation (do not alter)


Additionals
-----------------------------------------------------------------------------
Checkpoint_Data/ contains pretrained checkpoints for SOTA models (see paper)
Benchmark_Evaluation/ contains the corresponding test-set results
"""

# Hyperparameters
LEARNING_RATE = 1.00e-4
DEVICE = "cuda"
NUM_EPOCHS = 6
NUM_WORKERS = 5
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
BATCH_SIZE = 6

MASK_DICT = {
    0: [0, 0, 0],  # background water
    1: [0, 0, 255],  # human diver
    2: [0, 255, 0],  # plant / seagrass
    3: [0, 255, 255],  # wrecks / ruins
    4: [255, 0, 0],  # robots / instrumentation
    5: [255, 0, 255],  # coral reefs/invertebrates
    6: [255, 255, 0],  # fish / vertebrates
    7: [255, 255, 255],  # sand / rocks
}

# Training and Evaluation options
PIN_MEMORY = True
LOAD_MODEL = True
TRAIN_MODEL = False
FINE_TUNE = True

# Saving options
SAVE_MODEL = False
SAVE_PREDICTIONS = True
SAVE_METRICS = False

TRAIN_IMG_DIR = "Aquatic Data Splits/train/images/"
TRAIN_MASK_DIR = "Aquatic Data Splits/train/masks/"
VAL_IMG_DIR = "Aquatic Data Splits/val/images/"
VAL_MASK_DIR = "Aquatic Data Splits/val/masks/"
TEST_IMG_DIR = 'Aquatic Data Splits/test/images'
TEST_MASK_DIR = 'Aquatic Data Splits/test/masks'
SAVED_IMG_DIR = "Results/Saved Class Predictions/"
SAVED_STATE_DIR = 'Learned States/'
SAVED_METRIC_DIR = 'Results/Metrics/'
LOAD_DIR = f'{SAVED_STATE_DIR}best_trained_UNET_dict.pt'


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.to(device=DEVICE)

        # forward pass
        predictions = model(data)
        loss = loss_fn(predictions, targets)
        # backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    # Define transforms for image augmentation

    train_transform = A.Compose(
        [
            A.SmallestMaxSize(max_size=max(IMAGE_WIDTH, IMAGE_HEIGHT)),
            A.RandomCrop(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.HorizontalFlip(p=0.4),
            A.RandomRotate90(p=0.25),

            A.OneOf([
                A.GridDistortion(p=0.3, distort_limit=0.3),
                A.ElasticTransform(p=0.3, alpha=2.25, sigma=7.0),
                A.OpticalDistortion(p=0.3, distort_limit=.3, shift_limit=.2),
                A.Blur(blur_limit=3, p=0.3)
            ]),

            A.RGBShift(p=0.67, r_shift_limit=35, g_shift_limit=35, b_shift_limit=20),
            A.RandomBrightnessContrast(p=0.50, brightness_limit=.15, contrast_limit=.10),

            A.Normalize(
                mean=[0., 0., 0.],
                std=[1., 1., 1.],
                max_pixel_value=255.0,
            ),
        ],
    )

    val_transform = A.Compose(
        [
            A.SmallestMaxSize(max_size=max(IMAGE_WIDTH, IMAGE_HEIGHT)),
            A.RandomCrop(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),

            A.Normalize(
                mean=[0., 0., 0.],
                std=[1., 1., 1.],
                max_pixel_value=255.0,
            ),
        ],
    )

    # Use pretrained weights for UNET model

    ENCODER = 'efficientnet-b6'
    ENCODER_WEIGHTS = 'imagenet'
    ACTIVATION = None  # Logit should be used with nn.CrossEntropyLoss()

    UNET_model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        activation=ACTIVATION,
        classes=len(MASK_DICT),

    ).to(DEVICE)

    train_loader, val_loader, _ = get_loaders(
        TRAIN_IMG_DIR, TRAIN_MASK_DIR,
        VAL_IMG_DIR, VAL_MASK_DIR,
        TEST_IMG_DIR, TEST_MASK_DIR,
        BATCH_SIZE, train_transform,
        val_transform, MASK_DICT,
        NUM_WORKERS, PIN_MEMORY,
    )

    # Adjust loss function based on class frequencies...
    weights = pixel_counts(train_loader, MASK_DICT).cuda()
    weights = torch.max(weights).item() * torch.reciprocal(weights + 1e-3)

    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(UNET_model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()

    if LOAD_MODEL:
        UNET_model.load_state_dict(torch.load(LOAD_DIR))

    val_accs = []
    best_state = UNET_model.state_dict()

    lr_scheduler = scheduler.StepLR(
        optimizer,
        step_size=2,
        gamma=0.15,
    )

    for epoch in range(NUM_EPOCHS):
        if not TRAIN_MODEL:
            break

        train_fn(train_loader, UNET_model, optimizer, loss_fn, scaler)

        val_acc = get_accuracy(val_loader, UNET_model, device=DEVICE)
        if val_acc > max(val_accs, default=0.0):
            best_state = UNET_model.state_dict()
        val_accs.append(val_acc)

        if FINE_TUNE:
            lr_scheduler.step()

    if SAVE_MODEL:
        torch.save(best_state, f'{SAVED_STATE_DIR}best_trained_UNET_dict.pt')

    if SAVE_PREDICTIONS:
        save_predictions_as_imgs(val_loader, UNET_model,
                                 SAVED_IMG_DIR, MASK_DICT,
                                 sample_size=5)

    if SAVE_METRICS:
        save_evaluation_metrics(val_loader, UNET_model,
                                SAVED_METRIC_DIR, MASK_DICT)


if __name__ == "__main__":
    main()
