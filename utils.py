import torch
import numpy as np
import logging

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from PIL import Image

logging.getLogger().setLevel(logging.INFO)


def get_loaders(
        train_dir, train_maskdir,
        val_dir, val_maskdir,
        test_dir, test_maskdir,
        batch_size, train_transform,
        val_transform, mask_dict,
        num_workers=4, pin_memory=True,
):
    from datasets import AquaticDataset
    from torch.utils.data import DataLoader

    train_ds = AquaticDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
        img_dup=4,
        color_dict=mask_dict
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = AquaticDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
        color_dict=mask_dict,
        img_dup=1,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,

    )

    test_ds = AquaticDataset(
        image_dir=test_dir,
        mask_dir=test_maskdir,
        transform=val_transform,
        color_dict=mask_dict
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False
    )

    return train_loader, val_loader, test_loader


def get_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            preds = model(x)
            preds = torch.argmax(preds, dim=1)

            matches = torch.eq(preds, y)
            num_pixels += torch.numel(matches)
            num_correct += matches.sum().item()

    model.train()

    return round(num_correct / num_pixels, 4)


def get_confusion_matrix(loader, model, mask_dict, device="cuda"):
    model.eval()
    from sklearn.metrics import confusion_matrix

    n_classes = len(mask_dict)
    results = np.zeros(shape=(n_classes, n_classes), dtype=np.uint64)
    for x, y in loader:
        x, y = x.to(device), y.to(device).cpu().numpy()
        preds = model(x)
        preds = torch.argmax(preds, dim=1).cpu().numpy()
        confusion = confusion_matrix(y.flatten(), preds.flatten(), labels=range(n_classes))
        results = results + confusion

    for row in results:
        row /= sum(row)

    model.train()
    return results


def pixel_counts(loader, color_dict, skip=4):
    result = torch.zeros(size=(len(color_dict),), dtype=torch.long)
    for x, y in loader:
        y = y.flatten(start_dim=0, end_dim=0)

        for n in range(0, y.shape[0], skip):
            row = y[n, :]
            for m in range(0, row.shape[0], skip):
                result[row[m]] += 1

    return result


def save_predictions_as_imgs(
        loader, model, folder, color_dict, device="cuda", sample_size=None
):
    model.eval()
    if not os.path.isdir(folder):
        os.mkdir(folder)
    if sample_size is None:
        sample_size = len(loader)

    for idx, (x, y) in enumerate(loader):
        if idx == sample_size:
            break

        batch_dir = f'Evaluation_Batch_{idx}/'

        if not os.path.isdir(f'{folder}{batch_dir}'):
            os.mkdir(f'{folder}{batch_dir}')

        x = x.to(device=device)

        with torch.no_grad():

            preds = model(x)
            preds = torch.argmax(preds, dim=1)

            for j in range(preds.shape[0]):
                item_dir = f'Item_{j}/'

                if not os.path.isdir(f'{folder}{batch_dir}{item_dir}'):
                    os.mkdir(f'{folder}{batch_dir}{item_dir}')

                pred_rgb = preds[j, :, :].cpu().numpy()
                target_rgb = y[j, :, :].cpu().numpy()
                pred_rgb = label_to_rgb_encoder(pred_rgb, color_dict)
                target_rgb = label_to_rgb_encoder(target_rgb, color_dict)

                pred_img = Image.fromarray(pred_rgb, 'RGB')
                target_img = Image.fromarray(target_rgb, 'RGB')

                real_img = torch.permute(x[j, :, :, :], (1, 2, 0)).cpu().numpy()
                real_img *= 255
                real_img = real_img.astype(np.int8)
                real_img = Image.fromarray(real_img, 'RGB')

                pred_img.save(f'{folder}{batch_dir}{item_dir}prediction.jpg')
                target_img.save(f'{folder}{batch_dir}{item_dir}target.jpg')
                real_img.save(f'{folder}{batch_dir}{item_dir}real_rgb.jpg')

    model.train()


def save_evaluation_metrics(loader, model, folder, mask_dict, device="cuda"):
    if not os.path.isdir(folder):
        os.mkdir(folder)

    confusion_mtx = get_confusion_matrix(loader, model, mask_dict)
    net_accuracy = get_accuracy(loader, model)

    import seaborn as sns
    import matplotlib.pyplot as plt

    class_labels = ['Open Water', 'Human Diver', 'Seagrass', 'Shipwreck',
                    'Instruments', 'Reefs', 'Vertebrates', 'Sand+Rocks']

    plt.figure(figsize=(12, 12))
    plt.tight_layout()
    sns.set(font_scale=0.8)
    heatmap = sns.heatmap(confusion_mtx, xticklabels=class_labels, yticklabels=class_labels,
                          fmt='.2g', linewidth=2, linecolor='grey', annot=True, cmap='PiYG')
    plt.xlabel('Predicted Class', labelpad=100)
    plt.ylabel('True Class', labelpad=100)

    font = {'family': 'normal',
            'weight': 'bold',
            'size': 22}
    plt.rc('font', **font)

    plt.title('Validation Confusion Matrix')

    heatmap.get_figure().savefig(f'{folder}Confusion Matrix', dpi=120)
    print(net_accuracy)


def rgb_to_label_encoder(rgb_arr, color_dict):
    shape = rgb_arr.shape[:2]
    onehot_arr = np.zeros(shape=shape, dtype=np.int64)

    for i, (k, clr) in enumerate(color_dict.items()):
        onehot_arr += k * np.all(rgb_arr.reshape((-1, 3)) == clr, axis=1).reshape(shape)

    return onehot_arr


def label_to_rgb_encoder(label_arr, color_dict):
    shape = label_arr.shape + (3,)  # Add extra dimension for 3 color channels
    rgb_arr = np.zeros(shape=shape, dtype=np.int8)
    for k, clr in color_dict.items():
        rgb_arr[label_arr == k, :] = clr
    return rgb_arr
