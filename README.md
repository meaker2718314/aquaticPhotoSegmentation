## Project Summary
The goal of this project is to perform multi-class semantic segmentation on various underwater images. 
The model is built using the PyTorch framework and relies a UNET architecture, an especially common choice
for research in biomedical segmentation tasks. 
The UNET relies on an expansive and contracting convolutional layer
sequence alongside several **skip-connections** which helps the model remember earlier stages and improve performance.
![image](https://github.com/meaker2718314/aquaticPhotoSegmentation/assets/70233435/80a31f0d-8bc1-499d-8473-6677dd145006)

## Model Details
The structure for this particular model uses an EfficientNet-B6 encoder with pre-initialized weights from ImageNet.
This model uses the **Cross-Entropy-Loss** function weights adjusted based on the relative net frequency of classes in the training dataset.

## Final Metrics and Analysis

Validation set metrics can be located in **/Results/Metrics**
The final accuracy of the UNET Model is **81%**, meaning the proportion of correctly classified pixels. Analysis of the Confusion Matrix suggests that the most predominant misclassifications were as follows:

| True Class  | Predicted Class      |  Proportion |
| ----------  | -------------------- | ----------- |
| Seagrass    | Reefs/Invertebrates  |  55%        |
| Sand/Rocks  | Reefs/Invertebrates  |  17%        |
| Instruments | Human Diver          |  16%        |

## Predictions

5 Batches of test predictions can be accessed in **Results/Saved Class Predictions**, alongside the original RGB image and the manually annotated image. Below is an exemplar prediction computed by the model adjacent to the original RGB image.


![image](https://github.com/meaker2718314/aquaticPhotoSegmentation/assets/70233435/9314dc90-9ce7-4980-b52d-9d6b97947058)
![image](https://github.com/meaker2718314/aquaticPhotoSegmentation/assets/70233435/45dc08f8-c596-45b2-9fa9-671ee8c23def)


