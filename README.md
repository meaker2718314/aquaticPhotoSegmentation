## Project Summary
The goal of this project is to perform multi-class semantic segmentation on various underwater images. 
The model is built using the PyTorch framework and relies a UNET architecture, an especially common choice
for research in biomedical segmentation tasks. 
The UNET relies on an expansive and contracting convolutional layer
sequence alongside several **skip-connections** which helps the model remember earlier stages and improve performance.
![image](https://github.com/meaker2718314/aquaticPhotoSegmentation/assets/70233435/80a31f0d-8bc1-499d-8473-6677dd145006)

The SUIM Dataset was utilized for this project: https://irvlab.cs.umn.edu/image-segmentation/suim

## Model Details
The structure for this particular model uses an EfficientNet-B6 encoder with pre-initialized weights from ImageNet.
This model utilizes the **Cross-Entropy-Loss** function weights adjusted based on the relative net frequency of classes in the training dataset.
The **Adam** optimizer was chosen for training. 

## Final Metrics and Analysis

Validation set metrics can be located in **/Results/Metrics**
The final accuracy of the UNET Model is **81%**, which describes the proportion of correctly classified pixels. Analysis of the Confusion Matrix suggests that the most predominant misclassifications were as follows:

| True Class  | Predicted Class      |  Proportion |
| ----------  | -------------------- | ----------- |
| Seagrass    | Reefs/Invertebrates  |  55%        |
| Sand/Rocks  | Reefs/Invertebrates  |  17%        |
| Instruments | Human Diver          |  16%        |

## Predictions

5 Batches of test predictions can be accessed in **Results/Saved Class Predictions**, alongside the original RGB image and the manually annotated image.
The colors corresponding to each class are as follows:

|     **Color**       |        Class Label           |
| ------------------- | ---------------------------- |
| Black (0,0,0)       | - Open Water                 |
| Blue (0,0,255)      | - Human Diver                |
| Green (0,255,0)     | - Seagrass                   |
| Cyan (0,255,255)    | - Shipwreck or Ruins         |
| Red (255,0,0)       | - Artificial Instrumentation |
| Magenta (255,0,255) | - Reefs/Invertebrates        |
| White (255,255,255) | - Sand/Rocks                 | 




Below is an exemplar prediction computed by the model adjacent to the original RGB image.


![image](https://github.com/meaker2718314/aquaticPhotoSegmentation/assets/70233435/9314dc90-9ce7-4980-b52d-9d6b97947058)
![image](https://github.com/meaker2718314/aquaticPhotoSegmentation/assets/70233435/45dc08f8-c596-45b2-9fa9-671ee8c23def)

## Steps for Model Improvement 

- A different loss function could be utilized which is closer tied to final accuracies, such as a **Multi-Class Dice** loss criterion or **IoU** loss criterion.
- Especially poor performing batches during training could be revisited during training
- A larger, more robut encoder could be chosen to increase model complexity, or alternatively, a smaller encoder to prevent overfitting and improve generalization.
- Dataset quality could be improved with more geometric augmentations on the initial dataset, such as **rotations, orthogonal projections**, etc.

