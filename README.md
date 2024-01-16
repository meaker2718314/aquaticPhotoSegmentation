## Project Summary
The goal of this project is to perform multi-class semantic segmentation on various underwater images. 
The model is built using the PyTorch framework and relies a UNET architecture, an especially common choice
for research in biomedical segmentation tasks. 
The UNET relies on an expansive and contracting convolutional layer
sequence alongside several **skip-connections** which helps the model remember earlier stages and improve performance.
![image](https://github.com/meaker2718314/aquaticPhotoSegmentation/assets/70233435/80a31f0d-8bc1-499d-8473-6677dd145006)

## Model Details
The structure for this particular model uses an EfficientNet-B6 encoder with pre-initialized weights from ImageNet.


## Final Metrics and Analysis

The final performance of the model can be analyzed using the Confusion Matrix found in directory **/Results/Metrics**
