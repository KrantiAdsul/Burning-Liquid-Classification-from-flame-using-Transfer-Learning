# Burning-Liquid-Classification-from-flame-using-Transfer-Learning

## Overview
This project focuses on using transfer learning to classify burning liquids in static flame images. The goal is to fine-tune a pretrained ResNet-34 model and evaluate its performance comprehensively.

## Dataset
- **Data Source:** [Burning Liquid Dataset](https://doi.org/10.1007/s10973-021-10903-2 – Supplementary Information, File #2)
- **Description:** The dataset comprises 3000 high-resolution flame images of burning ethanol, pentane, and propanol.
- **Data Preparation:** Images should be extracted into a `data` folder and organized into subfolders based on their respective classes (ethanol, pentane, propanol).

## Model
- **Pretrained Model:** ResNet-34 from torchvision
- **Model Adaptation:** The final classification output layer is modified to accommodate the three burning liquid classes.

## Training
- **Fine-tuning:** The pretrained model is fine-tuned using the custom dataset.
- **Hyperparameter Exploration:** Experimented with learning rates, batch sizes, and training epochs.
- **Layer Freezing:** Implemented layer freezing techniques to optimize model adaptation.

## Layer Visualization
- **Internal Representation:** Visualized feature maps from different convolutional layers within the model.
- **Insight Gathering:** Gained insights into the model's internal representation using PyTorch hooks.
- **Visualization:** Produced image grids displaying output activations for selected layers.

## Analysis
- **Performance Evaluation:** Reported the accuracy of the fine-tuned model on the testing set and compare it with the baseline ResNet-34 model.
- **Confusion Matrix:** Generated a confusion matrix to analyze inter-class error rates.
- **Optional Metrics:** Utilized the sklearn.metrics module for comprehensive classification performance metrics.
- **Precision-Recall Curves:** Created precision-recall curves for each class to assess binary classification performance.

## References
- Martinka, J., Neˇcas, A., Rantuch, P. The recognition of selected burning liquids by convolutional neural networks under laboratory conditions. J Therm Anal Calorim 147, 5787-5799 (2022).
- [TorchVision ResNet Source Code](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)
