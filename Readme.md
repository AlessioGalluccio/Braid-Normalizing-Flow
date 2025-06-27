# Braid-Normalizing-Flow
A novel anomaly detection architecture for multi-view image analysis, extending CS-Flow to leverage triplet training with sidelight images. Developed as part of graduate thesis work on generative computer vision for industrial defect detection.
![plot](braid-normalizing-flows-architecture.jpg)

Overview
Braid Normalizing Flow processes three synchronized images (one diffuse + two sidelight) to detect anomalies by modeling normal data distribution. The "braid" architecture ensures each image representation interacts with the others before final anomaly scoring, mimicking expert visual inspection workflows.
Architecture
The model splits input features into three blocks (diffuse, sidelight1, sidelight2) and processes them through a sequence of CS-Flow coupling layers:

CSF0: Processes diffuse + sidelight1 features
CSF1: Processes sidelight1 (from CSF0) + sidelight2 features
CSF2: Processes diffuse (from CSF0) + sidelight2 (from CSF1) features

This rotation ensures every block pair interacts exactly once, enabling comprehensive multi-view anomaly detection.
Key Features

Multi-view anomaly detection using synchronized camera inputs
Maintains feature ordering through deactivated permutation layers
Independent parameter learning for each coupling flow
Industrial applicability for quality control scenarios

Applications
Designed for manufacturing environments where multiple camera angles provide complementary defect information (e.g., tire inspection, surface quality assessment).
Citation
Built upon CS-Flow: "Fully Convolutional Cross-Scale-Flows for Image-based Defect Detection" (WACV 2022) by Rudolph et al.

## License

This project is licensed under the MIT License.
