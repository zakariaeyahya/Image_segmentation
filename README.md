
---

# Urban Environment Segmentation Project with Deep Learning

This project aims to explore and compare different Deep Learning architectures for semantic segmentation of urban environments using the **Cityscapes** dataset. The models used are **DeepLabV3+**, **U-Net**, and **YOLO**. This project was carried out as part of the Deep Learning module at the National School of Applied Sciences in Tetouan.

## 🛠 Technologies

![Python](https://img.shields.io/badge/Python-3.12+-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-green)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Operations-blue)
![Tqdm](https://img.shields.io/badge/Tqdm-Progress%20Bar-orange)
![YAML](https://img.shields.io/badge/YAML-Data%20Serialization-yellow)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-red)
![YOLO](https://img.shields.io/badge/YOLO-Object%20Detection-purple)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Machine%20Learning-orange)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange)
![Torchvision](https://img.shields.io/badge/Torchvision-Computer%20Vision-green)
![PIL](https://img.shields.io/badge/PIL-Image%20Manipulation-blue)
![JSON](https://img.shields.io/badge/JSON-Data%20Manipulation-yellow)

## Project Structure

The project is organized as follows:

```
├── images/            # Images used for training and validation
│   ├── U-net/         # Images for U-Net
│   ├── YOLO/          # Images for YOLO
│   └── deeplab/       # Images for DeepLabV3+
├── src/               # Project source code
│   ├── Deeplab/       # DeepLabV3+ specific code
│   │   ├── model/     # Model files for DeepLabV3+
│   │   │   ├── __init__.py
│   │   │   ├── deeplabv3plus.py
│   │   │   ├── metrics.py
│   │   │   ├── prediction.py
│   │   │   └── train.py
│   │   └── processing/
│   │       └── data_preprocessing.py
│   ├── YOLO/          # YOLO specific code
│   │   ├── yolo-cityscapesv2.ipynb
│   ├── U-net/
├── README.md          # README file (this file)
```

## Results

The models' results are compared in terms of accuracy, mIoU (mean Intersection over Union), and inference time. Here is a summary of the performance:

| Model       | mIoU (Training) | mIoU (Validation) |
|-------------|----------------:|------------------:|
| DeepLabV3+  | 0.57            | 0.57              |
| U-Net       | 0.43            | 0.26              |
| YOLO        | ---             | 0.21              |

## Improvement Perspectives

- **DeepLabV3+**: Hyperparameter optimization and complexity reduction.
- **U-Net**: Adding regularization and data augmentation to reduce overfitting.
- **YOLO**: Adapting for segmentation and exploring newer versions.

## Contributions

Contributions are welcome! If you wish to improve this project, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

**Team Members:**
- **Zakariae Yahya** - *Data Scientist* - [GitHub Profile](https://github.com/zakariaeyahya)
- **Bouzhar Anouar** - *AI Engineer* - [LinkedIn Profile](https://www.linkedin.com/in/anouar-bouzhar-992519287/)
- **Kholti Hamza** - *Data Scientist* - [LinkedIn Profile](https://www.linkedin.com/in/hamza-kholti-075288209/)

**Supervisor:** Mr. BELCAID Anas
**Academic Year:** 2024-2025
