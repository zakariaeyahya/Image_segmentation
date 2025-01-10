# Projet de Segmentation d'Environnement avec Deep Learning

Ce projet vise à explorer et à comparer différentes architectures de Deep Learning pour la segmentation sémantique d'environnements urbains, en utilisant le dataset **Cityscapes**. Les modèles utilisés sont **DeepLabV3+**, **U-Net**, et **YOLO**. Ce projet a été réalisé dans le cadre du module de Deep Learning à l'École Nationale des Sciences Appliquées de Tétouan.

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

## Structure du Projet

Le projet est organisé comme suit :
```
├── images/            # Images utilisées pour l'entraînement et la validation
│   ├── U-net/         # Images pour U-Net
│   ├── YOLO/          # Images pour YOLO
│   └── deeplab/       # Images pour DeepLabV3+
├── src/               # Code source du projet
│   ├── Deeplab/       # Code spécifique à DeepLabV3+
│   │   ├── model/     # Fichiers de modèle pour DeepLabV3+
│   │   │   ├── init.py
│   │   │   ├── deeplabv3plus.py
│   │   │   ├── metrics.py
│   │   │   ├── prediction.py
│   │   │   └── train.py
│   │   └── processing/
│   │       └── data_preprocessing.py
│   ├── YOLO/          # Code spécifique à YOLO
│   │   ├── yolo-cityscapesv2.ipynb
│   ├── U-net
├── README.md          # Fichier README (ce fichier)
```


## Résultats

Les résultats des modèles sont comparés en termes de précision, de mIoU (mean Intersection over Union), et de temps d'inférence. Voici un résumé des performances :

| Modèle | mIoU (Entraînement) | mIoU (Validation) | Temps d'Inférence (s) |
|--------|--------------------:|------------------:|---------------------:|
| DeepLabV3+ | 0.57 | 0.57 | 0.04 |
| U-Net | 0.85 | 0.33 | 0.07 |
| YOLO | 0.18 | 0.12 | 0.02 |

## Perspectives d'Amélioration

- **DeepLabV3+** : Optimisation des hyperparamètres et réduction de la complexité.
- **U-Net** : Ajout de régularisation et augmentation des données pour réduire le surapprentissage.
- **YOLO** : Adaptation à la segmentation et exploration de versions plus récentes.

## Contributions

Les contributions sont les bienvenues ! Si vous souhaitez améliorer ce projet, veuillez ouvrir une issue ou soumettre une pull request.

## Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.
* **Zakariae Yahya** - *Data Scientist* - [Profil GitHub](https://github.com/zakariaeyahya)
* **Bouzhar Anouar** -*AI engineer * [Profil GitHub](https://www.linkedin.com/in/anouar-bouzhar-992519287/)
* **kholti hamza** - *Data Scientist* [Profil GitHub](https://www.linkedin.com/in/hamza-kholti-075288209/)

**Encadrant** : Mr. BELCAID Anas  
**Année Universitaire** : 2024-2025
