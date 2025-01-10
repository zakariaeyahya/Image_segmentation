# Projet de Segmentation d'Environnement avec Deep Learning

Ce projet vise à explorer et à comparer différentes architectures de Deep Learning pour la segmentation sémantique d'environnements urbains, en utilisant le dataset **Cityscapes**. Les modèles utilisés sont **DeepLabV3+**, **U-Net**, et **YOLO**. Ce projet a été réalisé dans le cadre du module de Deep Learning à l'École Nationale des Sciences Appliquées de Tétouan.

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
│   │   └── README.md
├── README.md          # Fichier README (ce fichier)
└── requirements.txt   # Liste des dépendances Python
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

## Auteurs
- Hamza Kholti
- Anouar Bouzhar

**Encadrant** : Mr. BELCAID Anas  
**Année Universitaire** : 2024-2025
