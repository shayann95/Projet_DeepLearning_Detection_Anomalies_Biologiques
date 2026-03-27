# Détection d'anomalies biologiques

## Présentation
Ce projet a été réalisé dans le cadre de mon apprentissage personnel du Machine Learning et du Deep Learning. L’objectif est de concevoir un modèle capable de classifier automatiquement des maladies de plantes à partir d’images de feuilles.

À travers ce projet, j’ai souhaité mettre en pratique mes compétences en traitement d’images, en modélisation et en utilisation de frameworks de Deep Learning, tout en travaillant sur un cas d’usage concret.

## Objectifs du projet
- Approfondir mes connaissances en Machine Learning et Deep Learning  
- Mettre en œuvre un pipeline complet de classification d’images  
- Manipuler un dataset réel et structuré  
- Comprendre les problématiques de surapprentissage et de généralisation  
- Développer un projet concret à valoriser dans le cadre de ma recherche d’alternance  

## Dataset
- Source : Kaggle (PlantVillage Dataset)  
- Environ 54 000 images  
- 38 classes (plantes saines et malades)  
- Images en couleur utilisées pour l’entraînement  

## Méthodologie

### Prétraitement des données
- Redimensionnement des images en 224x224  
- Normalisation des pixels  
- Séparation des données :
  - 80 % entraînement  
  - 20 % validation  

### Modélisation
Implémentation d’un réseau de neurones convolutif (CNN) avec TensorFlow/Keras :
- Conv2D (32 filtres) + ReLU  
- MaxPooling  
- Conv2D (64 filtres) + ReLU  
- MaxPooling  
- Flatten  
- Dense (256 neurones) + ReLU  
- Dense (38 classes) + Softmax  

Le modèle contient environ 47,8 millions de paramètres.

### Entraînement
- Optimiseur : Adam  
- Fonction de perte : categorical crossentropy  
- Nombre d’epochs : 5  
- Batch size : 32  

## Résultats
- Accuracy entraînement : ~98 %  
- Accuracy validation : ~84 %  

Le modèle montre une bonne capacité d’apprentissage avec une convergence rapide. Un léger surapprentissage apparaît après plusieurs epochs.

## Exemple de prédiction
```python
image_path = 'test_apple_black_rot.JPG'
prediction = predict_image_class(model, image_path, class_indices)
print(prediction)
```
Résultat attendu :
```
Apple___Black_rot
```

## Compétences développées
- Deep Learning (CNN, classification d’images)  
- TensorFlow / Keras  
- Prétraitement d’images  
- Manipulation de datasets (Kaggle API)  
- Analyse des performances d’un modèle  
- Compréhension du surapprentissage  

## Installation
1. Cloner le projet :
```bash
git clone https://github.com/votre-username/plant-disease-classification.git
cd plant-disease-classification
```
2. Installer les dépendances :
```bash
pip install -r requirements.txt
```
3. Configurer Kaggle :
```python
import os
os.environ['KAGGLE_USERNAME'] = "votre_username"
os.environ['KAGGLE_KEY'] = "votre_key"
```
4. Télécharger le dataset :
```bash
kaggle datasets download -d abdallahalidev/plantvillage-dataset
unzip plantvillage-dataset.zip
```

## Utilisation
Entraîner le modèle :
```bash
python train.py
```
Faire une prédiction :
```python
predict_image_class(model, "image.jpg", class_indices)
```

## Structure du projet
```
plant-disease-classification/
├── plantvillage dataset/
├── notebooks/
├── model/
├── class_indices.json
├── train.py
├── requirements.txt
└── README.md
```

## Limites
- Modèle volumineux (~180 MB)  
- Léger surapprentissage  
- Pas de data augmentation  
- Pas de transfer learning  

## Améliorations possibles
- Utilisation de modèles pré-entraînés (ResNet, EfficientNet)  
- Ajout de data augmentation  
- Optimisation du modèle  
- Déploiement en application web  
- Ajout d’explicabilité (Grad-CAM)  

## Démarche personnelle
Ce projet s’inscrit dans une démarche proactive d’apprentissage. Il m’a permis de consolider mes bases en Deep Learning et de développer des compétences concrètes en Computer Vision.

