# PulmoScan - Système de Détection du Cancer du Poumon

PulmoScan est un système complet pour la détection du cancer du poumon utilisant le modèle EfficientNetB7. Ce README explique comment utiliser les différentes interfaces pour tester votre modèle avec des images de poumons.

## Table des matières

1. [Installation](#installation)
2. [Structure du Projet](#structure-du-projet)
3. [Utilisation](#utilisation)
   - [Interface Web](#interface-web)
   - [Interface Graphique (GUI)](#interface-graphique-gui)
   - [Interface en Ligne de Commande (CLI)](#interface-en-ligne-de-commande-cli)
   - [Traitement par Lots](#traitement-par-lots)
4. [Conversion du Modèle](#conversion-du-modèle)
5. [Dépannage](#dépannage)
6. [Authentification](#authentification)
7. [Dashboard](#dashboard)

## Installation

1. Assurez-vous que toutes les dépendances sont installées :

```bash
pip install -r requirements.txt
```

2. Vérifiez que votre modèle EfficientNetB7 est présent dans le répertoire `models/` (par défaut : `models/efficientnetb7_final_model.keras`).

3. Appliquez les migrations de la base de données :

```bash
python manage.py migrate
```

## Structure du Projet

Le projet est organisé de la manière suivante :

```
PulmoScan/
├── accounts/              # Application Django pour l'authentification
├── api/                   # API REST pour les prédictions
├── core/                  # Configuration Django pour l'interface web
├── dashboard/             # Application Django pour le dashboard
├── docs/                  # Documentation et notebooks
├── media/                 # Dossier pour stocker les images uploadées
├── models/                # Modèles de machine learning
│   ├── demo_model/        # Modèle de démonstration
│   └── efficientnetb7_final_model.keras  # Modèle principal
├── results/               # Résultats des analyses
├── scripts/               # Scripts utilitaires
├── static/                # Fichiers statiques pour l'interface web
├── templates/             # Templates HTML pour l'interface web
├── batch_predict.py       # Script pour le traitement par lots d'images
├── manage.py              # Script Django pour gérer l'interface web
├── predict_with_efficientnet.py  # Module pour faire des prédictions
├── pulmoscan.py           # Script principal avec menu
├── pulmoscan_cli.py       # Interface en ligne de commande
├── pulmoscan_gui.py       # Interface graphique
└── requirements.txt       # Dépendances du projet
```

## Utilisation

Vous pouvez lancer PulmoScan de plusieurs façons :

```bash
# Interface principale avec menu
python pulmoscan.py

# Lancer directement l'interface web
python pulmoscan.py --web

# Lancer directement l'interface graphique
python pulmoscan.py --gui

# Lancer directement l'interface en ligne de commande
python pulmoscan.py --cli

# Lancer directement le traitement par lots
python pulmoscan.py --batch

# Lancer directement la conversion du modèle
python pulmoscan.py --convert
```

### Interface Web

L'interface web vous permet d'uploader et d'analyser des images via un navigateur web. Elle offre également un dashboard pour visualiser les résultats.

Pour lancer l'interface web :

```bash
python manage.py runserver
```

Puis ouvrez votre navigateur à l'adresse : http://127.0.0.1:8000/

### Interface Graphique (GUI)

L'interface graphique offre une expérience utilisateur simple pour charger et analyser des images.

Pour lancer l'interface graphique :

```bash
python pulmoscan_gui.py
```

### Interface en Ligne de Commande (CLI)

L'interface en ligne de commande permet d'analyser rapidement une image.

```bash
python pulmoscan_cli.py --model models/efficientnetb7_final_model.keras --image chemin/vers/image.jpg --threshold 0.5
```

### Traitement par Lots

Le traitement par lots permet d'analyser plusieurs images en une seule fois.

```bash
python batch_predict.py --model models/efficientnetb7_final_model.keras --input dossier/images --output resultats.json --threshold 0.5
```

## Conversion du Modèle

Si vous rencontrez des problèmes de compatibilité avec votre modèle, vous pouvez essayer de le convertir en format H5 :

```bash
python scripts/convert_model_h5.py --input models/efficientnetb7_final_model.keras --output models/modele_converti.h5
```

## Dépannage

### Problèmes de chargement du modèle

Si vous rencontrez des erreurs lors du chargement du modèle EfficientNetB7, essayez les solutions suivantes :

1. **Convertir le modèle** : Utilisez le script `scripts/convert_model_h5.py` pour convertir votre modèle en format H5.
2. **Vérifier la version de TensorFlow** : Assurez-vous que vous utilisez une version compatible de TensorFlow.
3. **Utiliser l'interface en ligne de commande** : L'interface CLI peut parfois fonctionner même si l'interface web rencontre des problèmes.

### Problèmes d'upload d'images

Si vous rencontrez des problèmes lors de l'upload d'images :

1. **Vérifier les permissions** : Assurez-vous que les dossiers `media` et `media/uploads` existent et ont les bonnes permissions.
2. **Utiliser l'interface graphique** : L'interface GUI peut être plus fiable pour charger des images.
3. **Vérifier le format de l'image** : Assurez-vous que l'image est dans un format supporté (JPG, PNG, etc.).

## Authentification

PulmoScan inclut un système d'authentification complet :

1. **Inscription** : Créez un compte à l'adresse http://127.0.0.1:8000/accounts/register/
2. **Connexion** : Connectez-vous à l'adresse http://127.0.0.1:8000/accounts/login/
3. **Profil** : Gérez votre profil à l'adresse http://127.0.0.1:8000/accounts/profile/

Les analyses sont associées à votre compte utilisateur, ce qui vous permet de suivre vos résultats.

## Dashboard

Le dashboard offre une vue d'ensemble de vos analyses :

1. **Statistiques** : Nombre total d'analyses, répartition entre nodules malins et bénins
2. **Graphiques** : Visualisation des tendances au fil du temps
3. **Activité récente** : Dernières analyses effectuées
4. **Actions rapides** : Accès rapide aux fonctionnalités principales

Pour accéder au dashboard, connectez-vous et visitez http://127.0.0.1:8000/

---

Pour toute question ou problème, n'hésitez pas à contacter l'équipe PulmoScan.
