# TP — Modèles génératifs avec PyTorch

## Objectif
Ce TP permet de comprendre la différence entre un modèle discriminatif, un autoencodeur simple et un autoencodeur variationnel (VAE), à partir du dataset MNIST.

## 0. Installation et lancement

Le projet utilise Python 3.13 et `uv` pour gérer ses dépendances. Depuis la racine du projet, exécutez les commandes suivantes.

```bash
uv sync --locked
uv run --locked --with jupyterlab jupyter lab
```

| Fichier ou dossier | Rôle |
|---|---|
| [TP_Modeles_Generatifs.ipynb](TP_Modeles_Generatifs.ipynb) | Entraînement, évaluation et visualisation des trois modèles |
| [pyproject.toml](pyproject.toml) et [uv.lock](uv.lock) | Dépendances et versions verrouillées |
| `data/` | Stockage local du jeu de données MNIST |
| [captures/](captures/) | Figures produites par le notebook |

## 1. Chargement du dataset
MNIST contient 60 000 images d’apprentissage et 10 000 images de test, de taille 28 × 28. Les pixels sont convertis en tenseurs entre 0 et 1. DataLoader permet de traiter les images par lots de 128.

![Exemples MNIST](captures/01_mnist.png)

## 2. Modèle discriminatif
Le modèle reçoit une image et prédit un chiffre entre 0 et 9. Il utilise une couche cachée de 128 neurones avec ReLU et une couche de sortie de 10 neurones. L’entraînement utilise Adam et CrossEntropyLoss pendant 3 époques. L’évaluation calcule l’accuracy manuellement et avec Torcheval.

Dans l’exécution du notebook fourni, la perte passe de 0,4156 à 0,1428 et l’accuracy de test est de **96,02 %**.

## 3. Autoencodeur simple
L’encodeur réduit les 784 pixels à deux valeurs. Le décodeur utilise ces valeurs pour reconstruire l’image. Le modèle est entraîné sans labels pendant 5 époques avec Adam et la MSE. La perte d’entraînement du notebook fourni diminue de 0,0671 à 0,0461. Une cellule supplémentaire calcule la MSE sur le jeu de test.

### Reconstruction
La première ligne contient les images originales et la deuxième les reconstructions. La compression en deux dimensions peut faire perdre certains détails.

![Reconstruction AE](captures/02_ae_reconstruction.png)

### Exploration de l’espace latent
Chaque image est représentée par un point. La couleur indique son chiffre réel. On peut aussi calculer le vecteur moyen des images du chiffre 1 puis le décoder.

![Espace latent AE](captures/03_ae_latent.png)

![Chiffre représentatif](captures/04_ae_chiffre.png)

## 4. VAE
Le VAE associe à chaque image une distribution latente gaussienne décrite par une moyenne et une log-variance. La reparamétrisation permet d’échantillonner un vecteur latent tout en propageant les gradients. La perte combine l’entropie croisée binaire (BCE) pour la reconstruction et la divergence KL, qui rapproche les distributions latentes de la loi normale standard. L’entraînement utilise Adam pendant 20 époques.

Dans le notebook fourni, la perte totale moyenne passe de 144,90 à 140,79 par image. Une cellule supplémentaire calcule les trois termes de perte sur le jeu de test.

### Reconstruction et espace latent
Les images sont reconstruites après passage par l’encodeur et le décodeur. Les moyennes latentes sont représentées dans un plan 2D.

![Reconstruction VAE](captures/05_vae_reconstruction.png)

![Espace latent VAE](captures/06_vae_latent.png)

### Génération de nouvelles images
Le décodeur reçoit des vecteurs aléatoires tirés d’une loi normale. Il peut donc produire des images sans recevoir une image originale. Une grille de points permet d’observer les transitions dans l’espace latent.

![Génération VAE](captures/07_vae_generation.png)

![Grille latente](captures/08_vae_grille.png)

## 5. Comparaison et conclusion

| Modèle | Objectif | Labels nécessaires à l’entraînement | Perte |
|---|---|---|---|
| Classifieur | Reconnaître un chiffre | Oui | CrossEntropy |
| Autoencodeur | Reconstruire une image | Non | MSE |
| VAE | Reconstruire et générer des images | Non | BCE + KL |

Le classifieur sert à reconnaître les chiffres. L’autoencodeur apprend à compresser et reconstruire les images. Le VAE facilite en plus la génération de nouvelles images grâce à son espace latent régularisé.

## Note sur les résultats
Les valeurs et illustrations présentées proviennent des sorties enregistrées dans le notebook de cours du Pr. Mohamed YOUSSFI. Exécutez les cellules d’évaluation pour obtenir les métriques de test de l’autoencodeur et du VAE. Les cellules de visualisation régénèrent les figures dans `captures/`.
