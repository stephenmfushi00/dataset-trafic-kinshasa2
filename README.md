# Dataset de trafic routier adapté à Kinshasa

Ce dépôt présente les scripts de construction et d'évaluation du dataset du mémoire de Stephen Mfushi. La contribution recherchée est un dataset documenté, contrôlé et adapté aux scènes collectées à Kinshasa. YOLO est un outil d'annotation assistée et un détecteur de référence.

## Procédure réalisée

1. Première pré-annotation par YOLO préentraîné, suivie de corrections dans CVAT.
2. Contrôle des correspondances images/labels, classes, valeurs et doublons.
3. Premier fine-tuning de YOLOv8n sur 508 images retenues : 346 train, 84 validation, 78 test.
4. Pré-annotation de nouvelles images avec le premier best.pt et correction humaine.
5. Deuxième fine-tuning depuis best.pt, avec les anciennes images d'entraînement et les nouvelles admissibles. Le nombre exact ajouté dépend des exclusions consignées dans l'audit ; 500 labels déposés ne signifie pas 500 images ajoutées au train.
6. Comparaison sur les mêmes ensembles de validation et de test.

Il s'agit d'annotation semi-automatique avec réentraînement supervisé itératif. Les expériences ne mettent pas en œuvre un algorithme semi-supervisé utilisant des pseudo-étiquettes non corrigées.

## Scripts

- scripts/00_preannotation_initiale.py : première assistance à l'annotation.
- scripts/01_fine_tuning.py : lecture Drive, audit, découpage par séquence, entraînement, sauvegardes et évaluation.
- scripts/02_preannotation_cvat.py : prédictions du modèle localement adapté, ZIP d'images et ZIP d'annotations YOLO 1.1 pour CVAT.
- scripts/03_reentrainement.py : conservation du manifeste initial, ajout des nouvelles données autorisées, deuxième fine-tuning et comparaison.

Les scripts sont des exports commentés des notebooks utilisés, et ciblent Google Colab. Ils ne constituent pas encore une application locale autonome. Ils effectuent des téléchargements et créent des dossiers de résultats Drive lorsqu'on les exécute.

## Exécution dans Colab

Télécharger ou cloner le dépôt dans Colab. Activer un GPU, installer les dépendances, puis copier config_drive.example.json vers config_drive.json et remplir les IDs correspondant à vos fichiers. Ne pas publier ce fichier personnel.

```python
%pip install -r /content/dataset-trafic-kinshasa/requirements.txt
%run /content/dataset-trafic-kinshasa/scripts/01_fine_tuning.py
```

Adapter le chemin au dossier réel. Se connecter au compte Google qui contient les données. Le script 03 exige le best.pt et split.csv du premier entraînement ainsi que les sources correspondantes. Le rôle des IDs est visible à leur point d'utilisation dans les scripts. En cas de contrôle bloquant, résoudre la cause au lieu de retirer la vérification.

## Classes

0 car ; 1 motorcycle ; 2 tricycle ; 3 minibus ; 4 bus ; 5 truck ; 6 handcart.

## Résultats de test observés

| Modèle | Précision | Rappel | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|
| Premier fine-tuning | 82,75 % | 46,53 % | 52,33 % | 27,94 % |
| Deuxième fine-tuning | 76,04 % | 46,92 % | 48,21 % | 26,33 % |

Sur la validation, la mAP50–95 passe de 35,67 % à 38,00 %. Cette progression ne se retrouve pas sur le test. Le premier modèle reste la référence actuelle. Les résultats proviennent des sorties d'exécution communiquées et des rapports contrôlés précédemment, pas d'un entraînement exécuté lors de la préparation de ce dépôt.

## Limites et validation du dataset

Les pré-annotations doivent être corrigées, y compris les objets oubliés. Le découpage par préfixe avant _frame doit être vérifié contre les vidéos réelles. Les classes rares demandent une analyse séparée ; le test comporte seulement deux bus. La qualité technique d'un label ne garantit pas sa justesse visuelle. Une mesure du temps d'annotation et un contrôle indépendant d'un échantillon restent à réaliser avant de revendiquer un gain de temps chiffré ou une qualité globale validée.

Le second modèle n'est pas automatiquement meilleur. Réutiliser constamment le test pour choisir le modèle limite son indépendance : prévoir un nouveau test final si ces résultats guident les développements suivants.

## Références méthodologiques

- Adhikari et Huttunen, Iterative Bounding Box Annotation for Object Detection, ICPR, 2021. https://doi.org/10.1109/ICPR48806.2021.9412956
- Varma et al., IDD: A Dataset for Exploring Problems of Autonomous Navigation in Unconstrained Environments, WACV, 2019. https://arxiv.org/abs/1811.10200
- Documentation YOLO : https://docs.ultralytics.com/modes/train/
- Format CVAT YOLO : https://docs.cvat.ai/docs/dataset_management/formats/format-yolo/

## Vérification de cette conversion

Syntaxe Python vérifiée. Aucun entraînement ni accès Drive complet n'a été réexécuté pour cette conversion. Les dépendances, autorisations et données doivent être disponibles dans Colab. Les images et les poids ne sont pas inclus dans le dépôt.
