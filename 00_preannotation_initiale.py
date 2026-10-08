"""Script exporté de l'expérience Colab ; exécuter dans Colab avec %run.
Les calculs d'entraînement ne sont pas lancés lors de la préparation du dépôt.
"""
import json
from pathlib import Path
CONFIG_PATH = Path(__file__).resolve().parents[1] / 'config_drive.json'
if not CONFIG_PATH.exists():
    raise FileNotFoundError('Copier config_drive.example.json en config_drive.json et renseigner les IDs Drive.')
CONFIG = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))

# # Pré-annotation directe de a_annoter → CVAT
# **Sans entraînement et sans ZIP corrigé à téléverser.** Le modèle YOLOv8n préentraîné sur COCO propose des boîtes ; elles restent à vérifier dans CVAT.
# 
# 1. Importer ce notebook dans Colab.
# 2. Choisir un GPU : Exécution → Modifier le type d'exécution → T4.
# 3. Exécuter les cellules dans l'ordre et autoriser Drive.
# 4. Le dossier lu est `Dataset_Trafic_Routiere_Kinshasa/a_annoter`. Modifier ROOT si nécessaire.
# 5. Traitement de 300 images en un seul lot par défaut. Pour tout traiter, modifier `NOMBRE_IMAGES = None`.
# 
# Le modèle standard propose `car`, `motorcycle`, `bus`, `truck`. Les IDs COCO sont traduits vers vos IDs : car→0, motorcycle→1, bus→4, truck→5. Les IDs 2 tricycle, 3 minibus, 6 handcart sont réservés pour votre correction manuelle : le modèle standard ne possède pas ces classes. Il peut confondre un minibus avec un bus ou une voiture ; corriger ces erreurs.
# 
# Tous les résultats sont rangés dans `Dataset_Trafic_Routiere_Kinshasa/Trafic routier annotation YOLO 300/`, dans un sous-dossier unique à chaque lancement. Les anciennes sorties sont conservées.

# Étape 2 du notebook original.
from google.colab import drive
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import random, re, math, csv, json, zipfile
from uuid import uuid4
import torch, ultralytics
from ultralytics import YOLO

drive.mount('/content/drive')
DOSSIER_PRINCIPAL = Path('/content/drive/MyDrive/Dataset_Trafic_Routiere_Kinshasa')
DOSSIER_PHOTOS = DOSSIER_PRINCIPAL / 'a_annoter'
NOMBRE_IMAGES = 300  # None pour toutes les photos.
TAILLE_LOT = 300
SEUIL_CONFIANCE = 0.60  # Seuil de départ, à évaluer ; ne garantit pas l'exactitude.
TAILLE_IMAGE = 960
GRAINE = 42
MODELE_YOLO = 'yolov8n.pt'
CLASSES = ['car','motorcycle','tricycle','minibus','bus','truck','handcart']
# Mapping par NOM, jamais par reprise aveugle des IDs COCO.
IDS_VEHICULES = {'car':0, 'motorcycle':1, 'bus':4, 'truck':5}
# Tous les résultats restent dans ce dossier ; chaque lancement a sa propre place.
DOSSIER_RESULTATS = DOSSIER_PRINCIPAL / 'Trafic routier annotation YOLO 300'
NOM_EXECUTION = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid4().hex[:8]
DOSSIER_SORTIE = DOSSIER_RESULTATS / NOM_EXECUTION
if not DOSSIER_PHOTOS.is_dir(): raise FileNotFoundError(f'Dossier absent : {DOSSIER_PHOTOS}. Corriger DOSSIER_PHOTOS.')
DOSSIER_SORTIE.mkdir(parents=True, exist_ok=False)
DOSSIER_LABELS = DOSSIER_SORTIE / 'labels'; DOSSIER_LABELS.mkdir()
APPAREIL = 0 if torch.cuda.is_available() else 'cpu'
print('Calcul :', APPAREIL, '(CPU plus lent)' if APPAREIL == 'cpu' else torch.cuda.get_device_name(0))
modele = YOLO(MODELE_YOLO)
correspondance_ids = {i:IDS_VEHICULES[name] for i,name in modele.names.items() if name in IDS_VEHICULES}
if len(correspondance_ids) != len(IDS_VEHICULES): raise ValueError('Le modèle ne contient pas les classes attendues.')
print('Mapping COCO → dataset :', correspondance_ids)


# Étape 3 du notebook original.
extensions = {'.jpg','.jpeg','.png','.bmp','.webp'}
photos = sorted(p for p in DOSSIER_PHOTOS.rglob('*') if p.is_file() and p.suffix.lower() in extensions and not p.name.startswith('._'))
noms_images = defaultdict(list)
for p in photos: noms_images[p.stem.casefold()].append(p)
doublons = [v for v in noms_images.values() if len(v) > 1]
if doublons: raise ValueError(f'Noms ambigus : {doublons[:3]}. Arrêt pour éviter écrasement ou renommage.')
if not photos: raise ValueError('Aucune image trouvée.')
print('Images trouvées :',len(photos))
if NOMBRE_IMAGES is not None and len(photos) < NOMBRE_IMAGES:
    raise ValueError(f'{len(photos)} photos disponibles, mais {NOMBRE_IMAGES} demandées. Vérifier le dossier DOSSIER_PHOTOS avant de continuer.')
if NOMBRE_IMAGES is not None:
    sequences = defaultdict(list)
    for p in photos: sequences[re.split('_frame',p.stem,maxsplit=1,flags=re.I)[0]].append(p)
    rng = random.Random(GRAINE)
    for v in sequences.values(): rng.shuffle(v)
    selection = []
    while len(selection) < NOMBRE_IMAGES and any(sequences.values()):
        for key in sorted(sequences):
            if sequences[key] and len(selection) < NOMBRE_IMAGES: selection.append(sequences[key].pop())
    photos = sorted(selection)
print('Images à traiter :',len(photos))
(DOSSIER_SORTIE / 'classes.txt').write_text(''.join(c+'\n' for c in CLASSES))
bilan=[]
with (DOSSIER_SORTIE / 'confiances.csv').open('w',newline='') as f:
    redacteur=csv.redacteur(f); redacteur.writerow(['image','id_dataset','classe','confiance','xc','yc','w','h'])
    for index,im in enumerate(photos,1):
        r=modele.predict(source=str(im),classes=list(correspondance_ids),conf=SEUIL_CONFIANCE,imgsz=TAILLE_IMAGE,
                        device=APPAREIL,max_det=1000,verbose=False)[0]
        lines=[]
        if r.boxes is not None:
            for cls,confidence,coords in zip(r.boxes.cls.cpu().tolist(),r.boxes.conf.cpu().tolist(),r.boxes.xywhn.cpu().tolist()):
                if int(cls) not in correspondance_ids or confidence < SEUIL_CONFIANCE: continue
                cid=correspondance_ids[int(cls)]; xc,yc,w,h=coords
                if not all(math.isfinite(v) for v in coords) or not (0<=xc<=1 and 0<=yc<=1 and 0<w<=1 and 0<h<=1):
                    raise ValueError(f'Boîte invalide : {im.name}')
                lines.append(f'{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}')
                redacteur.writerow([im.name,cid,CLASSES[cid],confidence,*coords])
        (DOSSIER_LABELS/(im.stem+'.txt')).write_text('\n'.join(lines)+('\n' if lines else ''))
        bilan.append({'image':im.name,'detections':len(lines),'statut':'non_verifiee'})
        if index <= 12:
            preview=DOSSIER_SORTIE/'apercus';preview.mkdir(exist_ok=True)
            r.save(filename=str(preview/im.name))
        if index%25==0 or index==len(photos):print(index,'/',len(photos))
with (DOSSIER_SORTIE/'rapport_images.csv').open('w',newline='') as f:
    redacteur=csv.DictWriter(f,fieldnames=['image','detections','statut']);redacteur.writeheader();redacteur.writerows(bilan)
(DOSSIER_SORTIE/'parametres.json').write_text(json.dumps({'modele':MODELE_YOLO,'entrainement_effectue':False,
    'confiance':SEUIL_CONFIANCE,'imgsz':TAILLE_IMAGE,'classes':dict(enumerate(CLASSES)),'mapping_coco':correspondance_ids,
    'photos':len(photos),'ultralytics':ultralytics.__version__,'statut':'preannotations_a_corriger'},indent=2,ensure_ascii=False))
print('Pré-annotations sauvegardées :',DOSSIER_SORTIE)


# Étape 4 du notebook original.
for start in range(0,len(photos),TAILLE_LOT):
    lot=photos[start:start+TAILLE_LOT];number=start//TAILLE_LOT+1
    az=DOSSIER_SORTIE/f'annotations_yolo11_lot_{number:04d}.zip'
    iz=DOSSIER_SORTIE/f'images_lot_{number:04d}.zip'
    with zipfile.ZipFile(az,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('obj.names',''.join(c+'\n' for c in CLASSES))
        z.writestr('obj.data',f'classes = {len(CLASSES)}\nnames = obj.names\ntrain = train.txt\n')
        z.writestr('train.txt',''.join('obj_train_data/'+im.name+'\n' for im in lot))
        for im in lot:z.write(DOSSIER_LABELS/(im.stem+'.txt'),'obj_train_data/'+im.stem+'.txt')
    with zipfile.ZipFile(iz,'w',zipfile.ZIP_STORED) as z:
        for im in lot:z.write(im,im.name)  # Octets et noms originaux inchangés.
    with zipfile.ZipFile(az) as z:
        assert {'obj.names','obj.data','train.txt'}.issubset(z.namelist())
        assert len([n for n in z.namelist() if n.startswith('obj_train_data/')])==len(lot)
    print('Lot',number,':',iz.name,'+',az.name)
print('TERMINÉ. Récupérer les ZIP dans Drive :',DOSSIER_SORTIE)


# ## Afficher les annotations dans CVAT
# Pour chaque paire de ZIP portant le même numéro :
# 
# 1. Créer une **nouvelle tâche** et définir les classes exactes : car, motorcycle, tricycle, minibus, bus, truck, handcart.
# 2. Charger **images_lot_XXXX.zip** comme données de cette tâche.
# 3. Une fois la tâche créée : **Actions → Upload annotations → YOLO 1.1**.
# 4. Charger **annotations_yolo11_lot_XXXX.zip**.
# 5. Attendre le succès de l'import, ouvrir le job et examiner les boîtes.
# 6. Corriger les classes et les boîtes, ajouter les véhicules manqués et sauvegarder.
# 
# Ne pas choisir « CVAT for images » pour ces ZIP : leur format est **YOLO 1.1**. Ne pas charger le ZIP annotations comme données de création de tâche. Il n'est pas nécessaire de réimporter des photos dans une tâche qui contient déjà exactement ces images, mais **l'importation remplace les annotations existantes** : conserver une sauvegarde ou utiliser une nouvelle tâche.
# 
# Un TXT vide indique « aucune détection retenue », pas « aucun véhicule ». Les aperçus gardent les noms de classes COCO du modèle ; les TXT et ZIP utilisent vos IDs remappés.
# 
# Il n'y a ni entraînement ni best.pt nouvellement entraîné. Le téléchargement initial des poids nécessite Internet dans Colab. Les ZIP sont stockés dans Drive ; ce notebook ne se connecte pas à CVAT et n'importe rien automatiquement.
# 
# Sources : [Format YOLO de CVAT](https://docs.cvat.ai/docs/manual/advanced/formats/format-yolo/) · [Importation CVAT](https://docs.cvat.ai/docs/manual/advanced/import-datasets/) · [Prédictions Ultralytics](https://docs.ultralytics.com/modes/predict/).
