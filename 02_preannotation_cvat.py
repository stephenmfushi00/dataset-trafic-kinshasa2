"""Script exporté de l'expérience Colab ; exécuter dans Colab avec %run.
Les calculs d'entraînement ne sont pas lancés lors de la préparation du dépôt.
"""
import json
from pathlib import Path
CONFIG_PATH = Path(__file__).resolve().parents[1] / 'config_drive.json'
if not CONFIG_PATH.exists():
    raise FileNotFoundError('Copier config_drive.example.json en config_drive.json et renseigner les IDs Drive.')
CONFIG = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))

# # Pré-annoter les nouvelles frames avec votre best.pt
# Ce notebook utilise le modèle déjà fine-tuné. **Il n'entraîne pas de nouveau modèle.**
# 
# Ouvrir dans Colab, choisir un GPU T4 si disponible, puis exécuter les cellules. Connecter `le compte Google qui contient vos données`. Le modèle et toutes les nouvelles photos sont téléchargés directement depuis les dossiers indiqués ; aucune limite de 54 ou 300 images.
# 
# Les sources restent inchangées. Chaque lancement crée un nouveau dossier de résultats dans le dossier des nouvelles frames. La prédiction est une aide à l'annotation : corriger manuellement dans CVAT avant de considérer les labels comme validés.

# Étape 2 du notebook original.
from google.colab import auth
import google.auth
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from pathlib import Path
from datetime import datetime
from uuid import uuid4
from collections import defaultdict
import io, json, csv, math, zipfile, hashlib
import torch, ultralytics
from ultralytics import YOLO

auth.authenticate_user()
identifiants,_=google.auth.default()
service=build('drive','v3',credentials=identifiants,cache_discovery=False)
ID_MODELE=CONFIG['drive_id_08']
ID_FRAMES=CONFIG['drive_id_09']
CLASSES=['car','motorcycle','tricycle','minibus','bus','truck','handcart']
SEUIL_CONFIANCE=0.60  # Point de départ : ajuster selon la validation et les erreurs observées.
TAILLE_IMAGE=960
TAILLE_LOT=300
EXECUTION=datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid4().hex[:8]
TRAVAIL=Path('/content')/('predictions_'+EXECUTION);TRAVAIL.mkdir()
DOSSIER_PHOTOS=TRAVAIL/'images';DOSSIER_PHOTOS.mkdir()
DOSSIER_LABELS=TRAVAIL/'labels';DOSSIER_LABELS.mkdir()
APERCUS=TRAVAIL/'apercus';APERCUS.mkdir()
APPAREIL=0 if torch.cuda.is_available() else 'cpu'
print('Calcul :',APPAREIL,'; CPU plus lent' if APPAREIL=='cpu' else '')


# Étape 3 du notebook original.
def lister(dossier):
    resultat=[];page=None
    while True:
        reponse=service.files().list(q=f"'{dossier}' in parents and trashed = false",
            fields='nextPageToken,files(id,name,mimeType)',pageSize=1000,pageToken=page).execute()
        resultat.extend(reponse.get('files',[]));page=reponse.get('nextPageToken')
        if not page:return resultat

def trouver_photos(dossier):
    resultat=[]
    for item in lister(dossier):
        if item['mimeType']=='application/vnd.google-apps.folder':
            # Ne pas retraiter les images archivées lors d'une ancienne exécution.
            if not item['name'].startswith('Preannotations_best_'):resultat.extend(trouver_photos(item['id']))
        elif Path(item['name']).suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.webp'}:resultat.append(item)
    return resultat

def telecharger(file_id,chemin):
    with chemin.open('wb') as f:
        transfert=MediaIoBaseDownload(f,service.files().get_media(fileId=file_id))
        termine=False
        while not termine:_,termine=transfert.next_chunk()

photos=sorted(trouver_photos(ID_FRAMES),key=lambda x:x['name'])
if not photos:raise ValueError('Aucune photo accessible dans ce dossier.')
noms=defaultdict(list)
for photo in photos:
    nom=photo['name']
    if Path(nom).name!=nom:raise ValueError('Nom de fichier non sûr')
    noms[Path(nom).stem.casefold()].append(nom)
if any(len(v)>1 for v in noms.values()):raise ValueError('Noms ambigus : arrêt pour éviter écrasement ou renommage.')
print('TOTAL à pré-annoter :',len(photos))
BEST=TRAVAIL/'best.pt';telecharger(ID_MODELE,BEST)
modele=YOLO(str(BEST))
if [modele.names[i] for i in range(len(modele.names))]!=CLASSES:
    raise ValueError(f'Classes du modèle différentes : {modele.names}')
print('Classes vérifiées :',modele.names)
# Utilisation directe des IDs appris : pas de mapping COCO.


# Étape 4 du notebook original.
sortie=service.files().create(body={'name':'Preannotations_best_'+EXECUTION,
    'mimeType':'application/vnd.google-apps.folder','parents':[ID_FRAMES]},fields='id,webViewLink').execute()
ID_SORTIE=sortie['id']
def envoyer(chemin):
    return service.files().create(body={'name':chemin.name,'parents':[ID_SORTIE]},
        media_body=MediaFileUpload(str(chemin),resumable=True),fields='id,webViewLink').execute()
print('Dossier des résultats :',sortie.get('webViewLink',ID_SORTIE))
bilan=[]
with (TRAVAIL/'confiances.csv').open('w',newline='') as f:
    redacteur=csv.writer(f);redacteur.writerow(['image','id','classe','confiance','xc','yc','w','h'])
    for index,photo in enumerate(photos,1):
        chemin=DOSSIER_PHOTOS/photo['name'];telecharger(photo['id'],chemin)
        prediction=modele.predict(source=str(chemin),conf=SEUIL_CONFIANCE,imgsz=TAILLE_IMAGE,
                                   device=APPAREIL,max_det=1000,verbose=False)[0]
        lignes=[]
        if prediction.boxes is not None:
            for cid,confiance,coords in zip(prediction.boxes.cls.cpu().tolist(),prediction.boxes.conf.cpu().tolist(),prediction.boxes.xywhn.cpu().tolist()):
                cid=int(cid)
                if confiance<SEUIL_CONFIANCE:continue
                x,y,w,h=coords
                if not 0<=cid<len(CLASSES) or not all(math.isfinite(v) for v in coords) or not (0<=x<=1 and 0<=y<=1 and 0<w<=1 and 0<h<=1):
                    raise ValueError(f'Détection invalide : {chemin.name}')
                lignes.append(f'{cid} {x:.6f} {y:.6f} {w:.6f} {h:.6f}')
                redacteur.writerow([chemin.name,cid,CLASSES[cid],confiance,*coords])
        (DOSSIER_LABELS/(chemin.stem+'.txt')).write_text('\n'.join(lignes)+('\n' if lignes else ''))
        bilan.append({'image':chemin.name,'detections':len(lignes),'statut':'preannotation_non_verifiee'})
        if index<=20:prediction.save(filename=str(APERCUS/chemin.name))
        if index%25==0 or index==len(photos):print(index,'/',len(photos),'images traitées')
        # Sauvegarder chaque lot complet sans attendre la fin de tout le dossier.
        if index%TAILLE_LOT==0 or index==len(photos):
            debut=((index-1)//TAILLE_LOT)*TAILLE_LOT
            lot=photos[debut:index];numero=debut//TAILLE_LOT+1
            archive_images=TRAVAIL/f'images_lot_{numero:04d}.zip'
            archive_labels=TRAVAIL/f'annotations_yolo11_lot_{numero:04d}.zip'
            with zipfile.ZipFile(archive_images,'w',zipfile.ZIP_STORED) as z:
                for item in lot:z.write(DOSSIER_PHOTOS/item['name'],item['name'])
            with zipfile.ZipFile(archive_labels,'w',zipfile.ZIP_DEFLATED) as z:
                z.writestr('obj.names',''.join(c+'\n' for c in CLASSES))
                z.writestr('obj.data',f'classes = {len(CLASSES)}\nnames = obj.names\ntrain = train.txt\n')
                z.writestr('train.txt',''.join('obj_train_data/'+item['name']+'\n' for item in lot))
                for item in lot:
                    label=Path(item['name']).stem+'.txt';z.write(DOSSIER_LABELS/label,'obj_train_data/'+label)
            with zipfile.ZipFile(archive_labels) as z:
                assert {'obj.names','obj.data','train.txt'}.issubset(z.namelist())
                assert len([n for n in z.namelist() if n.startswith('obj_train_data/')])==len(lot)
            # Vérification byte-for-byte des photos copiées, sans réencodage.
            with zipfile.ZipFile(archive_images) as z:
                for item in lot:
                    p=DOSSIER_PHOTOS/item['name']
                    assert hashlib.sha256(z.read(item['name'])).digest()==hashlib.sha256(p.read_bytes()).digest()
            envoyer(archive_images);envoyer(archive_labels)
            print('Lot',numero,'sauvegardé dans Drive :',len(lot),'images')


# Étape 5 du notebook original.
with (TRAVAIL/'rapport_images.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['image','detections','statut']);w.writeheader();w.writerows(bilan)
(TRAVAIL/'parametres.json').write_text(json.dumps({'modele_drive_id':ID_MODELE,
    'frames_drive_id':ID_FRAMES,'classes':dict(enumerate(CLASSES)),'confiance':SEUIL_CONFIANCE,
    'imgsz':TAILLE_IMAGE,'images':len(photos),'images_sans_detection':sum(b['detections']==0 for b in bilan),
    'ultralytics':ultralytics.__version__,'statut':'preannotations_a_corriger'},indent=2,ensure_ascii=False))
(TRAVAIL/'classes.txt').write_text(''.join(c+'\n' for c in CLASSES))
archive=TRAVAIL/'apercus.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_STORED) as z:
    for p in APERCUS.iterdir():z.write(p,p.name)
for p in [TRAVAIL/'rapport_images.csv',TRAVAIL/'parametres.json',TRAVAIL/'classes.txt',TRAVAIL/'confiances.csv',archive]:envoyer(p)
print('TERMINÉ — toutes les nouvelles frames ont été pré-annotées.')
print('Résultats :',sortie.get('webViewLink',ID_SORTIE))


# ## Voir le résultat puis importer dans CVAT
# 1. Ouvrir `apercus.zip` pour examiner les premières prédictions. Les boîtes sont dessinées uniquement sur les aperçus ; les photos originales sont préservées.
# 2. Pour chaque lot, créer une **nouvelle tâche** CVAT avec les sept classes et `images_lot_XXXX.zip`.
# 3. Après création : **Actions → Upload annotations → YOLO 1.1**, puis choisir `annotations_yolo11_lot_XXXX.zip` du même numéro.
# 4. Ouvrir le job, corriger les faux positifs, les classes, les boîtes et les véhicules manqués. Sauvegarder.
# 
# Ne pas sélectionner « CVAT for images » pour ces ZIP. Un TXT vide indique seulement qu'aucune détection n'a dépassé le seuil. Une confiance élevée ne garantit pas l'exactitude.
# 
# L'import dans une tâche existante remplace ses annotations : sauvegarder ou utiliser une nouvelle tâche. Ce notebook ne se connecte pas à CVAT.
# 
# Ces nouvelles frames servent à la pré-annotation. Si elles proviennent de vidéos déjà utilisées pour l'apprentissage, elles ne constituent pas une évaluation indépendante de généralisation.
# 
# En cas d'interruption, les lots déjà envoyés restent dans Drive. Une nouvelle exécution repart de toutes les photos et crée un autre dossier de résultats. Choisir les fichiers d'une seule exécution pour éviter les doublons.
# 
# Sources : [Prédictions Ultralytics](https://docs.ultralytics.com/modes/predict/) · [YOLO 1.1 CVAT](https://docs.cvat.ai/docs/manual/advanced/formats/format-yolo/).
