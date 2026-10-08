"""Script exporté de l'expérience Colab ; exécuter dans Colab avec %run.
Les calculs d'entraînement ne sont pas lancés lors de la préparation du dépôt.
"""
import json
from pathlib import Path
CONFIG_PATH = Path(__file__).resolve().parents[1] / 'config_drive.json'
if not CONFIG_PATH.exists():
    raise FileNotFoundError('Copier config_drive.example.json en config_drive.json et renseigner les IDs Drive.')
CONFIG = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))

# # Fine-tuning YOLO — véhicules de Kinshasa
# Les données sont lues directement depuis les dossiers Drive indiqués, sans demander un ZIP.
# 
# **Dans Colab : sélectionner un GPU T4, puis exécuter les cellules dans l'ordre.** Connecter le compte `le compte Google qui contient vos données`, qui contient les données. Rien n'est modifié dans les dossiers sources.
# 
# Sources : `03_preannotations_yolo/images` + **labels uniquement** ; `Stp_deuxieme_cvat_yolo/a_annoter` + `obj_train_data`. Les fichiers vides sont exclus. Au contrôle des métadonnées : 208 fichiers non vides dans le premier ensemble, 300 dans le second ; le total final dépend des correspondances et des doublons.
# 
# Hypothèse retenue sur instruction de l'utilisateur : les fichiers non vides de ces ensembles ont été corrigés. Si un fichier non vide est encore une pré-annotation non corrigée, il faut le retirer de l'ensemble avant l'entraînement.
# 
# Le notebook vérifie les annotations et les classes, sépare les séquences, entraîne un YOLOv8n préentraîné, puis évalue le meilleur modèle. Aucun entraînement n'a été exécuté par ChatGPT.

# Étape 2 du notebook original.
from google.colab import auth
import google.auth
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
from uuid import uuid4
import io, json, csv, math, re, random, shutil, hashlib, zipfile
import torch, yaml, ultralytics
from ultralytics import YOLO
from PIL import Image

auth.authenticate_user()
identifiants, _ = google.auth.default()
service = build('drive', 'v3', credentials=identifiants, cache_discovery=False)
PARENT = CONFIG['drive_id_01']
CLASSES = ['car','motorcycle','tricycle','minibus','bus','truck','handcart']
SOURCES = [
    {'nom':'premier', 'images':CONFIG['drive_id_02'],
     'labels':CONFIG['drive_id_03'],'classes':CONFIG['drive_id_04']},
    {'nom':'deuxieme', 'images':CONFIG['drive_id_05'],
     'labels':CONFIG['drive_id_06'],'classes':CONFIG['drive_id_07']}
]
EPOCHS = 80
TAILLE_IMAGE = 960
BATCH = 8  # Réduire à 4 si le GPU manque de mémoire.
GRAINE = 42
MODELE_DEPART = 'yolov8n.pt'
# Facultatif : choisir les séquences après avoir lu leur liste affichée.
SEQUENCES_VALIDATION = []
SEQUENCES_TEST = []
EXECUTION = datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid4().hex[:8]
TRAVAIL = Path('/content')/('finetuning_'+EXECUTION)
TRAVAIL.mkdir()
if not torch.cuda.is_available(): raise RuntimeError('Activer le GPU T4 dans Colab, puis réexécuter.')
print('GPU :',torch.cuda.get_device_name(0))


# Étape 3 du notebook original.
def lister(dossier):
    elements=[]; page=None
    while True:
        reponse=service.files().list(q=f"'{dossier}' in parents and trashed = false",
            fields='nextPageToken,files(id,name,mimeType,size)',pageSize=1000,pageToken=page).execute()
        elements.extend(reponse.get('files',[]));page=reponse.get('nextPageToken')
        if not page:return elements

def octets(file_id):
    contenu=io.BytesIO()
    telechargement=MediaIoBaseDownload(contenu,service.files().get_media(fileId=file_id))
    termine=False
    while not termine:_,termine=telechargement.next_chunk()
    return contenu.getvalue()

def telecharger(file_id,destination):
    with destination.open('wb') as f:
        transfert=MediaIoBaseDownload(f,service.files().get_media(fileId=file_id))
        termine=False
        while not termine:_,termine=transfert.next_chunk()

def images_recursives(dossier):
    resultat=[]
    for item in lister(dossier):
        if item['mimeType']=='application/vnd.google-apps.folder':resultat.extend(images_recursives(item['id']))
        elif Path(item['name']).suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.webp'}:resultat.append(item)
    return resultat

def lire_label(texte,nom):
    lignes=[]
    for numero,ligne in enumerate(texte.splitlines(),1):
        if not ligne.strip():continue
        valeurs=ligne.split()
        if len(valeurs)!=5:raise ValueError(f'{nom}, ligne {numero} : 5 valeurs attendues')
        c,x,y,w,h=map(float,valeurs)
        if not all(math.isfinite(v) for v in (c,x,y,w,h)):raise ValueError(f'{nom}: valeur non finie')
        if not c.is_integer() or not 0<=c<len(CLASSES):raise ValueError(f'{nom}: classe invalide')
        if not (0<=x<=1 and 0<=y<=1 and 0<w<=1 and 0<h<=1):raise ValueError(f'{nom}: boîte invalide')
        lignes.append((int(c),x,y,w,h))
    return lignes

paires=[];exclusions=[];noms_vus={};empreintes={};compteurs=Counter()
for source in SOURCES:
    classes=[s.strip() for s in octets(source['classes']).decode('utf-8-sig').splitlines() if s.strip()]
    if classes!=CLASSES:raise ValueError(f"Classes différentes dans {source['nom']}: {classes}")
    photos=defaultdict(list)
    for item in images_recursives(source['images']):photos[Path(item['name']).stem].append(item)
    destination=TRAVAIL/source['nom'];destination.mkdir()
    gardees=0
    for label in sorted(lister(source['labels']),key=lambda x:x['name']):
        if not label['name'].lower().endswith('.txt'):continue
        if label['name'] in {'train.txt','valid.txt','classes.txt'}:continue
        texte=octets(label['id']).decode('utf-8-sig') if int(label.get('size',0)) else ''
        objets=lire_label(texte,label['name'])
        if not objets:
            exclusions.append({'fichier':label['name'],'raison':'annotation_vide','source':source['nom']});continue
        candidats=photos.get(Path(label['name']).stem,[])
        if len(candidats)!=1:raise ValueError(f"{label['name']}: {len(candidats)} photos correspondantes")
        photo=candidats[0];nom=photo['name']
        if Path(nom).name!=nom:raise ValueError('Nom de fichier non sûr')
        chemin=destination/nom;telecharger(photo['id'],chemin)
        with Image.open(chemin) as im:im.verify()
        # Même image décodée : ne pas la compter deux fois, même si le fichier a été réencodé.
        with Image.open(chemin) as im:
            rgb=im.convert('RGB');digest=hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
        signature=tuple(sorted(objets))
        if digest in empreintes:
            if empreintes[digest]!=signature:raise ValueError(f'Doublon avec annotations différentes : {nom}')
            exclusions.append({'fichier':nom,'raison':'doublon_identique','source':source['nom']});continue
        if Path(nom).stem.casefold() in noms_vus:raise ValueError(f'Nom ambigu : {nom}, arrêt sans renommage.')
        noms_vus[Path(nom).stem.casefold()]=nom;empreintes[digest]=signature
        chemin_label=destination/(Path(nom).stem+'.txt');chemin_label.write_text(texte)
        seq=re.split('_frame',Path(nom).stem,maxsplit=1,flags=re.I)[0]
        paires.append({'image':chemin,'label':chemin_label,'sequence':seq,'classes':set(o[0] for o in objets),'source':source['nom']})
        compteurs.update(o[0] for o in objets);gardees+=1
        if gardees%25==0:print(source['nom'],gardees,'photos téléchargées')
    print(source['nom'],':',gardees,'images utilisables')
if len(paires)<20:raise ValueError('Trop peu de paires utilisables.')
print('TOTAL :',len(paires),'images uniques')
for i,nom in enumerate(CLASSES):print(i,nom,':',compteurs[i],'objets')
print('Classes absentes :',[CLASSES[i] for i in range(len(CLASSES)) if not compteurs[i]])


# ## Séparer les séquences et préserver le test
# Le nom avant `_frame` identifie la séquence. Les frames d'une même séquence restent dans un seul ensemble. Vérifier que cette règle reflète bien vos vidéos ; si deux angles appartiennent à la même prise, les regrouper selon votre protocole.
# 
# La séparation automatique cherche environ 70 % train / 15 % validation / 15 % test en gardant toutes les classes observées dans le train. Les proportions peuvent différer car les groupes sont indivisibles. Si une classe n'est pas représentée dans un ensemble, les résultats ne permettent pas de l'évaluer dans cet ensemble.

# Étape 5 du notebook original.
groupes=defaultdict(list)
for p in paires:groupes[p['sequence']].append(p)
for seq,items in sorted(groupes.items()):print(seq,':',len(items),'images')
if len(groupes)<3:raise ValueError('Il faut au moins 3 séquences pour ce protocole train/val/test.')
cles=sorted(groupes);classes_observees=set(compteurs)
def classes_des(items):return set().union(*(p['classes'] for p in items)) if items else set()
def separer(val,test):
    train=[p for k,v in groupes.items() if k not in val|test for p in v]
    validation=[p for k,v in groupes.items() if k in val for p in v]
    evaluation=[p for k,v in groupes.items() if k in test for p in v]
    return train,validation,evaluation
if SEQUENCES_VALIDATION or SEQUENCES_TEST:
    val,test=set(SEQUENCES_VALIDATION),set(SEQUENCES_TEST)
    if not val or not test or val&test or not (val|test).issubset(groupes):raise ValueError('Séquences invalides')
    entrainement,validation,test_final=separer(val,test)
else:
    rng=random.Random(GRAINE);meilleur=None
    for tentative in range(1000):
        ordre=cles.copy();rng.shuffle(ordre)
        # Réserver au moins une séquence à chaque ensemble.
        n=max(1,round(0.15*len(ordre)))
        val=set(ordre[:n]);test=set(ordre[n:2*n])
        tr,va,te=separer(val,test)
        if not tr or not va or not te or classes_des(tr)!=classes_observees:continue
        score=abs(len(va)/len(paires)-0.15)+abs(len(te)/len(paires)-0.15)
        # Préférer aussi une couverture des classes dans les ensembles d'évaluation.
        score+=0.1*(len(classes_observees-classes_des(va))+len(classes_observees-classes_des(te)))
        if meilleur is None or score<meilleur[0]:meilleur=(score,tr,va,te)
    if meilleur is None:raise ValueError('Séparation impossible sans perdre une classe du train ; ajouter des séquences corrigées.')
    _,entrainement,validation,test_final=meilleur
if not entrainement or classes_des(entrainement)!=classes_observees:raise ValueError('Train incomplet')
DS=TRAVAIL/'dataset';manifeste=[]
for subset,items in [('train',entrainement),('val',validation),('test',test_final)]:
    for kind in ['images','labels']:(DS/kind/subset).mkdir(parents=True)
    for p in items:
        shutil.copy2(p['image'],DS/'images'/subset/p['image'].name)
        shutil.copy2(p['label'],DS/'labels'/subset/p['label'].name)
        manifeste.append({'image':p['image'].name,'source':p['source'],'sequence':p['sequence'],'split':subset})
    print(subset,len(items),'images ; classes absentes :',[CLASSES[i] for i in classes_observees-classes_des(items)])
DATA=DS/'data.yaml'
DATA.write_text(yaml.safe_dump({'path':str(DS),'train':'images/train','val':'images/val','test':'images/test','names':dict(enumerate(CLASSES))},sort_keys=False))
with (TRAVAIL/'split.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['image','source','sequence','split']);w.writeheader();w.writerows(manifeste)
(TRAVAIL/'audit.json').write_text(json.dumps({'images_utilisables':len(paires),'objets_par_classe':dict(compteurs),'exclusions':exclusions},indent=2,ensure_ascii=False))


# ## Sauvegarder dans Drive
# Chaque exécution crée un dossier `Fine tuning YOLO Kinshasa_<date>_<identifiant>` dans votre dossier fourni. Les sources restent inchangées. Les poids sont copiés toutes les cinq époques et en fin d'entraînement ; aucune copie de checkpoint n'est garantie avant cette première sauvegarde.

# Étape 7 du notebook original.
dossier_drive=service.files().create(body={'name':'Fine tuning YOLO Kinshasa_'+EXECUTION,
    'mimeType':'application/vnd.google-apps.folder','parents':[PARENT]},fields='id,webViewLink').execute()
ID_SORTIE=dossier_drive['id'];fichiers_drive={}
def sauvegarder(chemin,nom=None):
    chemin=Path(chemin);nom=nom or chemin.name
    media=MediaFileUpload(str(chemin),resumable=True)
    if nom in fichiers_drive:
        service.files().update(fileId=fichiers_drive[nom],media_body=media).execute()
    else:
        resultat=service.files().create(body={'name':nom,'parents':[ID_SORTIE]},media_body=media,fields='id').execute()
        fichiers_drive[nom]=resultat['id']
for chemin in [TRAVAIL/'split.csv',TRAVAIL/'audit.json',DATA]:sauvegarder(chemin)
parametres={'modele_depart':MODELE_DEPART,'epochs':EPOCHS,'imgsz':TAILLE_IMAGE,'batch':BATCH,
    'seed':GRAINE,'classes':dict(enumerate(CLASSES)),'ultralytics':ultralytics.__version__,
    'torch':torch.__version__,'images_utilisables':len(paires)}
(TRAVAIL/'parametres.json').write_text(json.dumps(parametres,indent=2))
sauvegarder(TRAVAIL/'parametres.json')
print('Dossier des résultats :',dossier_drive.get('webViewLink',ID_SORTIE))


# Étape 8 du notebook original.
modele=YOLO(MODELE_DEPART)
def sauvegarde_periodique(trainer):
    if (trainer.epoch+1)%5==0:
        for chemin in [trainer.best,trainer.last,Path(trainer.save_dir)/'results.csv']:
            if Path(chemin).is_file():sauvegarder(chemin)
        print('Sauvegarde Drive effectuée à l’époque',trainer.epoch+1)
modele.add_callback('on_model_save',sauvegarde_periodique)
modele.train(data=str(DATA),epochs=EPOCHS,imgsz=TAILLE_IMAGE,batch=BATCH,device=0,
    seed=GRAINE,patience=20,workers=2,project=str(TRAVAIL/'runs'),name='vehicules',plots=True)
DOSSIER_ENTRAINEMENT=Path(modele.trainer.save_dir)
BEST=DOSSIER_ENTRAINEMENT/'weights'/'best.pt'
if not BEST.is_file():raise FileNotFoundError('best.pt non trouvé')
for chemin in [BEST,DOSSIER_ENTRAINEMENT/'weights'/'last.pt',DOSSIER_ENTRAINEMENT/'results.csv']:
    if chemin.is_file():sauvegarder(chemin)
print('Fine-tuning terminé et poids sauvegardés.')


# Étape 9 du notebook original.
meilleur_modele=YOLO(str(BEST))
assert [meilleur_modele.names[i] for i in range(len(CLASSES))]==CLASSES
for subset in ['val','test']:
    mesures=meilleur_modele.val(data=str(DATA),split=subset,imgsz=TAILLE_IMAGE,device=0,
                               project=str(TRAVAIL/'evaluation'),name=subset,plots=True)
    chemin=TRAVAIL/f'metriques_{subset}.json'
    chemin.write_text(json.dumps(mesures.results_dict,indent=2,default=float));sauvegarder(chemin)
    # Les indices de classes sont ceux réellement présents dans l'évaluation.
    indices=mesures.box.ap_class_index
    details=[]
    for position,cid in enumerate(indices):
        precision,rappel,map50,map5095=mesures.box.class_result(position)
        details.append({'id':int(cid),'classe':CLASSES[int(cid)],'precision':float(precision),
                        'rappel':float(rappel),'mAP50':float(map50),'mAP50_95':float(map5095)})
    chemin=TRAVAIL/f'metriques_par_classe_{subset}.json'
    chemin.write_text(json.dumps(details,indent=2));sauvegarder(chemin)
# Courbes, matrices, configuration et rapports ; les images sources ne sont pas incluses.
archive=TRAVAIL/'rapports_entrainement.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for racine in [DOSSIER_ENTRAINEMENT,TRAVAIL/'evaluation']:
        for p in racine.rglob('*'):
            if p.is_file() and p.suffix!='.pt':z.write(p,str(p.relative_to(TRAVAIL)))
sauvegarder(archive)
print('TERMINÉ : best.pt, last.pt, résultats, métriques et courbes sont sur Drive.')
print('Dossier :',dossier_drive.get('webViewLink',ID_SORTIE))


# ## Utiliser ensuite best.pt
# Ce notebook fait le fine-tuning et l'évaluation, pas la pré-annotation de nouveaux lots. Dans le notebook de pré-annotation, charger le `best.pt` appris, **retirer le mapping COCO** et utiliser directement les IDs de ce modèle : 0 car, 1 motorcycle, 2 tricycle, 3 minibus, 4 bus, 5 truck, 6 handcart. Une classe sans exemples corrigés ne peut pas être considérée comme apprise.
# 
# Le test doit rester réservé à l'évaluation : ne pas ajuster les paramètres en fonction de ses résultats à répétition. Pour la pré-annotation, régler les seuils sur la validation puis corriger dans CVAT.
# 
# Si Colab s'interrompt, les poids déjà sauvegardés restent sur Drive. Pour reprendre avec `last.pt`, il faut reconstituer les mêmes données, le même découpage et les chemins attendus par le checkpoint ; ne pas simplement changer le dataset puis utiliser `resume=True`.
# 
# Références : [entraînement YOLO](https://docs.ultralytics.com/modes/train/) ; [YOLOv8](https://docs.ultralytics.com/models/yolov8/) ; [callbacks](https://docs.ultralytics.com/usage/callbacks/).
