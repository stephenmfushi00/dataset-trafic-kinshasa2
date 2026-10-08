"""Script exporté de l'expérience Colab ; exécuter dans Colab avec %run.
Les calculs d'entraînement ne sont pas lancés lors de la préparation du dépôt.
"""
import json
from pathlib import Path
CONFIG_PATH = Path(__file__).resolve().parents[1] / 'config_drive.json'
if not CONFIG_PATH.exists():
    raise FileNotFoundError('Copier config_drive.example.json en config_drive.json et renseigner les IDs Drive.')
CONFIG = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))

# # Deuxième fine-tuning — annotations CVAT corrigées
# Activer un GPU Colab, se connecter avec le compte Google qui contient vos données puis exécuter dans l’ordre.
# On charge le premier best.pt, les anciennes données et les deux nouveaux lots (300 + 200 labels). Les annotations nouvelles sont considérées corrigées selon votre confirmation. Les 84 images de validation et 78 de test sont préservées grâce au manifeste précédent. Les nouvelles frames de leurs séquences sont mises à l'écart, sans modifier vos sources.
# Les contrôles complets des images et labels se font à l’exécution. Ce notebook n’a pas encore entraîné le nouveau modèle.

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
PARENT = CONFIG['drive_id_10']
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
BEST_DRIVE = CONFIG['drive_id_08']
SPLIT_DRIVE = CONFIG['drive_id_11']
NOUVEAUX = [
 {'nom':'lot300','zip':CONFIG['drive_id_12'],'labels':CONFIG['drive_id_13'],'classes':CONFIG['drive_id_14']},
 {'nom':'lot200','zip':CONFIG['drive_id_15'],'labels':CONFIG['drive_id_16'],'classes':CONFIG['drive_id_17']}
]
# Facultatif : choisir les séquences après avoir lu leur liste affichée.
SEQUENCES_VALIDATION = []
SEQUENCES_TEST = []
EXECUTION = datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid4().hex[:8]
TRAVAIL = Path('/content')/('finetuning_'+EXECUTION)
TRAVAIL.mkdir()
if not torch.cuda.is_available(): raise RuntimeError('Activer le GPU T4 dans Colab, puis réexécuter.')
print('GPU :',torch.cuda.get_device_name(0))

MODELE_DEPART = str(TRAVAIL/'depart_best.pt')


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

# Reprendre exactement la répartition du premier entraînement.
ancien_split = list(csv.DictReader(io.StringIO(octets(SPLIT_DRIVE).decode('utf-8-sig'))))
repartition = {r['image']:r['split'] for r in ancien_split}
if len(repartition)!=508: raise ValueError('Le manifeste ancien doit contenir 508 images uniques.')
if Counter(repartition.values())!=Counter({'train':346,'val':84,'test':78}): raise ValueError('Répartition ancienne inattendue.')
sequences_reservees = {r['sequence'] for r in ancien_split if r['split'] in {'val','test'}}
telecharger(BEST_DRIVE,Path(MODELE_DEPART))
# Les archives restent en local ; aucune modification des fichiers Drive.
photos_zip={}
for source in NOUVEAUX:
    archive=TRAVAIL/(source['nom']+'.zip'); telecharger(source['zip'],archive)
    destination=TRAVAIL/(source['nom']+'_images');destination.mkdir()
    catalogue=defaultdict(list)
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            if item.is_dir() or Path(item.filename).suffix.lower() not in {'.jpg','.jpeg','.png','.bmp','.webp'}: continue
            nom=Path(item.filename).name
            chemin=destination/nom
            if chemin.exists(): raise ValueError('Nom répété dans ZIP : '+nom)
            with z.open(item) as entree, chemin.open('wb') as sortie: shutil.copyfileobj(entree,sortie)
            catalogue[Path(nom).stem].append({'name':nom,'local':chemin})
    photos_zip[source['nom']]=catalogue
SOURCES += NOUVEAUX
paires=[];exclusions=[];noms_vus={};empreintes={};compteurs=Counter()
for source in SOURCES:
    classes=[s.strip() for s in octets(source['classes']).decode('utf-8-sig').splitlines() if s.strip()]
    if classes!=CLASSES:raise ValueError(f"Classes différentes dans {source['nom']}: {classes}")
    photos=defaultdict(list)
    if source['nom'] in photos_zip: photos=photos_zip[source['nom']]
    else:
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
        chemin=destination/nom
        if 'local' in photo: shutil.copy2(photo['local'],chemin)
        else: telecharger(photo['id'],chemin)
        with Image.open(chemin) as im:im.verify()
        # Même image décodée : ne pas la compter deux fois, même si le fichier a été réencodé.
        with Image.open(chemin) as im:
            rgb=im.convert('RGB');digest=hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
        signature=tuple(sorted(objets))
        if source['nom'] in photos_zip and (nom in repartition or re.split('_frame',Path(nom).stem,maxsplit=1,flags=re.I)[0] in sequences_reservees):
            exclusions.append({'fichier':nom,'raison':CONFIG['drive_id_18'],'source':source['nom']});continue
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


# ## Conserver le découpage antérieur
# Les anciennes images restent dans leur ensemble. Les nouvelles séquences admissibles enrichissent uniquement le train. Le manifeste sauvegardé permet de reproduire ce choix.

# Étape 5 du notebook original.
anciennes=[p for p in paires if p['source'] in {'premier','deuxieme'}]
if {p['image'].name for p in anciennes} != set(repartition): raise ValueError('Les anciennes données ont changé ; arrêt pour préserver la comparaison.')
entrainement=[p for p in anciennes if repartition[p['image'].name]=='train']
validation=[p for p in anciennes if repartition[p['image'].name]=='val']
test_final=[p for p in anciennes if repartition[p['image'].name]=='test']
nouvelles=[p for p in paires if p['source'] not in {'premier','deuxieme'}]
if not nouvelles: raise ValueError('Aucune nouvelle image admissible ; consulter les exclusions.')
entrainement += nouvelles
print('Nouvelles images ajoutées au train :',len(nouvelles))
classes_observees=set(compteurs)
def classes_des(items): return set().union(*(p['classes'] for p in items)) if items else set()
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


# ## Sauvegardes séparées
# Chaque exécution crée un nouveau dossier dans le dossier fourni. Le modèle précédent reste conservé. Les checkpoints sont copiés toutes les cinq époques et à la fin.

# Étape 7 du notebook original.
dossier_drive=service.files().create(body={'name':'Fine tuning YOLO round2_'+EXECUTION,
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
    'torch':torch.__version__,'images_utilisables':len(paires),'best_initial_drive':BEST_DRIVE,'split_initial_drive':SPLIT_DRIVE,'nouveaux_train':len(nouvelles),'repartition':{k:len(v) for k,v in [('train',entrainement),('val',validation),('test',test_final)]}}
(TRAVAIL/'parametres.json').write_text(json.dumps(parametres,indent=2))
sauvegarder(TRAVAIL/'parametres.json')
print('Dossier des résultats :',dossier_drive.get('webViewLink',ID_SORTIE))


# Étape 8 du notebook original.
modele=YOLO(MODELE_DEPART)
assert [modele.names[i] for i in range(len(CLASSES))]==CLASSES
# Nouveau fine-tuning depuis les poids appris, pas une reprise resume=True.
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

# Comparaison avec le modèle initial sur exactement les mêmes images et réglages.
comparaison={}
for nom,poids in [('initial',MODELE_DEPART),('round2',str(BEST))]:
    comparaison[nom]={}
    for subset in ['val','test']:
        m=YOLO(poids).val(data=str(DATA),split=subset,imgsz=TAILLE_IMAGE,device=0,
                         project=str(TRAVAIL/'comparaison'),name=nom+'_'+subset,plots=False)
        comparaison[nom][subset]=m.results_dict
rapport=TRAVAIL/'comparaison_modeles.json'
rapport.write_text(json.dumps(comparaison,indent=2,default=float));sauvegarder(rapport)
print(json.dumps(comparaison,indent=2,default=float))


# ## Après l’exécution
# Comparer les métriques par classe et le rappel, pas seulement la précision globale. Le nouveau best.pt se trouve dans un dossier de résultats distinct. L’amélioration n’est pas garantie. Ne pas régler les paramètres à répétition sur le test ; prévoir un test final indépendant si celui-ci guide les prochaines décisions.
# Si un contrôle échoue, corriger la cause indiquée au lieu de supprimer les contrôles. Les annotations vides sont exclues comme au premier entraînement ; les véritables images négatives doivent être distinguées des images non annotées avant de changer cette règle.
