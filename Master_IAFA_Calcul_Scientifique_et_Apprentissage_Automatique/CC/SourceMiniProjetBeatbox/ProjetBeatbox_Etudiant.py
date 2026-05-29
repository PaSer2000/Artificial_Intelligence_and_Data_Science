#!/usr/bin/env python
# coding: utf-8

# # Mini-Projet 2025 :  Détection de sons de Beatbox
# 
# ## Nom de l'étudiant : SERRATOS
# 
# L'objectif de ce TP consiste à détecter et reconnaître les différents sons de beatbox à partir de fichiers sonores au format *.wav* via l'analyse cepstrale.
# 
# 
# <img src="./Sons_Beatbox.png" width="800">
# 
# 
# 
# Vous disposez d'une base d'enregistrements de beatbox. Il faut reconnaître les 12 sons beatbox suivants :
# 
#  'cymbale_expiree', 'cymbale_inspiree', 'hihat_humming', 'hihat_power ferme', 'hihat_power ouvert', 'kick_humming', 
# 'kick_power','rimshot_humming', 'rimshot_power', 'snare_humming', 'snare_power', 'snare_power inverse'
# 
# 
# **Source :** Paroni, Annalisa, et al. *Vocal drum sounds in human beatboxing: An acoustic and articulatory exploration using electromagnetic articulography.* The Journal of the Acoustical Society of America 149.1 (2021): 191-206.
# 

import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn
import math

# # Chargement des données 
# 
# 
# Le fichier source sur lequel vous travaillez est la conversion des enregistrements audio en matrice de paramètres appélés MFCC (Mel Frequency Cepstral Coefficient) en utilisant la librairie python *librosa*. Ces paramètres permettent d'extraire au mieux le contenu vocal fréquenciel du signal audio.
# 
# La matrice de données est composée d'autant de vecteurs lignes que de fichiers audio. Le nombre de colonnes correspond à la dimension du vecteur moyen représentatif des MFCC : ici 13.


# Chargement des données à partir d'un fichier matlab .mat
import scipy.io as sio
test = sio.loadmat('2016HBXX_effets_mfcc_nodeltas_XY_nozcaPerFile_zcaX.mat')

# data = matrice des données : nombre d'enregistrements x 13 coefficients
data=test['X']
data=np.array(data)
data=data.astype('float64')
print('Dimension de la matrice des MFCC :',data.shape)

# label = Numéro correspondant à un son de beatbox associé à chaque enregistrement
label=test['Y']
label=np.ravel(label)
print('Dimension des labels', label.shape)

# liste des sons de beatbox par ordre
Nomlabel=['cymbale_expiree', 'cymbale_inspiree', 'hihat_humming', 'hihat_power ferme', 'hihat_power ouvert', 'kick_humming','kick_power','rimshot_humming', 'rimshot_power', 'snare_humming', 'snare_power', 'snare_power inverse']

# Description du jeu de données 
unique, counts = np.unique(label, return_counts=True)

for i in range(unique.shape[0]):
    print('Label',unique[i],'correspond à ',Nomlabel[i],':',counts[i],'enregistrements');



# ###  Découpage de l'ensemble de données en Apprentissage et test pour les approches supervisées. 

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=42)

print("Dimension de la base d apprentissage : ",X_train.shape)
print("Dimension de la base de test : ",X_test.shape) 


# ### Fonctions pour réaliser une table de correspondance entre le label des approches non supervisées et le label réel



def retrieve_info(cluster_labels,label):
 #Associe l'étiquette la plus probable à chaque groupe dans le modèle KMeans.
 #Résultats : dictionnaire des clusters associés à chaque étiquette.

# Initialisation
  reference_labels = np.zeros((len(np.unique(cluster_labels)),1))
# Loop pour chaque label 
  for i in range(len(np.unique(cluster_labels))):
    index = np.where(cluster_labels == i,1,0)
    num = np.bincount(label[index==1]).argmax()
    reference_labels[i] = num
  return reference_labels

def correspondance(y_pred,label):
  # Correspondance entre la partition du clusterin y_pred et les classes de la vérité terrain label
  reference_labels = retrieve_info(y_pred,label)
  number_labels = np.zeros(len(y_pred))
  for i in range(len(y_pred)):
    number_labels[i] = reference_labels[y_pred[i]]
  return number_labels


# # Partie I : Classification des sons beatbox sans prétraitement 
# 
# Réaliser une classification de ces sons de beatbox par une méthode d'apprentissage supervisé et une méthode d'apprentissage non supervisé de votre choix en utilisant les codes des TP précédents.
# 
# Pour chaque résultat obtenu, calculer la matrice de confusion et  et analyser les résultats en utilisant, par exemple, l'image des signaux présenté ci-dessus pour interpréter les résultats.
# 
# ## Méthode supervisée  : 
# Indiquer quelle est la méthode supervisée et les paramètres utilisées.  Calculer la matrice de confusion associée et commentez les résultats.

# (A compléter) - Méthode utilisé RANDOMFOREST
print('------------- MÉTHODE RANDOMFOREST -------------')

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Calcul de la matrice de confusion
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
print("Matrice de confusion (méthode supervisée) :\n", cm)
print("Précision :", acc)

# Affichage matrice de confusion
plt.figure(figsize=(8,6))
plt.imshow(cm)
plt.title("Matrice de confusion - Méthode supervisée (RandomForest)")
plt.xlabel("Classe prédite")
plt.ylabel("Classe réelle")
plt.colorbar()
plt.show()


# Ici j'ai tenté d'utiliser la méthode de k plus proches voisins mais je n'ai pas réussi à faire fonctionner
# celle du TP2. J'ai décidé de laisser une version en utilisant la fonction de la lib sklearn pour comparer les accuracy

print('----------- MÉTHODE K + PROCHES VOISINS -----------')

from sklearn.neighbors import KNeighborsClassifier

clf_kn = KNeighborsClassifier(n_neighbors=1)
clf_kn.fit(X_train, y_train)
y_pred_kn = clf_kn.predict(X_test)
cm_knn = confusion_matrix(y_test, y_pred_kn)
acc_sup = accuracy_score(y_test, y_pred_kn)
print(acc_sup)

# ## Commentaires sur la méthode supervisée utilisée et les résultats :
# 
# J'ai pas réussi à faire la matrice de confusion "à la main" du coup pour cet exo et le reste du tp
# j'ai utilisé confusion_matrix()

# ## Méthode non supervisée
# 
# **Rappels :** 
# - En apprentissage non supervisé, on utilise l'ensemble des données. 
# - Comme les labels ne sont pas utilisés, il faut faire une correspondance entre le numéro de la classe définie par la méthode non supervisée et le niveau réel de stress associé. Pour cela, vous utiliserez la fonction *correspondance*.
# 
# Indiquez la méthode utilisée et ses paramètres. Calculer la matrice de confusion et analyser les résutlats. 

# (A compléter) - Méthode utilisé KMEANS
print('------------- MÉTHODE KMEANS -------------')

from sklearn.cluster import KMeans

# Kmeans sans labels ici car non supervisé
kmeans = KMeans(n_clusters=12, random_state=42)
y_pred_nosup = kmeans.fit_predict(data)
y_corr = correspondance(y_pred_nosup, label)

# Calcul de la matrice de confusion
cm_unsup = confusion_matrix(label, y_corr)
acc_unsup = accuracy_score(label, y_corr)
print("Matrice de confusion de KMeans :\n", cm_unsup)
print("Précision :", acc_unsup)

# Affichage matrice de confusion
plt.figure(figsize=(8,6))
plt.imshow(cm_unsup)
plt.title("Matrice de confusion - Méthode non supervisée : KMeans")
plt.xlabel("Classe prédite")
plt.ylabel("Classe actuelle")
plt.colorbar()
plt.show()

# ## Commentaires sur la méthode non supervisée utilisée et les résultats :
# 
# KMeans normallement est utilisé dans l'apprentissage supervisé mais ici je n'ai pas 
# fourni les labels du coup elle doit trouver les groupes, donc non supervisée.
# La méthode KMeans fonctionne correctement avec un précision modérée/élevée (~80%).
# Cela semble assez bien et on peut expliquer les erreur car il peut avoir qq sons érronées
# entre les classes acoustiquement proches.

# # Partie II : Classification des sons beatbox avec prétraitement
# 
# - Réalisez maintenant une réduction de dimension par ACP. Expliquez quelle est la dimension la + intéressante.
# - Appliquez les mêmes méthodes supervisée et non supervisée réalisées en TP et commentez les paramètres choisis. 
# - Commentez les résultats. 
# 
# ## Réduction de dimension par Analyse en Composantes Principales (ACP)
# 
# Réaliser une ACP pour réduire la dimension et visualiser les données. Discuter sur le choix de la dimension.

# (A compléter) - ACP
print('------------- MÉTHODE ACP -------------')

from sklearn.decomposition import PCA

pca = PCA()
pca.fit(data)
var_cumulee = np.cumsum(pca.explained_variance_ratio_)

plt.figure()
plt.plot(var_cumulee, marker='o')
plt.xlabel('Nombre de composantes')
plt.ylabel('Variance cumulée')
plt.title('Variance expliquée par le nombre de composantes principales')
plt.grid(True)
plt.show()

# Trouver dimension optimale (commentaire : celle qui explique 95% de la variance)
dim_opt = np.argmax(var_cumulee >= 0.95) + 1
print("Dimension optimale trouvée :", dim_opt)

# Affichage du datset (commentaire : en 3D comme le tp1)
pca_3d = PCA(n_components=3)
data_3d = pca_3d.fit_transform(data)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data_3d[:,0], data_3d[:,1], data_3d[:,2], c=label)
ax.set_title('Projection 3D des données après ACP')
plt.show()

# ## Méthode supervisée avec réduction de dimension par ACP sur l'ensemble d'apprentissage
# Après avoir réalisé une ACP sur l'ensemble d'apprentissage pour réduire la dimension, visualiser les données
# puis projeter les données de l'ensemble de test dans la base de l'ACP.
# 
# 

# (A compléter) - RANDOMFOREST AVEC ACP
print('------------- MÉTHODE supervisée RandomForest par ACP -------------')

pca = PCA(n_components=dim_opt)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# (Commentaires : on utilise la méthode fôret aléatoire, randomForest)
clf_pca = RandomForestClassifier(n_estimators=100, random_state=42)
clf_pca.fit(X_train_pca, y_train)
y_pred_pca = clf_pca.predict(X_test_pca)

# Calcul matrice de confusion
cm_pca = confusion_matrix(y_test, y_pred_pca)
acc_pca = accuracy_score(y_test, y_pred_pca)
print("Matrice de confusion - supervisée avec ACP :\n", cm_pca)
print("Précision :", acc_pca)

# Affichage matrice de confusion
plt.figure(figsize=(8,6))
plt.imshow(cm_pca)
plt.title("Matrice de confusion - Méthode supervisée RandomForest par ACP")
plt.xlabel("Classe prédite")
plt.ylabel("Classe actuelle")
plt.colorbar()
plt.show()

# ## Commentaires sur la méthode supervisée utilisée et les résultats :
# 
# La précision reste assez proche que la version de KMeans sans ACP, soit environ 80%

# 
# ## Méthode non supervisée avec réduction de dimension par ACP sur l'ensemble des données 
# 
# Réaliser une ACP sur l'ensemble des données et indiquer quelle est la méthode supervisée de votre choix et calculer la matrice de confusion associée (confusion_matrix) et le score de performance (accuracy_score)

# (A compléter) - KMEANS AVEC ACP
print('------------- MÉTHODE non supervisée KMeans par ACP -------------')

pca_global = PCA(n_components=dim_opt)
X_pca_all = pca_global.fit_transform(data)

# (Commentaires : on utilise la méthode KMeans sans labels)
kmeans_pca = KMeans(n_clusters=12, random_state=42)
y_pred_pca_unsup = kmeans_pca.fit_predict(X_pca_all)
y_corr_pca = correspondance(y_pred_pca_unsup, label)

# Calcul matrice de confusion
cm_nosup_pca = confusion_matrix(label, y_corr_pca)
acc_unsup_pca = accuracy_score(label, y_corr_pca)
print("Matrice de confusion - non supervisée avec ACP :\n", cm_nosup_pca)
print("Précision :", acc_unsup_pca)

# Affichage matrice de confusion
plt.figure(figsize=(8,6))
plt.imshow(cm_nosup_pca)
plt.title("Matrice de confusion - Méthode non supervisée KMeans par ACP")
plt.xlabel("Classe prédite")
plt.ylabel("Classe actuelle")
plt.colorbar()
plt.show()

# ## Commentaires sur la méthode supervisée utilisée et les résultats :
# 
# La précision reste assez proche que la version de KMeans sans ACP, soit environ 80%

# # Bilan de l'analyse 
# 
# Les méthodes qui trouvent l'accuracy la plus élevé sont celles utilisant une méthode supervisée
# K+proches voisins : 95%
# Forêt aléatoire : 97%
# Tandis que la méthode KMeans sans labels (non supervisée) a des accuracy beaucoup plus basses
# KMeans sans labels : 83%
# En général l'ACP n'améliore pas l'accuracy mais elle aide à visualiser la structure et réduire le bruit




