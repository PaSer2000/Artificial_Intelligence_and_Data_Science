#!/usr/bin/env python
# coding: utf-8

# 
# # Mini-Projet TP 2 :  Détection d'activités à partir d'accéléromètre (1ère partie)
# 
# ## Nom de l'étudiant : SERRATOS
# 
# Dans ce projet, vous prédirez 5 différentes activités humaines en utilisant des méthodes supervisées :
# 
#  **3-Debout, 4-Marcher,  5-Velo, 7-aspirateur_nettoyage et 8-Repassage**
# 
# en utilisant le capteur d'accéléromètre placé à deux endroits différents du corps :  torse et cheville.
# 
# Pour réaliser cette classification, nous disposons de 156 caractéristiques statistiques (31), temporelles (14) et spectrales (111) extraits des profils temporels de 128 pas de temps des accéléromètres. Ces caractéristiques sont extraites avec la librairie TSFEL (*Time Series Feature Extraction Library*). les mesures respectivement en $ms^{-2}$  sont prélevées sur les 3-axes à une fréquence d'échantillonnage de 100Hz. 
# 
# Chaque mesure est étiquetée.
# 
# <img src="./HARDataset.JPG" width="800" height="600"  >
# 
# Il est fourni pour ce capteur et chaque position, une matrice des mesures collectées avec l'étiquette de l'activité (c'est-à-dire la verité terrain). 
# 
# Vous disposez des matrices suivantes
# - **X_train_Torse_TSFEL_TP2.npy** et **X_test_Torse_TSFEL_TP2.npy** deux matrices d'apprentissage et de test de dimensions 468 (concatenation des 156 caractéristiques extraites des signaux de 128 pas de temps sur les 3 axes X, Y et Z) du capteur d'accéléromètre situé sur le torse.
# -  **X_train_Cheville_TSFEL_TP2.npy** et **X_test_Cheville_TSFEL_TP2.npy** deux matrices d'apprentissage et de test de dimensions 468 (concatenation des 156 caractéristiques extraites des signaux de 128 pas de temps sur les 3 axes X, Y et Z) du capteur d'accéléromètre situé sur la cheville.
# -  **LabelTrain_TP2.npy** et **LabelTest_TP2.npy** les vecteurs respectifs des labels pour la base d'apprentissage et de test dont le numéro correspond à une des 5 activités présentées ci-dessus.
# -  **ListeVariableStat.npy**, **ListeVariableTemps.npy** et **ListeVariableSpectral.npy** les listes des variables statistiques, temporelles et spectrales respectives. 


## ====================================================================================================================================================== ##

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from scipy import stats
import sklearn

## ====================================================================================================================================================== ##

# # Chargement des données des capteurs placés sur le torse et la cheville

# CAPTEUR TORSE
# Chargement des données d'apprentissage
Xtrain_Torse=np.load('./Xtrain_Torse_TSFEL_TP2.npy')
print('Taille de la base d apprentissage :', Xtrain_Torse.shape)

# Chargement des données de test
Xtest_Torse=np.load('./Xtest_Torse_TSFEL_TP2.npy')
print('Taille de la base de test :', Xtest_Torse.shape)

# CAPTEUR CHEVILLE
Xtrain_Cheville=np.load('./Xtrain_Cheville_TSFEL_TP2.npy')
print('Taille de la base d apprentissage :', Xtrain_Cheville.shape)

# Chargement des données de test
Xtest_Cheville=np.load('./Xtest_Cheville_TSFEL_TP2.npy')
print('Taille de la base de test :', Xtest_Cheville.shape)

# LABELS
y_train=np.load('./LabelTrain_TP2.npy')
y_test=np.load('./LabelTest_TP2.npy')
print('Taille label apprentissage :',y_train.shape, 'Taille label test :',y_test.shape)


# liste des variables TSFEL : 156 caractéristiques décrites en statistiques (31), temporelles (14) et spectrales (111)
ListeVariablesStat=np.load('./ListeVariableStat.npy',allow_pickle=True)
print('Nombre de variables statistiques :',ListeVariablesStat.shape)
#print('Liste Variables Statistiques :', ListeVariablesStat)

ListeVariablesTemps=np.load('./ListeVariableTemps.npy',allow_pickle=True)
print('Nombre de variables temporelles :',ListeVariablesTemps.shape)
#print('Liste Variables Temporelles :',ListeVariablesTemps)

ListeVariablesSpectral=np.load('./ListeVariableSpectral.npy',allow_pickle=True)
print('Nombre de variables spectrales :',ListeVariablesSpectral.shape)
#print('Liste Variables Spectrales :',ListeVariablesSpectral)

## ====================================================================================================================================================== ##

# # Partie I : Classification des activités sans prétraitement
# 
# Réaliser une classification des activités par deux méthodes d'apprentissage supervisé : les kppv du TP2 et la classification bayésienne du TP4 avec la fonction de scikit-learn GaussianNB.
# 
# 
# ## Méthodes supervisées  : 
# Indiquer quelle est la méthode supervisée et les paramètres utilisées.  Calculer la matrice de confusion associée, le pourcentage de bonne classification et commentez les résultats.

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

print("\n\n--- Partie I : Sans Prétraitement (StandardScaler + PCA) ---")

# Paramètres choisis (on les impose)
k = 5  # pour le knn

# Instanciation des classifieurs
knn = KNeighborsClassifier(n_neighbors=k)
nb  = GaussianNB()

# ### Premier capteur 

print("\n--- Capteur : TORSE (sans prétraitement) ---")


# k-NN sur données brutes (TORSE)
knn.fit(Xtrain_Torse, y_train)
y_pred_knn_torse = knn.predict(Xtest_Torse)
mc_knn_torse = confusion_matrix(y_test, y_pred_knn_torse)
acc_knn_torse = accuracy_score(y_test, y_pred_knn_torse)

print("k-NN (k={}) - Matrice de confusion (TORSE):".format(k))
print(mc_knn_torse)
print("k-NN (TORSE) - Pourcentage de bonne classification: {:.2f}%".format(100*acc_knn_torse))
print(classification_report(y_test, y_pred_knn_torse, digits=4))

# GaussianNB sur données brutes (TORSE)
nb.fit(Xtrain_Torse, y_train)
y_pred_nb_torse = nb.predict(Xtest_Torse)
mc_nb_torse = confusion_matrix(y_test, y_pred_nb_torse)
acc_nb_torse= accuracy_score(y_test, y_pred_nb_torse)

print("GaussianNB - Matrice de confusion (TORSE):")
print(mc_nb_torse)
print("GaussianNB (TORSE) - Pourcentage de bonne classification: {:.2f}%".format(100*acc_nb_torse))
print(classification_report(y_test, y_pred_nb_torse, digits=4))

# Calcul de la matrice de confusion et du pourcentage de bonne classification

# L'affichage de la matrice de confusion et du pourcentage de bonne classification est fait ci-dessus.

## ====================================================================================================================================================== ##

# ### Deuxième capteur

print("\n--- Capteur : CHEVILLE (sans prétraitement) ---")

# k-NN sur données brutes (CHEVILLE)
knn.fit(Xtrain_Cheville, y_train)
y_pred_knn_cheville = knn.predict(Xtest_Cheville)
mc_knn_cheville = confusion_matrix(y_test, y_pred_knn_cheville)
acc_knn_cheville = accuracy_score(y_test, y_pred_knn_cheville)

print("k-NN (k={}) - Matrice de confusion (CHEVILLE):".format(k))
print(mc_knn_cheville)
print("k-NN (CHEVILLE) - Pourcentage de bonne classification: {:.2f}%".format(100*acc_knn_cheville))
print(classification_report(y_test, y_pred_knn_cheville, digits=4))

# GaussianNB sur données brutes (CHEVILLE)
nb.fit(Xtrain_Cheville, y_train)
y_pred_nb_cheville = nb.predict(Xtest_Cheville)
mc_nb_cheville = confusion_matrix(y_test, y_pred_nb_cheville)
acc_nb_cheville = accuracy_score(y_test, y_pred_nb_cheville)

print("GaussianNB - Matrice de confusion (CHEVILLE):")
print(mc_nb_cheville)
print("GaussianNB (CHEVILLE) - Pourcentage de bonne classification: {:.2f}%".format(100*acc_nb_cheville))
print(classification_report(y_test, y_pred_nb_cheville, digits=4))

# Calcul de la matrice de confusion et du pourcentage de bonne classification

# L'affichage de la matrice de confusion et du pourcentage de bonne classification est fait ci-dessus.

# ## Commentaires sur les méthodes supervisées utilisées et les résultats suivant les capteurs :
# 
# Le seule paramètre "forcé" c'est le k=5 pour la méthode knn.
# Pour le torse on obtient: 
# 86.77% de bonne classification (accuracy) en utilisant la méthode knn
# 85.84% de bonne classification (accuracy) en utilisant la méthode de classification Bayesienne avec GaussianNB
# Je pense que ces résultats sont assez bons peut-être parce que les donnés du torse ne sont pas très dispersées.   
# Pour la cheville on obtient: 
# 74.94% de bonne classification (accuracy) en utilisant la méthode knn
# 67.36% de bonne classification (accuracy) en utilisant la méthode de classification Bayesienne avec GaussianNB
# Pour la cheville on voit des résultats plus bas on pourrait l'expliquer en dusant que les donnés de la cheville 
# sont plus disperses et du coup plus difficiles à classifier correctement
# Les différences entre torse et cheville peuvent aussi s'expliquer car certaines activités sont mieux identifiés
# sur une capteur que l'autre. Dans ce cas TORSE les identifie mieux.
# Le recall nous aide aussi à savoir quelles classes ont été le mieux prédites.

## ====================================================================================================================================================== ##

# # Partie II : Classification des activités avec prétraitement
# 
# - Réalisez maintenant une réduction de dimension par ACP. Expliquez quelle est la dimension la + intéressante.
# - Appliquez les mêmes méthodes supervisées réalisées en TP et commentez les paramètres choisis. 
# - Commentez les résultats. 
# 
# ## Réduction de dimension par Analyse en Composantes Principales (ACP)
# 
# Réaliser une ACP pour réduire la dimension et visualiser les données. Discuter sur le choix de la dimension.

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

print("\n\n--- Partie II : Prétraitement (StandardScaler + PCA) ---")

# Ont doit standariser les données avant de réaliser le ACP
scaler_torse = StandardScaler()
Xtrain_Torse_scaled = scaler_torse.fit_transform(Xtrain_Torse)
Xtest_Torse_scaled = scaler_torse.transform(Xtest_Torse)

scaler_cheville = StandardScaler()
Xtrain_Cheville_scaled = scaler_cheville.fit_transform(Xtrain_Cheville)
Xtest_Cheville_scaled = scaler_cheville.transform(Xtest_Cheville)

# Déterminer le nombre de composantes pour expliquer 95% de la variance
pca_temp = PCA()
pca_temp.fit(Xtrain_Torse_scaled)
cumvar_torse = np.cumsum(pca_temp.explained_variance_ratio_)
n_comp_torse_95 = np.searchsorted(cumvar_torse, 0.95) + 1
n_comp_torse = min(n_comp_torse_95, 50) #on limite à 50

pca_temp.fit(Xtrain_Cheville_scaled)
cumvar_cheville = np.cumsum(pca_temp.explained_variance_ratio_)
n_comp_cheville_95 = np.searchsorted(cumvar_cheville, 0.95) + 1
n_comp_cheville = min(n_comp_cheville_95, 50)

# Le fait de choisir 50 composantes (au lieu de 88 ou 89) revient à garder uniquement les composantes les plus informatives,
# retirer beaucoup de bruit et rendre les distances euclidiennes plus fiables pour les méthodes comme k-nn.

print("\nTorse : composantes pour 95% variance (avant cap):", n_comp_torse_95)
print("Torse : choix retenu (cap 50):", n_comp_torse)
print("Cheville : composantes pour 95% variance (avant cap):", n_comp_cheville_95)
print("Cheville : choix retenu (cap 50):", n_comp_cheville)

# On a composantes nécessaires pour conserver 95% de la variance au torse, cela signifie que
# les données du torse sont très dispersées dans l’espace, il existe beaucoup de dimensions qui contiennent un peu 
# d’information mais aussi du bruitet du coup les signaux sont moins structurés ce qui rend la classification plus difficile.

# ## Premier capteur : méthodes supervisées avec réduction de dimension par ACP sur l'ensemble d'apprentissage
# Après avoir réalisé une ACP sur l'ensemble d'apprentissage pour réduire la dimension 
# puis projeter les données de l'ensemble de test dans la base de l'ACP, appliquez les deux méthodes supervisées.
# 
# 

print("\n--- TORSE : PCA + classification ---")
pca_torse = PCA(n_components=n_comp_torse)
Xtrain_Torse_pca = pca_torse.fit_transform(Xtrain_Torse_scaled)
Xtest_Torse_pca = pca_torse.transform(Xtest_Torse_scaled)

print("Shape après PCA (TORSE) - train:", Xtrain_Torse_pca.shape, " test:", Xtest_Torse_pca.shape)

# Classifieurs simples sur données réduites
knn_pca = KNeighborsClassifier(n_neighbors=k)
nb_pca  = GaussianNB()

# k-NN sur données avec ACP (TORSE)
knn_pca.fit(Xtrain_Torse_pca, y_train)
y_pred_knn_torse_pca = knn_pca.predict(Xtest_Torse_pca)
cm_knn_torse_pca = confusion_matrix(y_test, y_pred_knn_torse_pca)
acc_knn_torse_pca = accuracy_score(y_test, y_pred_knn_torse_pca)

print("k-NN (Torse + PCA) - Matrice de confusion:")
print(cm_knn_torse_pca)
print("k-NN (Torse + PCA) - Pourcentage de bonne classification: {:.2f}%".format(100*acc_knn_torse_pca))
print(classification_report(y_test, y_pred_knn_torse_pca, digits=4))

# GaussianNB sur données avec ACP (TORSE)
nb_pca.fit(Xtrain_Torse_pca, y_train)
y_pred_nb_torse_pca = nb_pca.predict(Xtest_Torse_pca)
cm_nb_torse_pca = confusion_matrix(y_test, y_pred_nb_torse_pca)
acc_nb_torse_pca = accuracy_score(y_test, y_pred_nb_torse_pca)

print("GaussianNB (Torse + PCA) - Matrice de confusion:")
print(cm_nb_torse_pca)
print("GaussianNB (Torse + PCA) - Pourcentage de bonne classification: {:.2f}%".format(100*acc_nb_torse_pca))
print(classification_report(y_test, y_pred_nb_torse_pca, digits=4))


# Calcul de la matrice de confusion et du pourcentage de bonne classification

# L'affichage de la matrice de confusion et du pourcentage de bonne classification est fait ci-dessus.

## ====================================================================================================================================================== ##

# ## Deuxième capteur :  méthodes supervisées avec réduction de dimension par ACP sur l'ensemble d'apprentissage
# Après avoir réalisé une ACP sur l'ensemble d'apprentissage pour réduire la dimension puis projeter les données de l'ensemble de test dans la base de l'ACP, appliquez les deux méthodes supervisées. 
# 

print("\n--- CHEVILLE : PCA + classification ---")
pca_cheville = PCA(n_components=n_comp_cheville)
Xtrain_Cheville_pca = pca_cheville.fit_transform(Xtrain_Cheville_scaled)
Xtest_Cheville_pca = pca_cheville.transform(Xtest_Cheville_scaled)

print("Shape après PCA (Cheville) - train:", Xtrain_Cheville_pca.shape, " test:", Xtest_Cheville_pca.shape)

# Classifieurs simples sur données réduites
knn_pca_c = KNeighborsClassifier(n_neighbors=k)
nb_pca_c  = GaussianNB()

# k-NN
knn_pca_c.fit(Xtrain_Cheville_pca, y_train)
y_pred_knn_cheville_pca = knn_pca_c.predict(Xtest_Cheville_pca)
cm_knn_cheville_pca = confusion_matrix(y_test, y_pred_knn_cheville_pca)
acc_knn_cheville_pca = accuracy_score(y_test, y_pred_knn_cheville_pca)

print("k-NN (Cheville + PCA) - Matrice de confusion:")
print(cm_knn_cheville_pca)
print("k-NN (Cheville + PCA) - Pourcentage de bonne classification: {:.2f}%".format(100*acc_knn_cheville_pca))
print(classification_report(y_test, y_pred_knn_cheville_pca, digits=4))

# GaussianNB
nb_pca_c.fit(Xtrain_Cheville_pca, y_train)
y_pred_nb_cheville_pca = nb_pca_c.predict(Xtest_Cheville_pca)
cm_nb_cheville_pca = confusion_matrix(y_test, y_pred_nb_cheville_pca)
acc_nb_cheville_pca = accuracy_score(y_test, y_pred_nb_cheville_pca)

print("GaussianNB (Cheville + PCA) - Matrice de confusion:")
print(cm_nb_cheville_pca)
print("GaussianNB (Cheville + PCA) - Pourcentage de bonne classification: {:.2f}%".format(100*acc_nb_cheville_pca))
print(classification_report(y_test, y_pred_nb_cheville_pca, digits=4))

# Calcul de la matrice de confusion et du pourcentage de bonne classification

# L'affichage de la matrice de confusion et du pourcentage de bonne classification est fait ci-dessus.

# ## Commentaires sur les méthodes supervisées utilisées avec ACP et les résultats suivant les capteurs :
#
# Pour le torse on obtient: 
# 89.60% de bonne classification (accuracy) en utilisant la méthode knn
# 84.80% de bonne classification (accuracy) en utilisant la méthode de classification Bayesienne avec GaussianNB
# Avec l'ACP implémenté on voit que les résultats restent bons, en théorie si avant on avait un bon résultat,
# il n'y a pas de raison qu'avec l'ACP celui-ci diminue.
# Pour la cheville on obtient: 
# 81.21% de bonne classification (accuracy) en utilisant la méthode knn
# 71.31% de bonne classification (accuracy) en utilisant la méthode de classification Bayesienne avec GaussianNB
# Pour la cheville on voit quand même que les résultats se sont améliorés, pour ce capteur spécifique l'implémentation
# avec ACP semble convenable.
# Les différences entre le torse et la cheville sont devenues plus petites (pour GaussianNB + ACP il reste un écart
# considérable). A nouveau les différences peuvent s'expliquer car certaines activités sont mieux identifiés
# sur une capteur que l'autre. Dans ce cas TORSE les identifie mieux.
 
# # Bilan de l'analyse :# Bilan de l'analyse : réaliser votre propre analyse sur les données et les différentes  méthodes utilisées.
# 
# Dans tous les capteurs les méthodes avec knn semblent les plus précises avec ou sans ACP. ACP est particulièrement utile pour la cheville
# Du coup k-nn est le meilleur modèle car les données sont corrélées et bien structurées localement.
# GaussianNB est inadapté car les signaux ne suivent pas une gaussienne et les features ne sont pas indépendantes.
# Le capteur du torse est nettement meilleur que la cheville (89.60% vs 81.21%) car les signaux sont plus propres et mieux séparés entre classes.
# La PCA améliore fortement la cheville, en supprimant du bruit.
# Sur le torse, la PCA apporte peu car les données sont déjà bien discriminantes.
# Meilleur résultat général : TORSE + PCA + K-NN (89.60%).



# Commande utilisé : python Projet2_CSAA_TDA2_Part1.py

## ====================================================================================================================================================== ##


