# -*- coding: utf-8 -*-
"""
Created on Fri May 23 09:51:14 2025

@author: braba
"""


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np
import shap

# === Paramètres ===
scores_path = "C:/Users/braba/Documents/UCL/TFE/GIS_V2/csv/scores_GIS_montecarlo.csv"
poids_path = "C:/Users/braba/Documents/UCL/TFE/GIS_V2/csv/poids_pert.csv"

# === Chargement des données ===
df = pd.read_csv(scores_path)
poids = pd.read_csv(poids_path)
criteres = poids['critere'].tolist()

X = df[criteres]
y = df['score_moyen']

# === Entraînement du modèle Random Forest ===
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Initialisation de l'explainer SHAP pour un modèle de type arbre
explainer = shap.TreeExplainer(model)

# Calcul des valeurs SHAP
shap_values = explainer.shap_values(X)
# === Dictionnaire pour renommer les critères ===
renaming_dict = {
    'score_dens': 'Densité',
    'nb_poi_sco': 'Points intérêt',
    'score_bus_score_bus': 'Bus',
    'Distance T': 'Train',
    'score_imat_score_imat': 'Nombre immatriculation',
    'score_tram': 'Tram',
    'score_pent': 'Relief'
}

# === Calcul de l’importance moyenne absolue des SHAP values ===
shap_values_abs_mean = np.abs(shap_values).mean(axis=0)

# === Renommage des critères ===
renamed_criteres = [renaming_dict.get(c, c) for c in criteres]

# === Création du DataFrame trié par importance décroissante ===
shap_importance_df = pd.DataFrame({
    'critere': renamed_criteres,
    'importance_moyenne': shap_values_abs_mean
}).sort_values(by='importance_moyenne', ascending=False)

# === Tracé du graphique barres horizontales ===
plt.figure(figsize=(8, 6))
plt.barh(shap_importance_df['critere'], shap_importance_df['importance_moyenne'])
plt.xlabel("Importance moyenne SHAP value")
plt.title("Importance des critères selon SHAP")
plt.gca().invert_yaxis()  # Pour avoir le plus important en haut
plt.tight_layout()
plt.savefig("shap_bar_plot.png")
plt.show()

"""# Exemple : effet détaillé d’un critère spécifique
# Remplace 'score_dens' par tout critère que tu veux explorer
plt.figure()
shap.dependence_plot("score_dens", shap_values, X, feature_names=criteres, show=False)
plt.tight_layout()
plt.savefig("shap_dependence_score_dens.png")
plt.close()"""

"""# === Importance des variables ===
importances = model.feature_importances_
sorted_idx = np.argsort(importances)[::-1]
sorted_criteres = np.array(criteres)[sorted_idx]
sorted_importances = importances[sorted_idx]

# Dictionnaire pour renommer les critères (à personnaliser)
renaming_dict = {
    'score_dens': 'Densiité',
    'nb_poi_sco': 'Points intérêt',
    'score_bus_score_bus': 'Bus',
    'Distance T': 'Train',
    'score_imat_score_imat': 'Nombre immatriculation',
    'score_tram': 'Tram',
    'score_pent': 'Relief'
    # ...
}

# Renommer les critères pour l'affichage
renamed_sorted_criteres = [renaming_dict.get(c, c) for c in sorted_criteres]


# === Sauvegarde CSV ===
importance_df = pd.DataFrame({
    'critere': renamed_sorted_criteres,
    'importance': sorted_importances
})
importance_df.to_csv("importances_random_forest.csv", index=False)

# === Graphique d'importance ===
plt.figure(figsize=(10, 6))
plt.barh(renamed_sorted_criteres[::-1], sorted_importances[::-1])
plt.xlabel("Importance (Random Forest)")
plt.title("Importance des critères pour prédire le score moyen")
plt.tight_layout()
plt.savefig("importances_random_forest.png")
plt.show()
"""