# -*- coding: utf-8 -*-
"""
Created on Tue May 20 12:43:07 2025

@author: braba
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import seaborn as sns
from scipy.stats import pearsonr

# === 1. Chargement des résultats des deux simulations ===
df_base = pd.read_csv("scores_GIS_montecarlo.csv")            # Résultat initial
df_mod  = pd.read_csv("scores_GIS_montecarlo_modifie.csv")    # Résultat avec poids modifiés

# === 2. Fusion des deux jeux de données ===
df_merged = df_base[['refnis', 'score_moyen', 'frequence_1ere_place']].merge(
    df_mod[['refnis', 'score_moyen', 'frequence_1ere_place']],
    on='refnis',
    suffixes=('_base', '_mod')
)

# === 3. Calculs des écarts ===
df_merged['delta_score'] = df_merged['score_moyen_mod'] - df_merged['score_moyen_base']
df_merged['delta_freq'] = df_merged['frequence_1ere_place_mod'] - df_merged['frequence_1ere_place_base']

# === 4. Export pour analyse ou rapport ===
df_merged.to_csv("comparaison_scores.csv", index=False)

# === 5. Visualisations ===
plt.figure(figsize=(8, 6))
plt.scatter(df_merged['score_moyen_base'], df_merged['score_moyen_mod'], alpha=0.7)
plt.plot([df_merged['score_moyen_base'].min(), df_merged['score_moyen_base'].max()],
         [df_merged['score_moyen_base'].min(), df_merged['score_moyen_base'].max()],
         'r--', label='Identité')
plt.xlabel("Score moyen (base)")
plt.ylabel("Score moyen (modifié)")
plt.title("Comparaison des scores moyens\navant/après modification des poids")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("graph_score_comparaison.png")
plt.show()

# Graphique : Changement de fréquence de classement en 1ʳᵉ position
plt.figure(figsize=(8, 6))
plt.hist(df_merged['delta_freq'], bins=30, edgecolor='black')
plt.title("Distribution des variations de fréquence en 1ère place")
plt.xlabel("Delta fréquence (modifié - base)")
plt.ylabel("Nombre de communes")
plt.grid(True)
plt.tight_layout()
plt.savefig("graph_delta_frequence.png")
plt.show()


top_base = df_base.sort_values("frequence_1ere_place", ascending=False).head(10)
top_mod = df_mod[df_mod["refnis"].isin(top_base["refnis"])]

plt.figure(figsize=(10, 6))
bar_width = 0.4
indices = np.arange(len(top_base))
plt.bar(indices, top_base['frequence_1ere_place'], bar_width, label='Base')
plt.bar(indices + bar_width, top_mod['frequence_1ere_place'], bar_width, label='Modifié')

plt.xticks(indices + bar_width / 2, top_base['refnis'], rotation=45)
plt.ylabel("Fréquence 1ʳᵉ place")
plt.title("Comparaison des communes les plus dominantes")
plt.legend()
plt.tight_layout()
plt.show()

