# -*- coding: utf-8 -*-
"""
Created on Sat May 17 15:50:50 2025

@author: braba
"""

# -*- coding: utf-8 -*-
"""
Analyse des voitures évitées avec incertitude - TFE
Auteur : braba
"""

import pandas as pd
import numpy as np

# === 1. Chargement des fichiers CSV ===
imat_df = pd.read_csv('C:/Users/braba/Documents/UCL/TFE/GIS_V2/csv/calcul_km_energyscope.csv', dtype={'refnis': str})
scores_sim = pd.read_csv('C:/Users/braba/Documents/UCL/TFE/GIS_V2/csv/scores_GIS_simulations.csv', dtype={'refnis': str})


# === 2. Fusion des données sur 'refnis' ===
merged_df = pd.merge(imat_df, scores_sim, on='refnis')

# === 3. Paramètres ===

IMMAT_COL = 'imat_communes_IMAT'

# === 4. Extraction des simulations ===
sim_cols = [col for col in merged_df.columns if col.startswith("sim_")]
sim_values = merged_df[sim_cols].values  # (n_communes, n_simulations)
immat_values = merged_df[IMMAT_COL].values[:, np.newaxis]  # (n_communes, 1)

# === 5. Calcul vectorisé ===
participation = sim_values / 5  #100%

#participation = np.where(sim_values < 3,0,(sim_values - 3) / 2)  #plus de 50%


"""participation = np.where(
    sim_values < 2,
    0,
    np.where(
        sim_values < 4,
        0.4 * (sim_values - 2),  # linearly goes from 0 to 0.8
        0.8
    )
)"""

replacement_rate = np.clip((sim_values-1) / 4 * (16 - 3) + 3, 3, 16)
vehicules_partages = immat_values * participation / replacement_rate  # (communes × simulations)
numerateur = (vehicules_partages * replacement_rate).sum()
denominateur = vehicules_partages.sum()
replacement_rate_pondere = numerateur / denominateur
print(f"🔁 Taux de remplacement moyen pondéré (national) : {replacement_rate_pondere:.2f}")

evitees = immat_values * participation * (1 - 1 / replacement_rate)  # (n_communes, n_simulations)

# === 6. Agrégation nationale ===
results_array = evitees.sum(axis=0)
mean_evitees = results_array.mean()
ci_low, ci_high = np.percentile(results_array, [2.5, 97.5])


# === 7. Résultats par commune ===
evitees_df = pd.DataFrame(evitees, columns=[f"sim_{i}" for i in range(evitees.shape[1])])
evitees_df['refnis'] = merged_df['refnis']
evitees_df['moyenne_evitees'] = evitees.mean(axis=1)
evitees_df['ci_2.5'] = np.percentile(evitees, 2.5, axis=1)
evitees_df['ci_97.5'] = np.percentile(evitees, 97.5, axis=1)

# === 8. Export CSV final ===
#evitees_df[['refnis', 'moyenne_evitees', 'ci_2.5', 'ci_97.5']].to_csv('voitures_evitees_par_commune_50.csv', index=False)

# === 9. Affichage résumé ===
print(f"\n✔ Moyenne nationale de voitures évitées : {mean_evitees:,.0f}")
print(f"✔ Intervalle de confiance 95% : [{ci_low:,.0f}, {ci_high:,.0f}]")
print(f"✔ Résultats exportés dans 'voitures_evitees_par_commune_2_4.csv'")

print(6089564 - mean_evitees)