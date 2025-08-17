# -*- coding: utf-8 -*-
"""
Created on Thu May 22 17:52:36 2025

Calcul de la demande en pass-km de la voiture partagée avec incertitude

@author: braba
"""

import pandas as pd
import numpy as np

# === 1. Charger les fichiers CSV ===
imat_df = pd.read_csv('C:/Users/braba/Documents/UCL/TFE/GIS_V2/csv/calcul_km_energyscope.csv', dtype={'refnis': str})
scores_sim = pd.read_csv('C:/Users/braba/Documents/UCL/TFE/GIS_V2/csv/scores_GIS_simulations.csv', dtype={'refnis': str})

# === 2. Fusion des données sur 'refnis' ===
merged_df = pd.merge(imat_df, scores_sim, on='refnis')

# === 3. Extraction des colonnes de simulation ===
sim_cols = [col for col in merged_df.columns if col.startswith("sim_")]
sim_values = merged_df[sim_cols].values  # (n_communes, n_simulations)

# === 4. Calcul de la participation (règle par morceaux) ===

participation = sim_values / 5   #100%
# 0 si <2.5, linéaire ensuite jusqu’à 100% à 5
#participation = np.where(sim_values < 3, 0, (sim_values - 3) / 2)
"""participation = np.where(
    sim_values < 2,
    0,
    np.where(
        sim_values < 4,
        0.4 * (sim_values - 2),  # linearly goes from 0 to 0.8
        0.8
    )
)"""
"""participation = np.where(sim_values < 2.5,0,np.where(
        sim_values < 4.5,
        (sim_values - 2.5) / 2.5,  # linearly goes from 0 to 0.8
        0.8
    ))"""

"""participation = np.where(
    sim_values < 2,
    0,
    np.where(
        sim_values < 4,
        0.4 * (sim_values - 2),  # linearly goes from 0 to 0.8
        0.8
    )
)"""

# === Calcul du taux de remplacement en fonction du score ===
replacement_rate = np.clip((sim_values-1) / 4 * (16 - 3) + 3, 3, 16)
print(replacement_rate)

immat = merged_df['imat_communes_IMAT'].values[:, np.newaxis]
# === Nombre de voitures individuelles remplacées ===
veh_replaced = participation * immat

# === Nombre de voitures partagées nécessaires ===
veh_shared = veh_replaced / replacement_rate

km_per_day = merged_df['km/j/pers'].values[:, np.newaxis]
pass_km_report =  (52*4.4 /(km_per_day * 365))*participation

pass_km = veh_replaced * km_per_day * 365 - pass_km_report

# === 6. Demande nationale par simulation ===
total_demand_per_sim = pass_km.sum(axis=0) / 1e6  # millions de pass-km

#total_dmd_report_per_sim = pass_km_report.sum(axis = 0) / 1e6

# === 7. Statistiques globales ===
mean_demand = total_demand_per_sim.mean()
#mean_report = total_dmd_report_per_sim.mean()
mean_report = pass_km_report.mean()

ci_low, ci_high = np.percentile(total_demand_per_sim, [2.5, 97.5])

# === 8. Affichage des résultats ===
print("🚗 Demande moyenne totale en pass-km (Mkm/an) :", round(mean_demand))
print("📊 Intervalle de confiance à 95% : [{:.0f}, {:.0f}] Mkm/an".format(ci_low, ci_high))
print ("part partagée" , mean_demand/126330)
print("Report modal vers le public" , mean_report )


# === 9. Export CSV (facultatif) ===
"""pd.DataFrame({
    'simulation': list(range(len(total_demand_per_sim))),
    'pass_km_total_Mkm': total_demand_per_sim
}).to_csv('resultats_demande_passkm_50.csv', index=False)"""