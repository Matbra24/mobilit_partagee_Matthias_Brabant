import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt


# === 1. Chargement des données ===
scores = pd.read_csv("C:/Users/braba/Documents/UCL/TFE/GIS_V2/csv/scores_GIS.csv")
poids = pd.read_csv("C:/Users/braba/Documents/UCL/TFE/GIS_V2/csv/poids_pert_modifie.csv")

# === 2. Fonction de tirage Beta-PERT ===
def beta_pert_sample(min_val, mode, max_val, lamb=4, size=10000):
    """
    Tire des échantillons d'une distribution Beta-PERT.
    """
    alpha = 1 + lamb * (mode - min_val) / (max_val - min_val)
    beta_ = 1 + lamb * (max_val - mode) / (max_val - min_val)
    return np.random.beta(alpha, beta_, size) * (max_val - min_val) + min_val

# === 3. Génération des poids simulés ===
num_simulations = 10000
poids_simules = {}

for _, row in poids.iterrows():
    crit = row['critere']
    poids_simules[crit] = beta_pert_sample(row['min'], row['mode'], row['max'], size=num_simulations)

poids_df = pd.DataFrame(poids_simules)
poids_df = poids_df.div(poids_df.sum(axis=1), axis=0)  # Normalisation ligne par ligne

# === 4. Calcul des scores pour chaque simulation ===
all_scores = []
first_place_counts = np.zeros(len(scores))

for i in range(num_simulations):
    w = poids_df.iloc[i]  # vecteur de poids simulé
    score_vector = np.zeros(len(scores))

    for crit in w.index:
        score_vector += scores[crit].values * w[crit]  # score global pondéré

    all_scores.append(score_vector)
    first_place_counts[np.argmax(score_vector)] += 1  # index du max
    



# === 5. Agrégation des résultats ===
all_scores_array = np.array(all_scores).T
scores['score_moyen'] = all_scores_array.mean(axis=1)
scores['score_ecart_type'] = all_scores_array.std(axis=1)
scores['classements_1ers'] = first_place_counts
scores['frequence_1ere_place'] = first_place_counts / num_simulations
    
# Ajouter refnis si dispo dans le CSV original
# Ajouter les refnis correctement
scores['refnis'] = scores['refnis'] if 'refnis' in scores.columns else scores.index

# Créer DataFrame avec les simulations
scores_full_sim = pd.DataFrame(all_scores_array, index=scores['refnis'].values)  # shape: (n_communes, n_simulations)
scores_full_sim.reset_index(inplace=True)
scores_full_sim = scores_full_sim.rename(columns={'index': 'refnis'})
scores_full_sim.columns = ['refnis'] + [f'sim_{i}' for i in range(num_simulations)]

# Export CSV avec refnis
scores_full_sim.to_csv("scores_GIS_simulations.csv", index=False)

# === 6. Statistiques sur les poids des critères ===
poids_stats = pd.DataFrame({
    "poids_moyen": poids_df.mean(),
    "poids_ecart_type": poids_df.std()
})

# === 7. Export des résultats ===
scores.to_csv("scores_GIS_montecarlo_modifie.csv", index=False)
poids_stats.to_csv("poids_stats_modifie.csv")

print("✔ Résultats exportés dans 'scores_GIS_montecarlo.csv' et 'poids_stats.csv'")



# === Extraire uniquement les colonnes pertinentes ===
critères = poids['critere'].tolist()

# S'assurer que les colonnes existent
cols_a_corriger = [c for c in critères if c in scores.columns]
corr_data = scores[cols_a_corriger + ['score_moyen']].copy()

# === Calcul de la matrice de corrélation de Pearson ===
corr_matrix = corr_data.corr()

# === Affichage sous forme de heatmap ===
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, fmt=".2f")
plt.title("Corrélation entre chaque critère et le score moyen final")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.show()

