import numpy as np
import matplotlib.pyplot as plt

sim_values = np.linspace(0, 6, 200)

# Scénario 0 : juste une base illustrative (0 ou 1 selon seuil 2.5)
def scenario0(values):
    return values / 5

# Scénario 1 : linéaire de 2.5 à 5 (0 → 1)
def scenario1(values):
    return np.where(
        values <= 3,
        0,
        np.where(
            values <4.6,
            (values - 3) / 2,
            0.8        )
    )

# Scénario 2 : linéaire de 2 à 4 (0 → 0.8), plafonné après
def scenario2(values):
    return np.where(
        values <= 2,
        0,
        np.where(
            values < 4,
            0.4 * (values - 2),
            0.8
        )
    )

# Pondérations pour chaque scénario
w0 = scenario0(sim_values)
w1 = scenario1(sim_values)
w2 = scenario2(sim_values)

# Création de la figure
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 10), sharex=False)

# Affichage
axes[0].plot(sim_values, w0, label="0 → 1 dès 2.5", linewidth=2)
axes[0].set_title("Scénario 1 : Interpolation linéaire de 0 à 1 sur [0, 5]")
axes[0].grid(False)
axes[0].set_ylabel("Participation")
axes[0].set_xlim(0, 5)
axes[0].set_ylim(0, 1.2)
axes[0].set_xticks([0, 1, 2, 3, 4, 5]) 

axes[1].plot(sim_values, w1, label="Linéaire entre 2.5 et 5", linewidth=2)
axes[1].set_title("Scénario 2 : Interpolation linéaire de 0 à 0.8 sur [3, 4.6]")
axes[1].grid(False)
axes[1].set_ylabel("Participation")
axes[1].axvline(3, color='gray', linestyle='--')
axes[1].axvline(4.6, color='gray', linestyle='--')
axes[1].set_xlim(0, 5)
axes[1].set_ylim(0, 1.2)
axes[1].set_xticks([0, 1, 2, 3, 4, 4.6, 5]) 

axes[2].plot(sim_values, w2, label="Linéaire entre 2 et 4, max 0.8", linewidth=2)
axes[2].set_title("Scénario 3 : Interpolation linéaire de 0 à 0.8 sur [2, 4]")
axes[2].grid(False)
axes[2].set_ylabel("Participation")
axes[2].set_xlabel("Score moyen")
axes[2].set_xticks([0, 1, 2, 3, 4, 5]) 
axes[2].set_xlim(0, 5)
axes[2].set_ylim(0, 1.2)
axes[2].axvline(2, color='gray', linestyle='--')
axes[2].axvline(4, color='gray', linestyle='--')

plt.savefig("mon_graphique.png", dpi=300)
plt.tight_layout()
plt.show()
