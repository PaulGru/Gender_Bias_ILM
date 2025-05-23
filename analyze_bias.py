import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# === PARAMÈTRES ===
bias_dir = "bias_outputs"
plot_path = "bias_summary_plot.png"

# === CHARGER TOUS LES FICHIERS CSV ===
records = []
pattern_with_lr = re.compile(r"split(\d+)_p(\d+)_lr(\d+)[eE]?(-?\d*)_(eLM|iLM)\.csv")
pattern_without_lr = re.compile(r"split(\d+)_p(\d+)_(eLM|iLM)\.csv")

for fname in os.listdir(bias_dir):
    match = pattern_with_lr.match(fname)
    if match:
        seed, pval, lr_base, lr_exp, method = match.groups()
        lr_str = f"{lr_base}e{lr_exp}" if lr_exp else lr_base
    else:
        match = pattern_without_lr.match(fname)
        if not match:
            print(f"Ignoré : {fname}")
            continue
        seed, pval, method = match.groups()
        lr_str = "1e-5"  # Valeur par défaut si absente

    try:
        lr = float(lr_str)
        p_env_a = float("0." + pval)
    except ValueError:
        continue

    df = pd.read_csv(os.path.join(bias_dir, fname))
    for val in df["bias"]:
        records.append({
            "seed": int(seed),
            "p_env_a": p_env_a,
            "lr": lr,
            "method": method,
            "bias": val
        })

if not records:
    raise RuntimeError("Aucun fichier de biais valide trouvé. Assure-toi que bias_outputs/ contient des .csv valides.")

# === ANALYSE DES RÉSULTATS ===
df_all = pd.DataFrame(records)
grouped = df_all.groupby(["p_env_a", "lr", "method"])["bias"].agg(["mean", "std"]).reset_index()

# === AFFICHAGE DES RÉSULTATS ===
print("\nBiais moyen et écart-type par p_env_a, learning_rate et méthode :\n")
print(grouped.to_string(index=False))

# === PLOT ===
plt.figure(figsize=(10, 6))
markers = {"eLM": "o", "iLM": "s"}
linestyles = {"eLM": "-", "iLM": "--"}
colors = {
    ("eLM", 1e-5): "tab:blue",
    ("iLM", 1e-5): "tab:orange",
    ("eLM", 5e-5): "tab:green",
    ("iLM", 5e-5): "tab:red"
}

for (lr, method), sub in grouped.groupby(["lr", "method"]):
    color = colors.get((method, lr), None)
    plt.errorbar(
        sub["p_env_a"], sub["mean"], yerr=sub["std"], 
        label=f"{method} (lr={lr:.0e})", capsize=5, 
        marker=markers[method], linestyle=linestyles[method],
        color=color, linewidth=2
    )

plt.title("Biais moyen ± écart-type selon p_env_a et learning rate")
plt.xlabel("p_env_a (proportion non inversée dans Env A)")
plt.ylabel("Biais moyen (1 - entropie normalisée)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(plot_path)
plt.show()
print(f"\n✅ Graphique sauvegardé sous {plot_path}")
