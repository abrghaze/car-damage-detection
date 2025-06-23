import os
import pandas as pd

# Dossier des trainings
base_dir = 'runs/segment'

best_map = 0
best_model = ""

# Parcourir tous les dossiers de training
for folder in os.listdir(base_dir):
    results_path = os.path.join(base_dir, folder, 'results.csv')

    if os.path.exists(results_path):
        df = pd.read_csv(results_path)
        last_epoch = df.iloc[-1]  # Dernière ligne = dernière époque

        map50 = last_epoch['metrics/mAP50(M)']
        map95 = last_epoch['metrics/mAP50-95(M)']
        print(f"{folder}: mAP50(M)={map50:.3f}, mAP50-95(M)={map95:.3f}")

        if map50 > best_map:
            best_map = map50
            best_model = folder

print(f"\n✅ Meilleur modèle: {best_model} avec mAP50(M) = {best_map:.3f}")
