"""
Car Damage Detection - Model Deployment
Selects the best trained model and deploys it to production
"""

import shutil
import pandas as pd
from pathlib import Path


# ============== CONFIGURATION ==============
BASE_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = BASE_DIR / "scripts" / "runs" / "segment"
DEPLOY_DIR = BASE_DIR / "models" / "yolo_weights"


def deploy_best_model():
    """Find the best model from training runs and deploy it."""
    
    print("=" * 50)
    print("🏆 MODEL DEPLOYMENT")
    print("=" * 50)
    
    DEPLOY_DIR.mkdir(parents=True, exist_ok=True)
    
    if not RUNS_DIR.exists():
        print(f"❌ No training runs found at: {RUNS_DIR}")
        return
    
    # Find all valid training runs
    candidates = []
    
    for folder in RUNS_DIR.iterdir():
        if not folder.is_dir():
            continue
        
        csv_path = folder / "results.csv"
        weights_path = folder / "weights" / "best.pt"
        
        if csv_path.exists() and weights_path.exists():
            try:
                df = pd.read_csv(csv_path)
                df.columns = [c.strip() for c in df.columns]
                
                # Get best mAP score
                metric = 'metrics/mAP50-95(M)'
                if metric not in df.columns:
                    metric = 'metrics/mAP50-95(B)'
                
                if metric in df.columns:
                    best_score = df[metric].max()
                    candidates.append({
                        "name": folder.name,
                        "score": best_score,
                        "path": weights_path
                    })
                    print(f"   📊 {folder.name}: mAP = {best_score:.4f}")
            except Exception as e:
                print(f"   ⚠️ Error reading {folder.name}: {e}")
    
    if not candidates:
        print("\n❌ No valid models found.")
        return
    
    # Select winner
    winner = max(candidates, key=lambda x: x['score'])
    
    print(f"\n🥇 WINNER: {winner['name']}")
    print(f"   Score: {winner['score']:.4f}")
    
    # Deploy
    dest = DEPLOY_DIR / "best.pt"
    
    if dest.exists():
        backup = DEPLOY_DIR / "best_backup.pt"
        shutil.move(str(dest), str(backup))
        print(f"\n📦 Previous model backed up")
    
    shutil.copy2(str(winner['path']), str(dest))
    print(f"✅ Deployed to: {dest}")


if __name__ == "__main__":
    deploy_best_model()