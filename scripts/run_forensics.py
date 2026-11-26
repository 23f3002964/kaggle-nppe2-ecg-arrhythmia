#!/usr/bin/env python3
"""
NPPE2 Competition - Forensics/Analysis Scripts
Run locally for post-competition analysis.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RECOVERED_DIR = Path("recovered_probs")

# ============================================================
# FX7A: Probability-Space Forensics (Q1, Q2, Q3, Q10, Q12)
# ============================================================
def run_fx7a():
    from sklearn.metrics import f1_score, precision_recall_fscore_support
    import numpy as np
    import pandas as pd
    
    print("="*74)
    print("FX7a — Probability-Space Forensics")
    print("="*74)
    
    # Load artifacts
    A = np.load("recovered_probs/probs_expA_r1pseudo.npy")
    Bt = np.load("out_s1b/probs_truealign_B.npy")
    V = np.load("out_s4/probs_v3full.npy")
    W = np.array([1., .25, .15, 2.65])
    S5a = .10*V + .225*A + .675*Bt
    pred5 = (S5a * W).argmax(1)
    
    # Q1: Disagreement analysis
    am, bm, vm = A.argmax(1), Bt.argmax(1), V.argmax(1)
    groups = {
        "UNANIMOUS": (A==Bt)&(Bt==V),
        "A=B!=V": (A==Bt)&(V!=A),
        "A=V!=B": (A==V)&(Bt!=A),
        "B=V!=A": (Bt==V)&(A!=V),
        "ALL_DIFF": (A!=Bt)&(Bt!=V)&(A!=V),
    }
    
    print("\n[Q1] Agreement analysis:")
    for name, mask in items:
        if mask.sum() == 0: continue
        print(f"  {name}: {mask.sum():>5d} ({mask.mean():.1%}) | "
              f"c3-share: {(S5a.argmax(1)[mask]==3).mean():.2%}")
    
    # Load OOF for Q2
    NTR = 71748
    oofs = {
        "A_old": np.load("recovered_probs/oof_expA_r1pseudo.npy")[:71748],
        "B_true": np.load("out_s1b/oof_B_probs_truealign_B.npy")[:71748],
        "V3": np.load("out_s4/oof_probs_v3full.npy")[:71748],
    }
    Ens = sum(oofs.values()) / 3
    y_tr = pd.read_csv("data/nppe2_dataset/train.csv", usecols=["label"]).label.values
    pred_ens = Ens.argmax(1)
    correct = pred_ens == y_tr
    margin = Ens.max(1) - np.sort(Ens, axis=1)[:, -2]
    
    # Margin buckets
    for name, mask in [
        ("margin>=0.9", margin >= 0.9),
        ("margin 0.5-0.9", (margin>=0.5)&(margin<0.9)),
        ("margin 0.2-0.5", (margin>=0.2)&(margin<0.5)),
        ("margin<0.2", margin<0.2),
    ]:
        m = mask.sum()
        if m == 0: continue
        err = 1 - (np.array([1 if pred_ens[i]==y_tr[i] else 0 for i in np.where(mask)[0]]).mean())
        f1 = f1_score(y_tr[mask], pred_ens[mask], average='macro')
        print(f"  {name:22s}: n={m:>6d} err={1-correct[msk].mean():.2%} F1={f1:.4f}")

if __name__ == "__main__":
    run_fx7a()