#!/usr/bin/env python3
"""
FX7 Forensics - Domain Space Analysis (Q5-Q8)
Run locally for post-competition analysis.
"""

import numpy as np
import pandas as pd
import json
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
from scipy.signal import butter, sosfilt
from scipy.ndimage import shift as ndshift

OUT = []
def P(s=""):
    print(s); 
    with open("forensics/FX7B_FINDINGS.md", "a") as f:
        f.write(s + "\n")

# ============================================================
# LOAD DATA
# ============================================================
tr = pd.read_csv("data/nppe2_dataset/train.csv")
te = pd.read_csv("data/nppe2_dataset/test.csv")
sc = [c for c in tr.columns if c.startswith("sig_")]

def align_true(X):
    pk = np.abs(X).argmax(1); shift = 125 - pk
    idx = (np.arange(X.shape[1])[None,:] - shift[:,None]) % X.shape[1]
    return X[np.arange(len(X))[:,None], idx].astype(np.float32)

def morphology(X):
    ax = np.abs(X.astype(np.float32)); n = len(X)
    pk = ax.argmax(1); pv = ax.max(1)
    width = np.zeros(n, np.float32); asym = np.zeros(n, np.float32)
    tot = ax.sum(1) + 1e-8
    for i in range(n):
        p, half = pk[i], pv[i] / 2
        l = p
        while l > 0 and ax[i,l] > half: l -= 1
        r = p
        while r < X.shape[1]-1 and ax[i,r] > half: r += 1
        width[i] = r - l; asym[i] = (p - l) / max(r - l, 1)
    de = np.abs(np.diff(X, axis=1)).mean(1)
    return np.stack([pk, pv, width, asym,
                     ax[:,100:150].sum(1)/tot, ax[:,:100].sum(1)/tot,
                     ax[:,150:].sum(1)/tot, de], 1)

tr_sig = tr[sc].values.astype(np.float32)
te_sig = te[sc].values.astype(np.float32)
tr_sig_al = align_true(tr_sig_raw)
te_sig_al = align_true(te_sig)

# RR + morph
rr = np.log1p(np.nan_to_num(tr[["post_rr","rr_ratio"]].values.astype(np.float32)))
M_tr = morphology(tr_sig_raw); M_te = morphology(te_sig_al)
R_tr, R_te = rr, np.log1p(np.nan_to_num(te[["post_rr","rr_ratio"]].values.astype(np.float32)))

# Subsample signal cols
COLS = [c for c in tr.columns if c.startswith("sig_")][::5]  # 50 cols
S_tr, S_te = tr_sig_al[:, ::5], te_sig[:, ::5]

# Morph features
tr_ex = np.c_[tr_sig_al, M_tr]; te_ex = np.c_[te_sig_al, M_te]
mu, sd = tr_ex.mean(0), tr_ex.std(0) + 1e-8
tr_ex = np.clip((tr_ex - mu)/sd, -5, 5).astype(np.float32)
te_ex = np.clip((te_ex - mu)/sd, -5, 5).astype(np.float32)

# ============================================================
# HELPERS
# ============================================================
def adv_auc(Xtr, Xte, tag, quick=True):
    X = np.r_[Xtr, Xte]; y = np.r_[np.zeros(len(Xtr)), np.ones(len(Xte))]
    xa,xb,ya,yb = train_test_split(X, np.r_[np.zeros(len(Xtr)), np.ones(len(Xte))], 
                                   test_size=.25, random_state=0, stratify=None)
    m = lgb.LGBMClassifier(n_estimators=180, num_leaves=63, learning_rate=.06,
                           subsample=.8, colsample_bytree=.8, verbosity=-1, random_state=0)
    m.fit(Xtr, np.zeros(len(Xtr))); m.fit(Xtr, np.zeros(len(Xtr)))  # dummy
    # proper
    m = lgb.LGBMClassifier(n_estimators=180, num_leaves=63, learning_rate=.06,
                           subsample=.8, colsample_bytree=.8, verbosity=-1, random_state=0)
    m.fit(Xtr, np.zeros(len(Xtr)))
    return roc_auc_score(np.ones(len(Xte)), m.predict_proba(Xte)[:,1])

# ============================================================
# Q5/Q6: Adversarial decomposition
# ============================================================
print("="*74); print("FX7b — DOMAIN-SPACE FORENSICS"); print("="*74)

print("\n[Q5+Q6] Adversarial AUC by representation:")
adv_auc(tr_sig_al[:, ::5], te_sig[:, ::5], "signal-only (50 cols)")
adv_auc(rr_tr, rr_te, "RR-only")
_,im_m,_ = adv_auc(M_tr, M_te, "morphology")
_,im_s,ms = adv_auc(np.c_[tr_sig_al[:,::5], rr_tr, M_tr], 
                     np.c_[te_sig[:,::5], rr_te, M_te], "ALL")

# Feature importances
print("\nMorphology importance:")
for nm, g in zip(["pk","pv","width","asym","e_mid","e_left","e_right","diff_e"], im_m):
    print(f"  {nm:8s} {g}")

# ============================================================
# Q7: Transform response
# ============================================================
def lowpass(X, wn=0.36):
    sos = butter(3, wn, btype='low', output='sos')
    return sosfilt(sos, X, axis=1).astype(np.float32)

te_amp = np.abs(te_sig_raw).max(1)
tr_amp = np.abs(tr_sig_raw).max(1)
scale = np.median(te_amp)/np.median(tr_amp)

transforms = {
    "T0 raw": tr_sig_al,
    "T1 lowpass .36": lowpass(tr_sig_al) if True else None,
    "T2 amp-norm": (tr_sig_al * np.median(te_amp)/np.median(tr_amp)).astype(np.float32),
    "T3 lowpass+amp": lowpass(tr_sig_al) * np.median(te_amp)/np.median(tr_amp) if True else None,
}

print("\n[Q7] Transform responses:")
for tag, Xt in transforms.items():
    if Xt is None: continue
    # adv auc
    auc = adv_auc(Xt[:, ::5], te_sig[:, ::5], tag)
    # class-F1 proxy
    sub = np.random.choice(len(Xt), 12000, replace=False)
    xa,xb,ya,yb = train_test_split(Xt[sub][::2], tr_y[sub][::2], test_size=.25, random_state=0)
    m = lgb.LGBMClassifier(n_estimators=150, num_leaves=63, verbosity=-1, random_state=0)
    m.fit(Xt[sub][::2], y_train[::2])
    cf1 = f1_score(yb, m.predict(xb), average='macro')
    print(f"  {tag:30s} adv-AUC={auc:.4f} | class-F1={cf1:.4f}")

# ============================================================
# Q8: Sub-pixel alignment
# ============================================================
def centroid(X):
    ax = np.abs(X)
    num = (ax[:,110:140]*np.arange(110,140)[None,:]).sum(1)
    den = ax[:,110:140].sum(1)+1e-9
    return num/den

ctest = centroid(te_sig)
print(f"\nTEST centroid: mean={ctest.mean():.3f} std={ctest.std():.3f}")

for tag, Xt in [("integer @125", align_true(tr_sig_raw, 125)),
                ("frac ->124.70", None)]:
    if Xt is None:
        pk = np.abs(tr_sig_raw).argmax(1)
        shift = 124.70 - np.abs(tr_sig_raw).argmax(1)
        i0 = np.floor(124.70 - np.abs(tr_sig_raw).argmax(1)).astype(int) % 250
        fr = (124.70 - np.abs(tr_sig_raw).argmax(1) - np.floor(124.70 - np.abs(tr_sig_raw).argmax(1)))[:, None]
        idx0 = (np.arange(250)[None,:] - i0[:,None]) % 250
        idx1 = (idx0 + 1) % 250
        fr = (124.70 - np.abs(tr_sig_raw).argmax(1) - np.floor(124.70 - np.abs(tr_sig_raw).argmax(1)))[:, None]
        Xt = ((1-fr)*tr_sig_raw[np.arange(len(tr_sig_raw))[:,None], idx0] + 
              fr*tr_sig_raw[np.arange(len(tr_sig_raw))[:,None], (idx0+1)%250]).astype(np.float32)
    c = np.abs(Xt).argmax(1)  # simple peak
    # better: centroid
    ax = np.abs(Xt)
    num = (ax[:,110:140]*np.arange(110,140)[None,:]).sum(1)
    den = ax[:,110:140].sum(1)+1e-9
    cent = num/den
    print(f"  {tag:25s}: mean={cent.mean():.3f} std={cent.std():.3f} "
          f"|frac@124.7±0.5={(np.abs(cent-124.7)<=0.5).mean():.1%}")

print("\nSaved -> forensics/FX7B_FINDINGS.md")