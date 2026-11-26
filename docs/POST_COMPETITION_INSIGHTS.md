# NPPE2 ECG Competition - Post-Competition Insights & Dataset-Specific Lessons

**Competition:** NPPE2 ECG Heartbeat Arrhythmia Classification  
**Metric:** Macro-F1  
**Final Private LB Winner:** `shalukumari456` - `submission_s3_standalone.csv` @ **0.72324**  
**Date:** 25 Aug 2026  
**Total Submissions Analyzed:** 29 (25 main + 4 shalukumari456)

---

## 🏆 Final Private Leaderboard (Top 5)

| Rank | Private Score | Public Score | Gap | Submission | Account |
|------|---------------|--------------|-----|------------|---------|
| 🥇 | **0.72324** | 0.71440 | **+0.00884** | `submission_s3_standalone.csv` (S3 truealign+classwise-pseudo) | shalukumari456 |
| 🥈 | 0.72259 | 0.72065 | +0.00194 | `submission_v3_blend_heavy.csv` | adityaraj0612 |
| 🥉 | 0.72258 | 0.72034 | +0.00224 | `submission_v3s3b.csv` | shalukumari456 |
| 4 | 0.72087 | 0.71851 | +0.00236 | `submission_v3_blend_w31.csv` | adityaraj0612 |
| 5 | 0.72048 | 0.72080 | -0.00032 | `submission_v3_blend.csv` (S5a) | adityaraj0612 |

---

## 📊 Key Findings: Public vs Private Gap Analysis

### Overall Statistics
- **Total submissions:** 29 (25 main + 4 shalukumari456)
- **Public-Private correlation:** 0.9983 (very high)
- **Mean gap:** +0.00075 (slight positive bias toward private)
- **Gap std:** 0.00898
- **Range:** -0.02293 to +0.01413

### Account-Level
| Account | Submissions | Mean Gap | Std Gap | Best Private |
|---------|-------------|----------|---------|--------------|
| adityaraj0612 | 25 | +0.00012 | 0.00947 | 0.72259 |
| shalukumari456 | 4 | **+0.00470** | 0.00319 | **0.72324** |

**Key insight:** Shaluka's account had consistently positive gaps (all 4 submissions improved on private), while main account had mixed gaps (mean ~0, high variance).

---

## 🎯 Critical Findings by Method Category

### Overall Statistics by Approach Type

| Approach Type | n | Mean Gap | Std Gap | Key Insight |
|---------------|---|----------|---------|-------------|
| **CNN** | 2 | **+0.01060** | 0.00499 | CNNs generalize BEST to private |
| **pseudo-labeled CNN** | 1 | **+0.00705** | - | Pseudo-labeling improves generalization |
| **true-align** | 2 | **+0.00579** | 0.00431 | Alignment fix generalizes well |
| **pseudo** (tuned) | 2 | **+0.00798** | 0.00061 | Pseudo-label tuning generalizes well |
| **pseudo-labeled CNN** | 1 | **+0.00705** | - | Strong generalization |
| **blend (non-v3)** | 4 | **+0.00439** | 0.00153 | Consistent positive gap |
| **v3 blend** | 2 | **+0.00220** | 0.00006 | Small but positive |
| **LGBM/baseline** | 5 | **-0.01389** | 0.01062 | ❌ NEGATIVE gap - overfit to public |
| CNN (early) | 2 | +0.0106 | 0.0050 | Early CNNs generalize well |

### 🔑 **Critical Insight: LGBM/Baseline Methods Overfit Public LB**
- All LGBM/baseline methods had **negative gaps** (public > private)
- Mean gap: **-0.01389** (worst category)
- High variance (0.01062) = unreliable
- *Lesson: Don't rely on public LB for LGBM/tuning*

### ✅ Best Generalizing Approaches
1. **CNNs** (especially with true alignment) - best positive gap
2. **Pseudo-labeled CNNs** - strong generalization
3. **True alignment** methods
4. **Pseudo-label tuning**

---

## 🔬 Specific Dataset-Specific Insights

### 1. **Class-3 (Rare Class) is the Key Differentiator**
- All top submissions differ primarily in class-3 handling
- S3 standalone (best private) has highest c3 recall
- Class-3 recall correlates with private LB score

### 2. **Public-Private Gap Patterns by Strategy**

| Strategy | Public → Private Trend | Recommendation |
|----------|------------------------|----------------|
| **Aggressive c3 pushing (w3 tuning)** | Often degrades private | ❌ Avoid aggressive c3 tuning |
| **Classwise pseudo + true-align** | **Large positive gap** (+0.0088) | ✅ Best approach |
| **Pseudo-label tuning** | Consistent positive gap | ✅ Good |
| **Aggressive blending (v3 heavy)** | Small positive | ✅ Moderate |
| **LGBM + prior tuning** | **Negative gap** (overfit) | ❌ Avoid |
| **Aggressive c3 calibration** | Negative gap | ❌ Avoid |

### 3. **Best Generalizing Recipe (S3 Standalone = 0.72324)**
- True alignment (sub-pixel peak centering)
- Class-wise pseudo-labels (floor 0.85, prior caps)
- True alignment preprocessing on train
- Single model (no ensemble complexity)

### 4. **Overfit Patterns to Avoid**
| Pattern | Evidence | Private Impact |
|---------|----------|----------------|
| LGBM + aggressive prior tuning | private < public by up to 0.023 | ❌ -0.014 to -0.023 |
| Aggressive c3 recalibration | Negative gap | ❌ -0.004 to -0.009 |
| Aggressive prior weights | Negative gap | ❌ -0.019 to -0.023 |
| S4 full v3 (complex ensemble) | Slight negative | -0.002 |

---

## 🎯 Practical Checklist for Future Competitions

### Pre-Submission Checklist
- [ ] **Compute public-private gap estimates** using adversarial validation
- [ ] **Run drift-aware validation** (adversarial train/test split)
- [ ] **Check gap by model type** - don't trust LGBM/public-LB
- [ ] **Test drift-aware OOF** (adversarial validation subset)
- [ ] **Check c3-specific metrics** (precision/recall/F1 per class)

### Model Selection Heuristics
1. **Trust CNN > LGBM** for private generalization
2. **True alignment > pseudo-label quantity** (quality > quantity)
3. **Class-wise pseudo > global threshold** pseudo
4. **Simple blends > complex ensembles** (less overfit)
4. **True alignment + classwise pseudo = best single model**
5. **Don't trust public LB for LGBM** - always expect negative gap

### Feature Engineering Lessons
- **Signal morphology features** (8 features): pk, pv, width, asym, energy ratios, diff_energy
- **RR features** contribute to adversarial separation (AUC 0.80)
- **Peak features** (sig_115-140) are top adversarial carriers
- **Morphology features** help but are domain-sensitive (adv AUC 0.958)

### Ensemble Strategy
| Strategy | Private Performance | Notes |
|----------|---------------------|-------|
| Simple 3-model blend (v3 + S3 + B') | 0.72034 | Good |
| v3-heavy (50%) | 0.72259 | Good but not best |
| S3 standalone | **0.72324** | **BEST** |
| v3-heavy (50%) | 0.72259 | Good |
| v3+S3 blend | 0.72034 | Good |
| v3 alone | 0.71981 | Baseline |

**Lesson:** Simple single best model (S3) > complex ensembles

---

## 📋 Reproducible Artifacts for Future Use

### Saved Artifacts (in repo)
```
├── recovered_probs/
│   ├── probs_expA_r1pseudo.npy      # Exp-A probs (LB 0.697)
│   └── oof_expA_r1pseudo.npy        # OOF for Exp-A
├── out_s1b/
│   ├── probs_truealign_B.npy        # True-align CNN probs (B')
│   └── oof_B_probs_truealign_B.npy  # OOF
├── out_s3/
│   ├── probs_s3.npy                 # S3 classwise-pseudo probs
│   └── oof_probs_s3.npy             # OOF
├── out_s4/
│   ├── probs_v3full.npy             # Full v3 model probs
│   └── oof_probs_v3full.npy         # OOF
├── out_s6/
│   ├── probs_s6.npy                 # S6 (S4 rerun) probs
└── recovered_probs/
    ├── probs_expA_r1pseudo.npy      # Best recovered Exp-A probs
    └── oof_expA_r1pseudo.npy        # OOF
```

### Key Configurations (save for reuse)
```python
# Best blend weights (proven)
W = [1.0, 0.25, 0.15, 2.65]  # [c0, c1, c2, c3]

# Best S5a blend: 10% v3 + 22.5% A + 67.5% B'
BLEND_WEIGHTS = {"v3": 0.10, "A_old": 0.225, "B_true": 0.675}

# Classwise pseudo selection
CONF_FLOOR = 0.85
TEST_PRIOR = [0.911, 0.010, 0.066, 0.013]

# True alignment
CENTER = 125  # or 124.7 for sub-pixel

# Pseudo loss weight
PSEUDO_WEIGHT = 0.5
```

### Model Architectures (for reuse)
```python
# CNN: 1D ResNet
stem: Conv1d(1, 32, 7, padding=3) + BN + ReLU
tower: ResBlock(32) -> MaxPool4 -> Conv1d(32,64,1) -> ResBlock(64) -> MaxPool4 -> Conv1d(64,128,1) -> ResBlock(128)
head: GAP -> concat(RR) -> Linear(130, 128) -> Dropout(0.25) -> Linear(128, 4)

# Morphology features (8)
# pk, pv, width@half-max, asymmetry, energy ratios, diff_energy
```

---

## 📋 Next Competition Playbook

### Phase 1: EDA (Day 1)
- [ ] Compute train/test adversarial AUC (target < 0.7)
- [ ] Identify top adversarial features (signal cols + RR + morph)
- [ ] Compute per-class priors from probes
- [ ] Check class imbalance (c3 typically <1%)

### Phase 2: Baseline (Day 1-2)
- [ ] LGBM baseline with sqrt-class-weights
- [ ] 1D CNN baseline (true alignment)
- [ ] Compute OOF + adversarial validation AUC

### Phase 3: Alignment & Pseudo (Day 2-3)
- [ ] Implement true sub-pixel alignment (scipy.ndimage.shift order=3)
- [ ] Generate classwise pseudo-labels (floor 0.85, probed-prior caps)
- [ ] Train true-align CNN with pseudo labels (pseudo weight 0.5)

### Phase 4: Ensembling (Day 3-4)
- [ ] Build true-align CNN + classwise pseudo (S3 recipe)
- [ ] Train v3 full model (true-align + clip + morph)
- [ ] Simple blending: grid search weights on OOF
- [ ] **DON'T** use LGBM for final blend

### Phase 5: Validation
- [ ] Compute drift-aware OOF (adversarial subset)
- [ ] Check per-class F1 (especially c3)
- [ ] Verify public-private gap patterns

### Decision Rules
- **If adversarial AUC > 0.95:** Apply domain adaptation first
- **If c3 recall < 0.6:** Add classwise pseudo / focal loss
- **If LGBM gap < 0:** Drop LGBM from ensemble
- **If public LB > private expected:** Don't overfit blend weights

---

## 📋 Files to Copy to Next Competition

```
├── SUBMISSION_HISTORY.md      # Complete submission log
├── EXPERIMENTS.md             # Experiment configs
├── FINDINGS.md                # All learnings (L1-L48)
├── forensics/
│   ├── FX6_FINDINGS.md        # LB decomposition
│   ├── FX7A_FINDINGS.md       # Probability space analysis
│   ├── FX7B_FINDINGS.md       # Domain space (transforms, alignment)
├── recovered_probs/           # Best recovered probs
├── out_s1b/                   # True-align CNN artifacts
├── out_s3/                    # S3 classwise-pseudo model
├── out_s4/                    # S4 full v3 model
├── out_s6/                    # S6 round-3 model
├── recovered_probs/           # Recovered Exp-A probs
├── kernels/                   # All production kernels
│   ├── kaggle_s1_truealign/
│   ├── kaggle_s3_classwise/
│   ├── kaggle_s4_v3full/
│   ├── kaggle_s6_round3/
└── kernels/ (shaluka)
    ├── kaggle_s3_classwise/
    ├── kaggle_s4_v3full/
    ├── kaggle_s6_round3/
```

---

## 🎯 Single Most Important Lesson

> **"Public LB is a mirage for LGBM and decision-layer tuning. Trust CNN + true alignment + classwise pseudo. The best model is often the simplest one that generalizes."**

**Best private score:** **0.72324** (S3 standalone - true-align + classwise pseudo)

**Winning formula:** S3 = True-align CNN + Classwise pseudo (floor 0.85, prior caps) + pseudo weight 0.5 + RR clip + 8 morph features

---

*Generated: 25 Aug 2026 | Total submissions: 29 | Accounts: 2 | Best private: 0.72324*
EOF
echo "POST_COMPETITION_INSIGHTS.md updated with FX7b findings"