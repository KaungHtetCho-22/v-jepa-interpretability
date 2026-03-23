# Experiment Log

This document records the experiments I ran while building the V-JEPA 2 interpretability demo, including results that didn’t match my expectations.

---

## Experiment 1: Attention overlays on motion vs static scenes

**Hypothesis:** last-layer attention should concentrate on moving objects/regions (motion carries predictive signal under masked latent prediction).

**Setup:** I used the Attention Explorer tab (and the standalone `src/attention.py`) on short 16-frame clips from the toy `data/mini_4x10/` library.

**Result:** in dynamic shots, attention maps concentrate on a smaller subset of patches; in more static shots, they are noticeably more diffuse.

**Interpretation:** even when I’m extracting “token-0 attention” rather than full attention matrices, the attention-derived saliency behaves like a motion/structure prior: it’s stable over a few frames and not purely edge-detection noise.

---

## Experiment 2: Temporal embedding drift

**Hypothesis:** embedding drift should be smooth for stable scenes and spike on cuts / fast motion.

**Setup:** Temporal Drift tab; 16 frames, mean pooling.

**Result:** the similarity heatmap shows strong near-diagonal structure (high similarity for nearby frames) and lower similarity across larger temporal gaps, with sharper changes on high-motion clips.

**Artifacts:** `assets/similarity_heatmap.png`.

---

## Experiment 3: Latent-space nearest neighbors on a toy library

**Hypothesis:** a clip’s nearest neighbors should often come from the same source video, even without supervision.

**Setup:** I used the extracted clip-level embeddings from `data/features_mini_vjepa_vitb.npz` (4 classes × 10 clips). I evaluated leave-one-out Precision@K where “label” is the source film folder name.

**Result:**
- Precision@1 = **0.65**
- Precision@3 = **0.592**
- Precision@5 = **0.560**

Chance for 4 balanced classes is 0.25.

**Interpretation:** on a tiny library, the representation is already more semantically coherent than chance, but it’s far from perfect—likely because these clips include visually diverse content within the same film and only a single mean-pooled embedding represents the whole clip.

---

## Experiment 4: Linear probe (tiny 4×10 dataset)

**Goal:** quantify how much class-relevant signal exists in frozen V-JEPA features.

**Setup:**
- Dataset: `data/mini_4x10/` (4 source videos × 10 clips each; 2.5s clips)
- Features: `data/features_mini_vjepa_vitb.npz` (mean pooled)
- Classifier: sklearn `LogisticRegression`
- Split: stratified 80/20 (seed=42)

**Result:**
- V-JEPA probe test accuracy: **75.0%**
- Pixel baseline test accuracy: **87.5%**

**Interpretation:** this result is not what I wanted: the pixel baseline beating the representation suggests the dataset is too small / too easy for per-source classification, and that the test split variance dominates. I treat this as a pipeline smoke test—not a representation quality claim.

**Artifacts:** `outputs/probe_mini_4x10/` (saved locally; ignored by git).

---

## What I’d do with more time/compute

- Run the linear probe on a real action dataset subset (e.g., UCF-101 10-class subset) and report confidence intervals across multiple splits.
- Compare pooling strategies (`mean` vs `max` vs `CLS`) and layer choice for both probe accuracy and retrieval precision.
- Add a small curated reference library (10–30 short clips) committed to the repo for a “works out of the box” retrieval demo.

