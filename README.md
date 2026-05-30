# Deep Learning

Coursework submission for a Deep Learning module. A single notebook ([`deep.ipynb`](deep.ipynb))
answers six questions spanning classical regression, decision trees with PCA, word
embeddings, and three trained neural networks. Questions 4–6 additionally ship as
importable Python scripts with their own pretrained weights so the marker can run the
trained models against an unseen test set.

All code is Python 3 and is written to run top-to-bottom on **Google Colab**. Every script and
notebook cell assumes the data files live in the same folder as the notebook.

## Repository Contents

| File | Purpose |
| --- | --- |
| [`deep.ipynb`](deep.ipynb) | Main notebook — code, results, and written explanations. |
| [`predict_decay.py`](predict_decay.py) | Q4 deliverable — `predict(parameters)` for orbital-decay regression. |
| [`predict_product.py`](predict_product.py) | Q5 deliverable — `predict(images)` for dice-product classification. |
| [`compress_outlines.py`](compress_outlines.py) | Q6 deliverable — `encode(points, lengths)` / `decode(latents)` for glyph compression. |
| `weights_q4.pkl` | Trained weights for Q4 (~201 KiB; budget 1 MiB). |
| `weights_q5.pkl` | Trained weights for Q5 (~3.24 MiB; budget 15 MiB). |
| `weights_q6.pkl` | Trained weights for Q6 (~5.71 MiB; budget 20 MiB). |

The `.pkl` files are PyTorch `state_dict`s saved with `torch.save`; each `predict`/`encode`
script reloads them by relative path, so the scripts must be run from this directory.

## The Six Questions

### Q1 — Linear Regression Models (Hospital Bed-Days)
Fits OLS, Ridge, and Lasso with cross-validated hyperparameter tuning, then uses an
automated CV procedure to pick a final model. The three candidates are statistically tied
(CV-RMSE ≈ 1.515, spread ≈ 4e-5), so selection falls to **Ridge (α ≈ 7.94)**; held-out test
performance is RMSE 1.53 / MAE 1.23 bed-days / R² 0.67. The written parts argue why the
automated choice is reasonable here, and why an engineer might still override it on
operational grounds (predictor cost/availability, interpretability), with a concrete failure
mode.

### Q2 — Decision Trees, PCA, and Model Complexity
Uses `sklearn.tree.DecisionTreeClassifier` on a 75:25 split with a fixed seed. Sweeps
`max_depth` to find the smallest depth reaching 85% test accuracy on raw features (the
threshold is not reached within depth 1–20; best ≈ 0.832 at depth 4), repeats after
standardisation + PCA, and plots accuracy-vs-depth curves for **k ∈ {1, 2, 8}** retained
components. The discussion links retained components to information preservation, noise, and
the axis-aligned inductive bias of trees.

### Q3 — Word Embeddings & Semantic Similarity
Loads the pretrained **GloVe `glove-wiki-gigaword-100`** model via `gensim` and tokenises
*Pride and Prejudice* (Project Gutenberg #1342). For the keywords
`{good, bad, happy, sad, angry}` it reports the three nearest in-corpus neighbours by cosine
similarity, comments on semantic plausibility, highlights a misleading case, and discusses a
limitation of static embeddings for sentiment in literary text.

> Q3 pins package versions for reproducibility on Colab:
> `pip install --upgrade "scipy==1.16.3" "gensim==4.4.0"` (restart the runtime if prompted).

### Q4 — Neural Network Regression (Orbital Decay)
Predicts time-to-decay (days) from six orbital/satellite parameters. An MLP
(`DecayPredictionNetwork`): 6 inputs → 4 hidden layers of 128 units with SiLU → 1 output.
Mass and cross-sectional area are log-transformed, inputs are standardised, and the target
is modelled in log-space. Trained with a cosine LR schedule and best-validation snapshot.
**Best validation MAE ≈ 5.3 days**, well inside the < 50-day target.

### Q5 — Dice Product Prediction (CNN)
Predicts the product of the three visible faces of a rendered die, end-to-end from the image
(no classical pip detection). Framed as **16-class classification** over the distinct possible
products `{6, 8, 10, … , 120}`. `DiceProductCNN` is a from-scratch CNN: four conv blocks
(32→64→128→192) with BatchNorm/SiLU/max-pool, adaptive average pooling, dropout 0.3, and a
linear head (~844k params). Trained at 64×64; the predict script resizes the spec's 128×128
inputs down before inference. **Best validation accuracy ≈ 99.7%**, above the 95% target.

### Q6 — Font Glyph Compression (Autoencoder)
Compresses a variable-length closed-polygon glyph outline to a **32-D latent** and
reconstructs it. The encoder runs 1-D convolutions over the (x, y) sequence and pools with a
length-masked max + mean (concatenated) so padding to 200 points is handled correctly; an
MLP decoder maps the latent back to 200×2 points with a `tanh` output. Trained from scratch
against **Chamfer distance**. **Best validation Chamfer ≈ 0.0021**, inside the ≤ 0.005 target.

## Deliverable Script API

Each script loads its own weights and applies all preprocessing internally, matching the
interfaces specified in the coursework brief:

```python
# Q4 — predict_decay.py
predict(parameters)   # parameters: (B, 6) tensor  ->  (B, 1) decay times in days

# Q5 — predict_product.py
predict(images)       # images: (B, 3, 128, 128) tensor, values in (0, 1)  ->  (B, 1) integer products

# Q6 — compress_outlines.py
encode(points, lengths)   # points: (B, 200, 2), lengths: (B,)  ->  (B, 32) latent codes
decode(latents)           # latents: (B, 32)  ->  (B, 200, 2) reconstructed points
```
