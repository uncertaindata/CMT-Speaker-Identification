
## 1. The graded metric is worst-speaker accuracy, not mean

- State early that the grading metric is **min-across-speakers accuracy**, not mean.
- Why it changes everything: it's a min-max objective. A model that's 99% on 8 speakers and 40% on one is *worse* than a model at 85% on all nine. Analogous to p99 latency vs mean latency.
- This framing drives every downstream choice: stratified CV, per-speaker reporting, class-imbalance awareness, not blindly optimising cross-entropy / plain accuracy.

**Why this belongs in the deck:** Shows you read the problem statement carefully and that your methodology is metric-aware. Sets up every later slide (per-speaker bars, confusion matrices) — the audience immediately knows why those views matter.

---

## 2. Dataset intuition — what this data actually is

- 9 male speakers, each saying **the same single vowel** for 0.7–2.9 seconds. No words, no phoneme transitions.
- Organisers pre-processed the audio into **12 LPC Cepstrum coefficients per ~100 ms frame** → each utterance is a `(T, 12)` matrix, T ∈ [7, 29]. Think of each row as a snapshot of the speaker's vocal-tract shape at one instant.
- Since the vowel is held, the signal is **approximately stationary** — each column (one coefficient over time) is a noisy near-constant, not a meaningful trajectory.
- Speaker identity lives in **timbre → vocal-tract shape → the distribution of each LPC coefficient**, not in the temporal ordering of rows.

**Why this belongs in the deck:** You can't justify modelling choices without first planting the picture of what the data *is*. This slide is the foundation for every design choice that follows.

---

## 3. Approach 1 — 49 hand-crafted statistics

- For each of the 12 coefficients, compute **mean, std, min, max** over time → 48 features. Add utterance **length** → **49**. Fixed-length regardless of T.
- Throwing away temporal order is **fair for this task**: for an approximately stationary signal, the distribution *is* the signal. Onset/offset shape is minor compared to the speaker's vocal-tract fingerprint.
- Note: **std itself is a temporal feature in disguise** — it summarises how much the coefficient varied over the utterance.
- This would be malpractice for word/phoneme classification where order matters. It's right-sized for speaker-ID on a single steady vowel.

**Interview soundbite:**

> *"The README specifies a single vowel per utterance. There's no phoneme transition, so the signal is approximately stationary. Per-coefficient stats capture the distribution that defines the speaker; I'm losing onset/offset shape but not the speaker's vocal-tract fingerprint. If the task involved phonemes or transitions, I'd have kept the time axis — but it didn't, so I didn't."*

**Why this belongs in the deck:** Pre-empts *"why didn't you use an RNN/LSTM?"* and turns the feature choice from a hack into a deliberate match between representation and task structure.

---

## 4. PCA sanity check as a feedback loop

```
Raw data (T, 12) per utterance
        ↓
Hand-crafted features (49 stats per utterance)
        ↓
PCA sanity check  ← "did the features preserve speaker signal?"
        ↓
   (if yes) → train classifier
   (if no)  → back to feature engineering
```

- Before committing compute to training an SVM / MOIRAI / anything, project the feature matrix to 2D via PCA and **colour points by speaker**.
- If speakers form visibly separable clusters in 2D, the features carry the signal and a classifier will likely work. If the colours are smeared together, no classifier will save you — go back and fix the features.
- This is a **seconds-long test** that catches bad feature design before you waste hours training.

**Why this belongs in the deck:** It's the CMT-advised "establish a feedback loop early" principle made concrete (README advice line 101). Shows disciplined ML practice: *verify representation quality before scaling up*. Sets up section 8, which is the plot this methodology produces.

---

## 5. Approach 2 — MOIRAI embeddings

- MOIRAI is **not** fed the 49 stats. It's fed the raw `(T, 12)` utterance, patch-by-patch: each of the 12 coefficients is chunked into patches of 8 timesteps, passed through a 13.8M-param transformer encoder, and the patch hidden states (384-dim each) are **mean-pooled** into one 384-dim vector per utterance.
- That 384-dim vector goes into the **same SVM-RBF classifier** used for the stat approach.
- **Clean experimental design:** classifier head held constant; only the feature extractor changes. Any performance gap is attributable to the representation, not the downstream model.

**Why this belongs in the deck:** Positions the two approaches as a controlled comparison. Lets you later explain the performance gap in terms of *representation quality*, not classifier tuning or hyperparameter luck.

---

## 6. Classifier choice — SVM + RBF (common to both approaches)

- PCA showed **tight but curved** cluster structure in 49-dim feature space → a linear classifier would underfit. Need a non-linear boundary.
- SVM + RBF gives smooth curved boundaries, is **sample-efficient on n=370**, handles correlated features (mean/std/min/max of the same coefficient) naturally, and has only **2 hyperparameters (C, γ)** — a big win on a 1-day timeline.

**Alternatives rejected:**

| Classifier | Problem on this data |
|---|---|
| Logistic regression | Linear only → underfits curved clusters |
| Random forest / XGBoost | Axis-aligned splits, want more data, weaker on correlated features |
| k-NN | No learned margin, sensitive to noisy neighbours |
| Neural net | Overkill for 370 × 49, high overfit risk |

**Interview soundbite:**

> *"49 hand-crafted features produced curved but tight cluster structure in PCA, so I needed a non-linear classifier. RBF-SVM is sample-efficient on n=370, handles correlated features naturally, and has only 2 hyperparameters — right-sized for a 1-day project. Random forest / XGBoost are strong tabular defaults but want more data; neural nets would overfit."*

**Why this belongs in the deck:** Justifies the classifier choice with data-grounded reasoning, not "I used SVM because it worked." Shows you matched tool to data regime. (SVM mechanics in Backup B1–B5.)

---

## 7. Results — conventional ML beat MOIRAI

- Headline (**5-fold CV on training**): **stat+SVM: 0.973 mean / 0.897 worst** vs **MOIRAI+SVM: 0.862 mean / 0.690 worst**. MOIRAI loses by ~11 pp on mean and ~21 pp on the graded worst-speaker metric.
- **Why stat features win:** the data is *tiny* (370 samples) and *short* (7–29 timesteps). Pretrained time-series foundation models shine on long, rich signals with lots of temporal structure — not short LPC frames where a handful of distributional stats already capture almost everything.
- **The three-way diagnosis for MOIRAI's underperformance:**
  1. **Tiny n (370):** 384-dim learned features + 370 samples is a poorly-posed learning problem vs 49 hand-crafted features + 370 samples.
  2. **No domain prior:** MOIRAI was pretrained on generic time series — never saw speech or LPC. Most of its 384 dims are probably irrelevant *to this task*, and on 370 samples the SVM can't sift through them.
  3. **Short signals:** foundation time-series models shine on long, rich signals — not near-stationary LPC frames that fit in 1–4 patches.

**Why this belongs in the deck:** The central empirical result. Earns the right to talk critically about *when* foundation models help vs. when simple, domain-aligned features win. Shows scientific honesty — the "fancy" approach lost, and you can explain *why* in terms of data regime, not vibes.

---

## 8. PCA plot walkthrough — one picture that explains the whole result

Refer to [pca_comparison.png](../data_science_ai_interview_project_cmt/pca_comparison.png). Two 2D PCA projections of the same **370 training utterances** (we don't have test labels), coloured by speaker.

**Left (stat features, 49-dim → 2D, 34.3% var):**
- 7 of 9 speakers form visibly distinct clusters: S2 (green, far left), S6 (gray, top), S4 (brown, upper right), S0 (blue, right), S1 (orange, bottom-centre), S5 (pink, lower right).
- S3 (red), S7 (olive), S8 (cyan) overlap in the centre — the **confusable region**.
- This is the "PCA sanity check" from section 4 *passing*: before training any classifier, the features were visibly carrying speaker signal.

**Right (MOIRAI embeddings, 384-dim → 2D, 37.9% var):**
- No visible clusters. All colours smeared into one blob.
- Variance explained is actually *higher* than the stat plot — so this isn't a lossy-projection artefact. The top-PC directions of MOIRAI's 384-dim space simply aren't aligned with speaker identity.

**The central-region overlap in the left plot is the visual prediction of the confusion-matrix hotspot.** S8 at 0.897 (stat's worst) isn't a mystery — you could have predicted it from this plot before training anything.

**Push-back to be ready for:**

> *"Maybe MOIRAI's signal is in higher PCs, not PC1/PC2?"*

Answer: *"Possible, but (a) if the signal lives in higher PCs, the SVM has to find it among 384 mostly-irrelevant dims on 370 samples — the curse-of-dimensionality regime; (b) empirically the SVM did worse on MOIRAI features (0.862 vs 0.973). A larger dataset might change the story; on 370 samples, stat features' alignment with speaker-relevant structure is why they won."*

**Interview soundbite:**

> *"The left plot shows stat features projecting speakers into 7 visible clusters at 34% variance — the features are doing most of the classification job before SVM even starts. The right plot shows MOIRAI's embeddings have no visible speaker structure at 38% variance. This single comparison predicts the entire result."*

**Why this belongs in the deck:** One picture argues four things at once — that the feature design worked, that MOIRAI underperformed because of data regime not model quality, that there's a central acoustic-similarity region to worry about (section 9), and that PCA-as-feedback-loop is load-bearing, not decorative. If you pick one image for the deck, pick this one.

---

## 9. Confusion matrices — where errors go tells you *why*

Refer to [cm_conventional.png](../data_science_ai_interview_project_cmt/cm_conventional.png) and [cm_moirai.png](../data_science_ai_interview_project_cmt/cm_moirai.png). Rows = true speaker, columns = predicted. **Both matrices are built from 5-fold CV predictions on the training set** (we don't have test labels); see the "(CV)" in the plot titles.

**Stat + SVM (blue):**
- Near-pure diagonal; off-diagonals are few and **concentrated**.
- **S8 → S0 (2/3 of S8's errors)**, S8 → S4 (1) — worst speaker fails in a *specific direction* (acoustic twin).
- **S7 is an error hub**: S0, S1, S3 all occasionally mis-predicted as S7. Matches the PCA central overlap (red/olive/cyan) from section 8.

**MOIRAI + SVM (green):**
- Diagonal dominant but **off-diagonals are everywhere**.
- **S6 → S2 (6 errors!)**, S7 → S2 (4), S3 → S2 (3): **S2 is a class-frequency magnet** — 88 training samples (largest class), so the classifier defaults toward it when features are weakly discriminative.
- **S4's 9 errors spread across 5 different classes** — no preferred confusion target, just no signal.

**The key contrast:**

| | Stat + SVM | MOIRAI + SVM |
|---|---|---|
| Error pattern | **Concentrated** on specific pairs | **Diffuse** across many pairs |
| Diagnosis | Acoustic similarity between specific speakers | Weak discriminative features |
| Fix | Engineer features to separate S8/S0 | Bigger dataset or speech-domain pretrain |

**Interview soundbite:**

> *"Both models are near-diagonal, but the off-diagonal pattern is diagnostic. Stat+SVM's errors are concentrated — S8's 3 errors split 2/1 between S0 and S4, a specific acoustic-twin story. MOIRAI's errors are diffuse — S4 fails across 5 different classes, and S2 (largest class) pulls errors from 4 other speakers, which is the class-prior leaking through when features are weak."*

**Slide layout suggestion (2-up with callouts):**
1. Stat matrix → point at **S8 → S0 = 2** ("worst speaker fails in a specific direction")
2. MOIRAI matrix → point at **S6 → S2 = 6** ("errors pulled toward the high-prior class")
3. MOIRAI matrix → point at **S4 row spread across 5 classes** ("diffuse = no speaker signal, not just a confusable pair")

**Why this belongs in the deck:** Gives the audience *specific numbers to remember* (6, 2, 9) rather than abstract worst-speaker percentages. Sets up section 10's interpretation.

---

## 10. Class size isn't the bottleneck — acoustic similarity is

Evidence (5-fold CV accuracy per speaker):

| Speaker | n (train) | Stat+SVM | MOIRAI |
|---------|-----------|----------|--------|
| S5      | **24 (smallest)** | **1.000** | 0.958 |
| S4      | 29        | 1.000    | **0.690 (MOIRAI worst)** |
| S8      | 29        | **0.897 (stat worst)** | 0.828 |
| S2      | 88 (largest) | 1.000 | 0.955 |

The smallest class (S5) is near-perfect on both models. The worst speakers have mid-sized sample counts. **Sample count is not what's driving worst-speaker accuracy.** What is: acoustic similarity between specific speakers — exactly what sections 8 and 9 showed visually and numerically.

**Answer to the inevitable interviewer question ("did you try class weighting / oversampling?"):**

> *"I looked at it, but class size isn't what's driving worst-speaker accuracy in this dataset. S5 is the smallest class and gets 100%, while S8 (larger) is the worst. So oversampling wouldn't fix the real problem — the real problem is acoustic similarity between certain speaker pairs, which needs better features, not rebalancing."*

**Why this belongs in the deck:** Pre-empts the obvious interviewer question with a data-grounded answer. Shows you actually looked at the per-speaker numbers instead of reaching for a textbook imbalance fix.

---

## 11. What I'd improve — prioritized next steps

Four moves, in order of expected gain-to-effort ratio:

**1. Soft-vote ensemble of stat+SVM and MOIRAI+SVM**

This is the single highest-value move. The evidence:

- CV shows the two models fail on *different* speakers (S8 for stat, S4 for MOIRAI). On the *other* model's weak spot each is strong — stat+SVM gets **1.000** on S4; MOIRAI gets 0.828 on S8.
- **Test-set model-to-model agreement = 81.1%** — the two models pick the *same* speaker on 81.1% of the 270 test blocks and pick *different* speakers on ~51 blocks. This is **not accuracy** (we don't have test labels); it's inter-model diversity. 81% is the sweet spot where models agree on easy cases and disagree on hard ones — textbook precondition for ensembling.
- *Soft vote* = average per-class probabilities from both models, then argmax.
- *Stacking* = train a logistic regression on the 9 + 9 = 18 probability columns (out-of-fold CV predictions to avoid leakage). More powerful in theory but overfits easier on 370 samples.
- **CV-based expectation:** worst-speaker lifts from 0.897 to ≥0.92. **Honest caveat:** actual test-set gain can only be verified once CMT scores the submission.

**2. Tune SVM hyperparameters with worst-speaker as the CV scoring metric**

Currently `C=10, gamma='scale'` picked off the cuff. Grid-search or Bayesian over (C, γ) — but **match the scoring objective to the grading metric**, not plain accuracy. Small but essentially free gain.

**3. Swap MOIRAI for a speech-domain foundation model (Wav2Vec 2.0 / HuBERT / WavLM)**

MOIRAI underperformed because of *domain mismatch* (generic time-series pretraining vs speech), not because foundation models don't work. A speech-pretrained model would have speaker-relevant features out of the box. This is where the foundation-model approach actually pays off on this data.

**4. Contrastive / metric learning on the 49 features**

Train a small network to produce embeddings where same-speaker distances are small and different-speaker distances are large, then k-NN in the embedding space. Particularly good at sharpening confusable pairs like S8/S0.

**What I'd *not* do:** deep learning from scratch (370 samples isn't a neural-net regime), heavy LPC preprocessing (organisers already did the speech-specific work), or over-tuning the current 49-feature approach past diminishing returns.

**Interview soundbite:**

> *"The biggest unlocks are: ensemble the two models since they fail on different speakers; tune SVM hyperparameters with worst-speaker as the scoring metric — matching the objective to the grading; swap MOIRAI for a speech-domain pretrained model like Wav2Vec because MOIRAI's issue was domain mismatch, not the foundation-model paradigm; and try contrastive metric learning to sharpen confusable speaker pairs. Everything else is either overkill for 370 samples or fights the given representation."*

**Why this belongs in the deck:** "What I'd do next" slides are where interviewers gauge your prioritization sense — can you pick the meaningful moves without getting lost in complexity? Right-sizing the proposals to the data regime (370 samples) is itself a signal.

---

# Backup Notes / Q&A Prep

*Detail you don't need on a slide, but should have in your head in case an interviewer probes.*

## B1. What SVM actually does (geometric picture)

- Each utterance = one point in 49-dim feature space. Same-speaker utterances cluster.
- SVM finds the boundary between two classes with the **widest empty corridor** around it. Wide corridor = robust classifier; noisy test points are less likely to flip.
- The boundary is pinned by whichever training points sit closest to it — those are the **support vectors**. Interior points (deep in their cluster) don't matter; delete them and retrain, you get the same boundary.
- Support vectors are **discovered by the optimiser, not chosen**. Typically ~10–40% of training points become SVs; the rest are discarded.

## B2. How prediction works

For a new test point:
1. Measure RBF similarity to each support vector.
2. Weighted-sum the similarities (weights = α values learned during training).
3. Sign of the sum → predicted class.

RBF decays exponentially with distance, so **only nearby support vectors contribute**. It's effectively a smart, learned, margin-aware version of k-NN.

## B3. What the RBF kernel is

- RBF = Gaussian similarity between two feature vectors: `k(x, y) = exp(-γ · ||x - y||²)`.
- Close in 49-dim Euclidean space → very similar; far → negligible.
- The kernel trick lets SVM draw smooth **curved shells** around clusters instead of flat cuts, without ever computing coordinates in the (conceptually infinite-dim) kernel space.

## B4. Multi-class via one-vs-one voting

- SVM is binary by construction. sklearn handles the 9-class problem via **one-vs-one**: train C(9,2) = **36 pairwise classifiers**, each on ~60–120 points from just its two speakers. Each pairwise problem is small and balanced — ideal for SVM.
- **Prediction:** run all 36, each votes for one of its two classes; majority wins.
- **Why the voting is robust:** each class is in only 8 of the 36 pairs → **max 8 votes** per class. If features are good, the 8 S3-vs-X classifiers all vote S3 → S3 wins with 8. Every other class is capped at 7, because their head-to-head against S3 correctly votes S3 instead of them.
- **Real errors come from head-to-head failures** (confusable neighbouring speakers), not from the 28 "irrelevant" pairs. That's why the confusion matrix is the diagnostic to read.

## B5. One-liner worth internalising

> **RBF-SVM is a smart, regularized, margin-aware k-NN** — it learns which training points define the boundaries and weights distance to them via a Gaussian.

## B6. MOIRAI forward-pass detail (in case asked "walk me through it")

For a T=15 utterance:
1. **Patch:** each of 12 coefficients chopped into chunks of 8 timesteps → ceil(15/8) = 2 patches per coefficient → 24 patches total.
2. **Pad:** each 8-value patch padded with zeros to width 128 (MOIRAI's fixed input width — supports multiple patch sizes via one weight matrix).
3. **Project:** each 128-wide patch × learned weight matrix W (128×384) + bias → one 384-dim token. This is where the 8 raw values get linearly combined.
4. **Add positional embeddings:** `time_id` (which time-chunk) and `variate_id` (which coefficient) — each looked up in learned embedding tables and added to the token. Without these, attention would be permutation-invariant and useless.
5. **Self-attention:** all 24 tokens attend to each other; each token's 384-dim state gets updated based on the whole set. Positional embeddings let attention distinguish "same-variate-across-time" vs "same-time-across-variates."
6. **Mean-pool across the 24 tokens → 384-dim utterance embedding.**

Why mean-pool rather than max-pool, concat, or CLS token? After attention, every token is globally-informed, so no single one is privileged. Averaging gives a consensus summary, respects the stationary / permutation-invariant structure of the data, produces a fixed-size vector regardless of T, and has zero learned parameters (we can't afford to learn a pooling head on 370 samples).
