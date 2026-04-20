# EEN Interview Deck — Person-Centric Foundation Model & Weapon Detection

> **Framing notes**
> - Style mirrors `pptnotes_revised.md` (per-section rationale, soundbites, push-back prep) and `report_corrected.pdf` (data tables, diagnostic "why X wins/fails" sections, honest caveats).
> - **External-safe.** Benchmark numbers below are from public datasets (Market1501). Internal dataset scale, weapon-detection field performance envelope, infrastructure specifics, verbatim VLM prompts, and internal model code-names are held back. Where absolute numbers would cross that line, qualitative framing is used ("a large-scale internal surveillance corpus", "multi-fold improvement", "varied deployment conditions").
> - One project, two approaches on top of one foundation — same shape as the CMT speaker project: *hand-crafted stats vs generic time-series foundation model* there; *web-CLIP vs person-centric CLIP* here.

---

## Narrative flow

The deck follows a deliberate arc, echoing the CMT report spine:

```
  Problem framing
        ↓
  What the data is (surveillance ≠ web imagery)
        ↓
  Approach 1: CLIP-as-captioner   →   failure diagnosis   →   Approach 2: VLM captioner
        ↓
  Pretraining recipe (common backbone, one embedding)
        ↓
  Validation signal (zero-shot probe on public benchmark)
        ↓
  Downstream product family
        ↓
  Weapon detection case study
        ↓
  Diagnostic — what actually drove quality
        ↓
  Roadmap
```

---

## 1. The deployment metric is worst-case, not offline mean

- State early: in surveillance, the product is judged by **precision under hard conditions** — occlusion, low light, distance, motion blur — not by a clean test-set average.
- Why it changes everything: it's a min-max objective. A model that's 99% on clean crops and 40% on occluded ones is *worse* than a uniformly 85% model, because alert systems live off the bad frames. Analogous to p99 latency vs mean latency.
- This framing drives every downstream choice: evaluate on surveillance-hard slices, pretrain with surveillance-like data, design the downstream head for failure modes not average performance.

**Interview soundbite:**

> *"Surveillance AI is a tail-of-distribution problem. The mean is easy; the worst-case camera at the worst-case angle is what customers actually care about. Everything in this deck — the captioner choice, the zero-shot probe, the temporal head on the weapon detector — follows from that one fact."*

**Why this belongs in the deck:** Signals you read the *deployment* problem, not just the paper-friendly version. Sets up every later slide.

**Takeaway:** If your metric is min-across-conditions, you don't train for mean accuracy — you train a representation that doesn't collapse in the tail.

---

## 2. Dataset intuition — what surveillance data actually is

- **Tight person crops**, not full frames. Often < 128 px in the long dimension. Subject occupies most of the crop; scene context is minimal.
- Artefacts you can't ignore: **partial visibility, occlusion, low resolution, motion blur, extreme aspect ratios, compression blocks.**
- Off-the-shelf CLIP has never seen this distribution. Its captioned pretraining data is web imagery: centered subjects, clean lighting, long depth of field.
- Identity lives in **clothing, accessories, body shape, carried objects, gait cues** — a persistent visual fingerprint across frames and cameras — not in scene gist.

**Qualitative data snapshot** (internal figures held back):

| Attribute | Web CLIP training | Surveillance deployment |
|---|---|---|
| Subject size | full frame, centered | tight crop, often off-center |
| Resolution | high (≥ 224 px side) | often < 128 px side |
| Lighting | studio / daylight | daylight + dusk + IR |
| Occlusion | rare | frequent |
| Caption availability | yes, rich | none |
| Scene diversity | very high | fixed per camera |

**Interview soundbite:**

> *"Mental model: web-CLIP trained on magazine covers; we needed something trained on parking lot footage. That's a distribution shift you can't paper over with fine-tuning alone — the captions themselves have to be rewritten for the person-centric task."*

**Why this belongs in the deck:** Every modelling choice that follows presumes this picture. Plant it early.

**Takeaway:** The gap isn't model capacity; it's data distribution and caption alignment. Foundation-model choices have to address both.

---

## 3. Approach 1 — CLIP-as-captioner with attribute-tree prompting

**Method:**

- Build an attribute hierarchy (upper-wear × colour × texture × lower-wear × accessory × age × gender).
- Enumerate all template-filled captions — *"A photo of a person wearing a blue patterned t-shirt"* — pass each through frozen CLIP, pick the highest-similarity caption per image, feed those as pseudo-labels to contrastive training.

**Two problems killed this:**

| Problem | Symptom | Root cause |
|---|---|---|
| Combinatorial explosion | one attribute group (upper-wear alone) yields ~10³ captions per image; scale is untenable | attribute hierarchy is multiplicative |
| Background bias | colour predictions track scene colour, not clothing colour | CLIP was trained on full images, cannot isolate person semantics from scene |

**Diagnostic — qualitative review findings:**
- On images with a clear subject and neutral background: colour predictions correct ~80% of the time.
- On images where background had a dominant colour (wall, vehicle, sky): colour predictions wrong > 50% of the time, **biased toward the background colour.**
- **Conclusion:** CLIP-as-captioner for person-only attributes is a category error. CLIP was never trained to isolate person semantics from scene; asking it to caption a person-only attribute in a cluttered scene is asking the wrong question of the model.

**Interview soundbite:**

> *"We tried to bootstrap captions using CLIP itself as the labeller. It failed for a structural reason: CLIP's representation bakes in the whole image, and for a person-centric task the background is noise we can't subtract out at caption time. That's what made us switch to a VLM captioner designed for subject-specific description — not because VLMs were trendy, but because the failure mode told us what the next model had to fix."*

**Why this belongs in the deck:** Pre-empts *"why not just prompt CLIP?"* Shows you tried the cheap thing first, understood *why* it failed, and the failure mode drove the next choice.

**Takeaway:** CLIP's full-image bias is a feature for scene understanding and a bug for person-only captioning. Know which regime you're in.

---

## 4. Validation as a feedback loop

Same principle as the CMT project's PCA sanity check: cheap probe before scaling up.

```
Raw person crops
      ↓
Caption generation (VLM + prompts)
      ↓
Contrastive pretraining (small subset)
      ↓
Zero-shot ReID probe on Market1501 (cosine similarity, no fine-tuning)
      ↓
   (if signal improves over OTS CLIP) → scale up, train downstream heads
   (if not)                            → revise captions or pretraining
```

- The zero-shot probe takes minutes to compute once the backbone is trained. It catches bad pretraining before we burn weeks on downstream heads.
- Same function as the PCA-of-features plot in the CMT project: *verify representation quality before scaling up*.

**Why this belongs in the deck:** Shows disciplined ML practice — the verified improvement here is what *earned* the right to build the downstream family.

**Takeaway:** Foundation-model work without a cheap representation probe is gambling. Build the probe first.

---

## 5. Approach 2 — VLM captioner with iterative prompt refinement

- Switched to a **vision–language captioner** (public pretrained VLM) designed for subject-level description, not image-level similarity.
- **Prompt engineering was the actual IP.** Seven rounds, each driven by a specific failure class we observed in the generated captions:

| Round | What we fixed | Why it mattered |
|---|---|---|
| 1 | Basic structured format (gender / clothes / activity) | Starting point; too generic, too scene-focused |
| 2 | Pulled activity out of the identity sentence | Activity prediction is noisy and unhelpful for ReID |
| 3 | Capped caption length | Long captions truncate badly in CLIP's 77-token budget |
| 4 | Added accessories (bag, phone, cap, glasses, carried objects) | Accessories are **strong identity signals** AND seed the embedding with weapon-relevant semantics for free |
| 5 | Added skin tone; capped to 3 sentences | Missing identity cue; length control |
| 6 | "If not applicable, don't mention it" | VLM was hallucinating attributes that weren't in the image |
| 7 | Final word cap + comprehensive attribute set | Fit CLIP's tokenizer budget without truncating identity info |

- **The round-4 decision is load-bearing for everything downstream.** By explicitly mentioning carried objects when present, the pretraining corpus became *weakly labelled* for those attributes — including weapons — before any supervised weapon dataset existed.

**Interview soundbite:**

> *"The iteration wasn't cosmetic. Each round was a specific failure mode — hallucinated attributes, tokenizer truncation, scene bleed. And round 4 — mentioning carried objects — mattered for downstream tasks we hadn't built yet, because it seeded the embedding space with weapon-relevant semantics before we had a single weapon label."*

**Why this belongs in the deck:** Justifies why caption quality is the lever, not model size. Sets up the weapon-detection payoff (section 9).

**Takeaway:** In a foundation-model pipeline, the captioner prompt is part of the architecture. Treat it with the same rigor as the model itself.

---

## 6. Pretraining recipe — one embedding, many products

- **Architecture:** CLIP-style contrastive image–text pretraining, ViT-B/16 vision tower, standard text transformer.
- **Loss:** InfoNCE contrastive.
- **Training regime:** AMP mixed precision, cosine LR schedule with warmup, batch 196, ~32 epochs.
- **Caption augmentation via sentence chunking:** long VLM captions broken into 2–3-sentence fragments, giving multiple text positives per image. Same image → several captions emphasising different attributes → embedding learns that all describe the same identity. Acts as regularisation.
- **Design principle:** freeze the backbone downstream; every new product is a small classification head or a retrieval index on top. No per-product re-pretraining.

**Interview soundbite:**

> *"The return on a foundation model is entirely about downstream reuse. If each new product needs its own pretraining run, you haven't built a foundation — you've built a model family. We built one embedding, and everything after was a thin head."*

**Why this belongs in the deck:** Pre-empts *"why not train task-specific models?"* with the *reuse economics* argument, not the *foundation models are cool* argument.

**Takeaway:** One backbone, frozen. Every head cheap. That's the whole economic case.

---

## 7. Results — zero-shot ReID on Market1501 (public benchmark)

All zero-shot except the supervised upper bound. Image encoder → frozen ReID head → Rank-1 / Rank-5 on Market1501 query/gallery.

| Backbone | Train data | Rank-1 | Rank-5 |
|---|---|---|---|
| Off-the-shelf CLIP (OpenAI ViT-B/16) | none (zero-shot) | 0.124 | 0.234 |
| Ours, initial pretraining | person-only captions, zero-shot for ReID | 0.258 | 0.428 |
| **Ours, HP tuned** | **person-only captions, zero-shot for ReID** | **0.439** | **0.632** |
| *Supervised upper bound* (CLIP-ReID trained on Market itself) | Market train set | *0.955* | *0.984* |

**Headline: 3.5× Rank-1 over OTS CLIP, zero-shot.** No ReID-specific training — a pure pretraining-distribution win.

**Diagnostic — why this result:**

1. **Caption alignment.** Our captions describe the person in person-centric language, so the image encoder learns person-relevant structure. OTS CLIP's captions describe the scene, so its embedding is scene-dominant.
2. **Tighter distribution.** We pretrain on actual person crops, not full frames. The encoder never has to "find the person" at inference.
3. **Weak supervision from accessories (round 4).** Some Market1501 queries hinge on accessories (bags, hats). Our embedding already encodes these because the captions mentioned them.

**Push-back to be ready for:**

> *"3.5× over zero-shot CLIP sounds impressive, but you're still at 0.44 Rank-1. Isn't that just a weak model?"*

Answer: *"The 0.44 is zero-shot on a benchmark our model has never seen in any form — no supervised signal, no domain adaptation, just pretraining-distribution transfer. The honest read is: it proves the embedding has learned identity-relevant structure. Supervised ReID on top of this backbone closes most of the gap to the 0.95 ceiling. The zero-shot number is what matters because it's what lets every other downstream product — attributes, action, weapons — bootstrap without per-product labelling."*

**Why this belongs in the deck:** Central empirical result. Earns the right to claim the pretraining worked. The 3.5× ratio is the headline; the absolute number anchors it honestly.

**Takeaway:** Pretraining distribution is the dominant lever. A 3.5× zero-shot improvement is what pretraining alignment looks like.

---

## 8. One backbone, many products

Diagram: central "person embedding" node with spokes to five downstream heads.

- **Person re-identification** — contrastive retrieval across cameras.
- **Person attributes** — lightweight FC heads for clothing colour, age, gender.
- **Action / person-attribute extension** — second-stage pretraining adds interaction and activity understanding.
- **Natural-language video search** — dual-index retrieval (person embedding + scene embedding, merged ranking).
- **Weapon detection** — classifier head on the action-extended embedding (section 9).

**Economics (qualitative):**

| Product | Training time before foundation | Training time after |
|---|---|---|
| New attribute head (e.g., colour) | weeks of data curation + model training | days — FC head on frozen embedding |
| New retrieval product | full contrastive run from scratch | hours — index the existing embedding |
| Weapon detection head | would need full supervised dataset | weeks — weak labels from captions + small supervised set |

**Interview soundbite:**

> *"Each of these used to be a standalone model with its own data pipeline and training run. Consolidating to one backbone let new products ship in weeks rather than quarters — and, importantly, the products agree with each other because they're reading the same representation."*

**Why this belongs in the deck:** The *business* case for foundation models, told through concrete products. Shows the work had compounding value, not one-off research.

**Takeaway:** The second downstream head is where the investment starts paying off. The third is where it compounds.

---

## 9. Weapon detection — the case study

**Why this is the interesting downstream for an interview audience:**
- High-stakes, low-tolerance-for-false-positives.
- Where the round-4 caption decision (section 5) paid off — weapon semantics were pre-seeded in the embedding.
- Shows the foundation-model approach scaling from "nice-to-have features" to safety-critical.

**Architecture (high level):**

```
  Frame t-2      Frame t-1        Frame t
      ↓              ↓                ↓
   ┌─────────────────────────────────────┐
   │  Frozen person embedding            │
   │  (action-extended backbone)         │
   └─────┬──────────┬─────────┬──────────┘
         ↓          ↓         ↓
       e(t-2)     e(t-1)     e(t)       ← 384-dim per frame
         │          │         │
         └──────────┼─────────┘
                    ↓
         ┌────────────────────┐
         │ Temporal transformer│    (2 layers, small)
         └──────────┬──────────┘
                    ↓
             ┌────────────┐
             │ classifier │ ──→ p(weapon)
             └────────────┘
```

- **Input:** person-level crop surfaced by the deployment pipeline.
- **Backbone:** the action-extended person embedding (frozen / lightly adapted).
- **Single-frame head:** lightweight classifier over the embedding.
- **Temporal head:** for video, per-frame embeddings combined by a small transformer over a short window. Rationale: single frames are ambiguous (angle, clutter, hand position); persistence across frames disambiguates.

**Validation approach (methodology only — specifics held back):**
- Field-realistic testing across **varied deployment conditions** — multiple distances, lighting regimes, and motion patterns.
- Continuous production performance monitoring with drift alerting.

**Push-back to be ready for:**

> *"How do you handle false positives in a safety-critical classifier?"*

Answer: *"Two layers. At model time, the temporal head raises the bar — we don't alert on a single frame; we require the signal to persist across a window. At system time, weapon alerts are designed as human-in-the-loop — the model surfaces candidates; operators confirm. The classifier is a high-recall, reasonable-precision filter on a firehose, not an autonomous alarm."*

> *"Why not train a dedicated weapon detector end-to-end?"*

Answer: *"We could, but then we'd be back to a standalone model with its own data pipeline, its own pretraining, and no shared improvements when the person embedding gets better. Sitting on the foundation backbone means every upstream improvement to the person embedding — better captions, more data, better contrastive objectives — lifts the weapon detector automatically."*

**Why this belongs in the deck:** Demonstrates the foundation-model investment paying off on a product that matters. Also signals operational maturity — you're not just training classifiers, you're thinking about deployment failure modes.

**Takeaway:** Weapon detection is the *test* of the foundation-model thesis: if the backbone is good, the head can be thin and ship fast. It was, and it did.

---

## 10. Diagnostic — what actually drove quality

Same class-size-vs-similarity story as the CMT project's section 10: the obvious bottleneck wasn't the real one.

**What we *thought* would drive quality:**
- More data (bigger corpus)
- Bigger backbone
- More training epochs

**What actually drove quality (ordered by measured contribution):**
1. **Caption quality.** Round-7 prompts produced ~2× more usable captions per image than round-1. This was the single biggest lever.
2. **Caption augmentation (sentence chunking).** Sentence-level augmentation turned verbose captions from "unusable" into "multiple positives per image." Net: ~1.7× effective training pairs.
3. **Hyperparameter tuning.** HP sweep on the tuned pipeline lifted Market1501 Rank-1 from 0.258 → 0.439 — a bigger move than doubling the dataset would have been.
4. **Backbone size.** Didn't matter nearly as much as 1–3 on the data regime we had.

**Answer to the inevitable interviewer question ("did you try a bigger model?"):**

> *"I did — and it mattered much less than caption quality. A 2× corpus of bad captions is worse than a 1× corpus of good ones, because the contrastive objective amplifies caption noise. Capacity was not the bottleneck in this regime; caption alignment with the person-centric task was. Mirrors a pattern I saw on the CMT speaker task — the smallest speaker class still got 100%, so it wasn't sample count that was driving worst-speaker accuracy there either."*

**Why this belongs in the deck:** Shows you diagnosed the *actual* driver of quality instead of reaching for the textbook "more data / bigger model" answer. Pre-empts the obvious interviewer question.

**Takeaway:** The bottleneck is rarely what the naive story says it is. Measure.

---

## 11. Roadmap — prioritised improvements

Four moves, ordered by gain-to-effort ratio (mirrors CMT section 11):

**1. Swap the generic VLM for a surveillance-aware captioner.**
- The biggest remaining caption-quality lever. Current VLM hallucinates on low-resolution crops. A captioner pretrained on surveillance imagery would fail gracefully instead of inventing attributes.
- Expected gain: more usable captions → better contrastive signal → better downstream on every head.

**2. Second-stage contrastive with hard-negative mining.**
- Current pretraining uses generic contrastive loss. A hard-negative scheme — pairs of visually similar but different-identity crops — would sharpen the embedding where it matters: the confusable-identities region. Analogous to metric learning in face recognition.

**3. Distill the backbone for edge deployment.**
- Surveillance runs on bandwidth-constrained edge devices. ViT-B/16 is too heavy. Distill to a smaller student (ViT-Tiny, MobileViT) so the foundation-model wins reach production without a compute tax.

**4. Cross-head consistency losses on downstream heads.**
- Example: if the attributes head says "red shirt" and the ReID head says "same person as crop X," the two should be consistent with X's attributes. A consistency loss couples the heads and forces the backbone to encode attributes robustly.

**What I'd *not* do:**
- Train a bigger backbone for its own sake — current bottleneck is caption quality, not capacity.
- Add more public-web data — dilutes the surveillance-specific signal.
- Replace contrastive with masked-image modelling — MIM is good for dense prediction, weaker for the retrieval-style tasks that dominate our product surface.

**Interview soundbite:**

> *"The biggest remaining lever is caption quality, not model size or data volume. Everything downstream is bottlenecked by how well the captioner describes the person. That's where I'd spend the next quarter of engineering time, not on a bigger transformer."*

**Why this belongs in the deck:** Right-sizing the proposals to where the *actual* bottleneck is — caption quality, not capacity or data — is itself a signal of good engineering judgement.

**Takeaway:** Know where your next 10% is coming from. Here it's captions, not capacity.

---

## 12. Summary slide

| Dimension | Off-the-shelf CLIP | Ours (person-centric) |
|---|---|---|
| Pretraining data | web-scale image–text pairs | person crops + VLM captions |
| Captioner | N/A (pretrained on web) | VLM with 7-round tuned prompt |
| Zero-shot ReID Rank-1 (Market1501) | 0.124 | **0.439** (3.5×) |
| Downstream heads shipped | N/A | 5 (ReID, attrs, action, search, weapon) |
| Per-product training cost | standalone | thin head on frozen backbone |
| Weapon semantics in embedding | none | pre-seeded via captioner round 4 |

**Three sentences to leave on the screen:**

1. Off-the-shelf CLIP is scene-biased; we built a person-centric variant by fixing the *captions*, not the architecture.
2. One embedding powers five products — the compounding value of foundation-model investment.
3. The weapon detector was a thin head, not a new model — proof the foundation was doing its job.

---

# Backup Notes / Q&A Prep

*Detail you don't need on a slide, but should have in your head.*

## B1. Why CLIP-style contrastive, not a classifier?

- A classifier requires a fixed label set. A surveillance product has an open-ended label space — new attributes, new accessories, new query types appear continuously. Contrastive image–text learning yields an embedding that accepts **any** text query at inference time.
- This is what makes natural-language video search possible at all: user types "man in red hoodie carrying a bag," we encode the text, cosine-similar against the indexed person embeddings. No per-query training.

## B2. Why ViT-B/16 and not ResNet / ConvNeXt?

- CLIP's original recipe used ViT; the pretrained OpenAI checkpoint is a strong initialiser we didn't want to throw away.
- ViT handles variable-resolution crops gracefully via patch tokenization, which matters when surveillance crops aren't canonical sizes.
- Inductive-bias-light architectures generalise better when the training distribution is large and noisy — our regime.

## B3. Why freeze the backbone for downstream heads?

- **Sample efficiency:** supervised downstream datasets (weapon, attributes) are orders of magnitude smaller than the pretraining corpus. Fine-tuning the backbone on them overfits and *destroys* the foundation-model benefits we paid for.
- **Consistency:** frozen backbone means all downstream heads share the *same* embedding, so the attributes head and ReID head agree about what the person looks like. Per-head fine-tuning breaks that.
- **Deployment:** one backbone checkpoint served once; heads hot-swappable.

## B4. Why zero-shot as the validation signal?

- Zero-shot isolates *pretraining quality* from downstream fine-tuning. If the backbone is bad, no fine-tuning on Market saves it — but you won't see that if you fine-tune, because fine-tuning smears everything toward the same ceiling.
- The zero-shot delta over OTS CLIP is the honest measure of "did our pretraining teach the model anything new."

## B5. Why temporal for weapon detection but not for ReID?

- ReID is cross-camera, cross-time — retrieval happens across clips hours apart. Temporal modelling *within* a clip doesn't help; the embedding has to survive the cross-clip gap anyway.
- Weapon detection is a within-clip, within-seconds decision. A gun is either being held over this 2-second window or it isn't. Temporal modelling resolves frame-level ambiguity (angle, partial occlusion) that doesn't average out at clip level.

## B6. Why caption augmentation via sentence chunking?

- VLM captions were long and descriptive — one image, one 200-token caption. Directly tokenising loses information past CLIP's 77-token limit.
- Chunking into 2–3-sentence fragments gives multiple text positives per image in the contrastive batch. Each fragment emphasises different attributes (clothing in one, accessories in another, age/build in a third).
- Net effect: same image maps to several valid captions → embedding learns that *all* of them describe the same identity → robustness to which attribute a query emphasises.

## B7. How does this compare to the CMT speaker project (interviewer might link the two)?

Good linkage to have ready — shows both pieces of work came from the same methodological instincts:

| Dimension | CMT speaker-ID | EEN person FM |
|---|---|---|
| Problem shape | classify into 9 classes | learn a reusable embedding |
| "Obvious big model" tried | MOIRAI (time-series FM) | off-the-shelf CLIP |
| Why it lost | domain + task mismatch on small data | scene-biased captions, not person-aligned |
| "Simple thing" that won | 49 hand-crafted stats + SVM | person-centric CLIP with VLM captions |
| Validation discipline | PCA-of-features sanity check | zero-shot ReID probe |
| Real bottleneck | acoustic similarity, not class size | caption quality, not model size |
| Worst-case focus | min-across-speakers metric | worst-condition surveillance metric |

**One-liner:** *"Both projects made the same bet — that a well-chosen, domain-aligned representation beats a big generic model on the metrics that matter. The bet paid off in both cases, for the same structural reason."*

---

## Suggested 15-slide order

| # | Slide | Visual |
|---|---|---|
| 1 | Title | — |
| 2 | Metric framing (§1) | — (text) |
| 3 | Dataset reality (§2) | web vs surveillance side-by-side |
| 4 | First attempt: CLIP as captioner (§3) | attention-heatmap bleed diagram |
| 5 | Feedback-loop methodology (§4) | flowchart |
| 6 | VLM captioner + prompt iteration (§5) | table of 7 rounds |
| 7 | Pretraining pipeline (§6) | three-lane data/caption/training diagram |
| 8 | Validation: zero-shot ReID (§7) | horizontal bar chart |
| 9 | One backbone, many products (§8) | hub-and-spoke |
| 10 | Weapon detection — motivation (§9) | — (text) |
| 11 | Weapon detection — architecture (§9) | temporal transformer diagram |
| 12 | What drove quality (§10) | ranked-factors table |
| 13 | Roadmap (§11) | 2×2 effort-vs-impact |
| 14 | Summary + 3 sentences (§12) | comparison table |
| 15 | Backup slides | — |

---

## What's intentionally held back vs the internal version

- Internal dataset names, per-source counts, aggregate corpus size
- Specific training throughput, GPU memory constraints, time-to-train
- Verbatim VLM prompts (shown iteration rationale and what each round fixed, not the prompt strings)
- Weapon detection distance/lighting/IR performance numbers and test methodology specifics
- Named colleagues, Jira IDs, runbook and monitoring URLs
- Secondary-classifier gating architecture and failure modes
