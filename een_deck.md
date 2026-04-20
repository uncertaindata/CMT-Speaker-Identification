# Person Foundation Model — how it came together

> **Note on this deck**
> This is a walkthrough of how the person-centric foundation model we built actually came together — the starting problem, the dead ends, the pivots, what ended up mattering, and where weapon detection fits into the story.
>
> It's told in the order things happened, not as a post-hoc defence of every decision. Some choices seemed right at the time and turned out great; some looked right and weren't; some felt like detours and became load-bearing.
>
> External-safe framing: specific internal dataset counts, infrastructure specs, verbatim prompts, and weapon-detection field numbers are held back. Benchmark numbers shown are on public datasets (Market1501).

---

## How the story flows

```
  Person ReID wasn't working well enough on our footage
        ↓
  We figured out why — it was a distribution/caption problem, not capacity
        ↓
  First idea (CLIP-as-captioner) failed for a structural reason
        ↓
  Second idea (VLM captioner + prompt iteration) worked
        ↓
  The zero-shot probe told us pretraining was learning something real
        ↓
  Once the backbone was good, the downstream family emerged on its own
        ↓
  Action-person extension opened the door to weapon detection
        ↓
  Looking back: caption quality, not model size, is what actually drove results
```

---

## 1. Where we started

Person re-identification — matching the same individual across cameras — is a core surveillance capability. It drives alert routing, investigation workflows, and cross-camera search. For us it was an obvious place to invest.

Off-the-shelf models weren't delivering on our footage. Before reaching for bigger models, we wanted to understand *why* an image–text model trained on 400M web image–caption pairs couldn't do well on what looks like a pretty basic retrieval task.

The mental model we converged on: **CLIP was trained on magazine covers; our cameras look at parking lots.** Those are two different distributions, and the gap isn't something fine-tuning can cleanly close.

| | Web imagery (CLIP's training) | Surveillance imagery (our footage) |
|---|---|---|
| Subject size | full frame, centred | tight crop, often off-centre |
| Resolution | ≥ 224 px side | often < 128 px side |
| Lighting | studio / daylight | daylight + dusk + IR |
| Occlusion | rare | frequent |
| Caption availability | yes, rich | none |
| Scene diversity | enormous | fixed per camera |

Once we laid the two distributions side by side, the problem wasn't "is CLIP big enough" — it was "CLIP is aimed at the wrong thing."

---

## 2. Why generic CLIP kept falling short

Qualitatively — running CLIP on person crops, reading its attribute predictions, and comparing them against what we could see by eye — a consistent pattern showed up: **its representations were scene-dominant, not person-dominant.**

Concretely — if we asked it to tell us what colour shirt someone was wearing, it often gave us the colour of the wall behind them. Same for vehicles: tell me about the car, get a description biased by the street colour.

Why? Because CLIP was trained on *full images* paired with captions describing *the whole scene*. "A person in a blue shirt standing in front of a red wall" — the caption referred to everything, and CLIP learned to encode everything. For a web search task this is fine. For a person-centric surveillance task it's a bug, because we need the model to isolate the person from the scene.

**This reframed the problem for us.** We didn't need a bigger model. We needed a model whose representations were centred on *the person*, trained on data where the caption described only the person.

That's what "person-centric foundation model" means in our context.

---

## 3. The first idea — and why it didn't work

Our source was a mix of **public image datasets** — CC3M, Visual Genome, SBU, COCO, LUP — plus our own **internal surveillance footage**. But we didn't want full-frame images as the input to captioning — we wanted just the person, so the model would describe the person and not the scene around them.

So we ran an **in-house person detector** over the source images and extracted tight person crops. Those crops — several million in total, covering both the public and the internal footage — became the actual input to the caption-generation pipeline. Where a source dataset already came as crops (e.g. LUP), we used them as-is; everywhere else we ran detection and cropped.

None of the crops had captions usable for our task. The public datasets had whole-scene captions describing everything in the original image, not the person inside it. Our surveillance footage had no captions at all. Generating captions manually at this scale was not on the table, so we needed to automate.

The first idea was natural: **use CLIP itself to generate captions.** The approach had three stages:

1. **Attribute phrase extraction** — break the description of a person into attribute phrases (upper-wear type / colour / texture / lower-wear / accessories / age / gender).
2. **Match attribute phrases with CLIP** — assemble each combination into a templated caption like *"A photo of a person wearing a blue patterned t-shirt,"* run it through frozen CLIP, score cosine similarity against the image.
3. **Sentence generation** — pick the highest-similarity caption per leaf node in the attribute tree; stitch the winners into a coherent sentence.

Two things broke this pipeline.

**The combinatorics blew up.** Even for a single attribute group like upper-wear, enumerating the captions meant **14 clothing types × 13 colours × 6 textures = 1,092 caption variants per image**, each requiring a CLIP forward pass. Multiplied across lower-wear, accessories, and the other attribute groups, and then across several million images, the compute was untenable. Transformer-based CLIP inference is not cheap at that scale.

**And more importantly — the captions it picked were wrong in a predictable way.** The colour predictions tracked the *background*, not the clothing. Concrete failure: on a crop of someone wearing a **green shirt** with a piece of **purple gym equipment** behind them, the best-matching caption came out as *"A person wearing a purple plain coloured hoodie."* Wrong colour, wrong garment type — the caption was driven almost entirely by the background object. When we reviewed mispredictions one by one, the pattern was unambiguous: if the scene had a dominant coloured object, the caption's clothing colour biased toward it.

We'd just rediscovered the reason CLIP wasn't good for our task in the first place. Using CLIP to caption a person was like asking a friend who can't tell left from right for directions — the same blind spot was going to corrupt everything downstream.

So we dropped the attribute-tree approach. The lesson: **a scene-biased model can't label a person-only attribute.** We needed a captioner that was *designed* to describe subjects in isolation.

---

## 4. Switching captioners and iterating the prompt

Vision–language models like **DeepSeek-VL2** were designed for subject-level captioning, not image-level similarity. That was the right kind of tool for the job. We switched.

**Picking the right size mattered.** We evaluated the larger DeepSeek-VL2 variant against a smaller one — roughly 7B vs 1.3B parameters. The smaller model was faster and cheaper per image, but on complex prompts it struggled — captions became non-meaningful more often, and the failure rate on long attribute lists was noticeably higher. Since caption generation was already the rate-limiting step of the whole pipeline regardless of model size, paying the extra compute for the larger model's reliability was the right trade.

A better tool doesn't automatically produce better captions, though. The first prompt we fed the VLM was naive, and the captions it produced had their own failure modes — truncation past CLIP's token budget, hallucinated attributes, activity descriptions drowning out identity cues. We needed to iterate the prompt, and we needed caption-level feedback that didn't require running a full pretraining pass every time we tried a new version.

So we built an inner loop using a stronger LLM as the judge:

```
Draft prompt
   ↓
DeepSeek-VL2 generates captions on a sample
   ↓
Gemini (stronger judge) grades caption quality at scale
   ↓
Revise prompt based on what Gemini flagged
   ↓  (repeat until captions looked consistently good)
Lock in the prompt
```

Using Gemini to judge DeepSeek-VL2's output let us evaluate hundreds of captions per round instead of eyeballing twenty. Honest caveat: Gemini is still an opinion, not ground truth — it can be consistently wrong in ways neither it nor we would catch. We'll come back to that in §10.

Seven rounds, each driven by a specific class of failure the judge flagged:

| Round | What we fixed | Why it mattered |
|---|---|---|
| 1 | Basic structured format (gender / clothes / activity) | Starting point; too generic, activity drowned out identity |
| 2 | Moved activity out of the identity sentence | Activity is noisy and unhelpful for ReID |
| 3 | Capped caption length | Long captions truncate in CLIP's 77-token budget |
| 4 | Added accessories (bag, phone, cap, glasses, carried objects) | Accessories are strong identity cues — and, as it turned out, load-bearing for weapon detection later |
| 5 | Added skin tone; capped to 3 sentences | Missing identity cue; length control |
| 6 | "If not applicable, don't mention it" | VLM was hallucinating attributes that weren't in the image |
| 7 | Final word cap + comprehensive attribute set | Fit CLIP's tokenizer budget without truncating identity info |

The round-4 decision — asking the captioner to explicitly mention carried objects — looked like a simple thoroughness choice at the time. It turned out to matter for a downstream task we hadn't built yet. By the time we trained the backbone, the embedding space had already absorbed weapon-relevant semantics, for free, because thousands of captions described people carrying weapons. If there's one thing I'd point at as the "quiet decision that compounded," it's round 4.

---

## 5. Validating the backbone after pretraining

The inner loop in §4 told us when captions *looked* good. But looking good isn't the same as producing a better *representation* — which is what we actually care about. For that we needed an outer loop, and we only got to run it once the backbone had been trained.

```
Trained backbone
   ↓
Zero-shot ReID on Market1501 (public benchmark, no fine-tuning)
   ↓
   Did Rank-1 move vs off-the-shelf CLIP?
   ↓
yes → pretraining learned something identity-relevant → scale up, build downstream heads
no  → rethink captions or pretraining recipe
```

Market1501 is a public person ReID benchmark. Running the image encoder on it with no fine-tuning and checking Rank-1 against off-the-shelf CLIP was a cheap, honest probe of whether pretraining had learned something identity-relevant.

This was a **post-training check, not a per-iteration signal.** Running the probe required a full pretraining pass — too expensive to do per prompt tweak. That's why the inner loop in §4 had to carry the fast iteration work.

**Why two loops, not one.** The inner loop gave fast caption-level feedback; the outer loop told us whether better-looking captions actually translated into a better representation. Either alone would have been worse — pretraining-only feedback is too slow to iterate prompts; caption-grading alone leaves you guessing whether good captions produce a good embedding. Same discipline as "check PCA before training" in small-data projects: **verify the representation is doing something useful before burning compute on downstream heads.**

---

## 6. Training the backbone

Once the captions were good, the training recipe itself was fairly standard — but the data pipeline before training did real work.

### Data pipeline — from raw captions to training pairs

| Stage | What we filtered / added | Pairs |
|---|---|---|
| Raw image–caption pairs after the VLM pass | (all successfully captioned crops) | ~1.3M |
| Filter blurry / low-information crops | images where identity cues weren't extractable | ~950K |
| Filter verbose / untokenisable captions | captions exceeding CLIP's 77-token budget | ~800K |
| Sentence chunking augmentation | split long captions into 2–3-sentence fragments | ~1.4M effective pairs |

Why the augmentation step matters: the VLM produced long, information-dense captions — one image, one ~200-token caption. Directly tokenising them lost everything past CLIP's 77-token limit, and the pre-filter threw away captions that didn't fit. Instead of discarding, we broke each long caption into 2–3-sentence fragments and treated each fragment as a separate text positive for the same image. Different fragments emphasised different attributes — clothing in one, accessories in another, build and posture in a third. The same image mapped to several valid captions; the embedding learned that all of them described the same identity. Acted as regularisation, and recovered most of the pairs lost to the length filter.

### Training configuration

- **Architecture:** CLIP-style contrastive, ViT-B/16 vision tower, standard text transformer.
- **Loss:** InfoNCE contrastive.
- **Initialisation:** off-the-shelf CLIP weights (no reason to throw away a strong starting point).
- **Optimiser:** AdamW with β1 = 0.9, β2 = 0.98, ε = 1e-6, weight decay 0.1.
- **Schedule:** cosine LR, base 1e-6, 2000-step warmup.
- **Batch size:** 196, AMP mixed precision.
- **Epochs:** varied across experiments; typically in the low tens.
- **Split:** 80 / 10 / 10 train / validation / test.

---

## 7. First real signal — zero-shot ReID

The zero-shot probe told us whether the pretraining had worked, isolated from any downstream fine-tuning.

| Backbone | Train data | Rank-1 | Rank-5 |
|---|---|---|---|
| Off-the-shelf CLIP (OpenAI ViT-B/16) | none (zero-shot) | 0.124 | 0.234 |
| Ours, initial pretraining | person-only captions, zero-shot for ReID | 0.258 | 0.428 |
| **Ours, HP-tuned** | **person-only captions, zero-shot for ReID** | **0.439** | **0.632** |
| *Supervised reference* (CLIP-ReID trained on Market itself) | Market train set | *0.955* | *0.984* |

**3.5× Rank-1 over off-the-shelf CLIP, without ever training on ReID data.**

The honest framing: 0.44 isn't a production number — supervised CLIP-ReID trained on Market gets 0.95. The point of the zero-shot probe was never to match the supervised ceiling. It was to isolate *how much the pretraining contributed* to identity understanding, independent of any task-specific fine-tuning.

A 3.5× gain from zero-shot pretraining alone told us the embedding had learned person-centric structure. That's the signal we were waiting for. From this point on, we knew the backbone was worth building on.

---

## 8. One model, five products

Once the backbone was good, a second story started playing out, mostly unplanned: **downstream products stopped needing their own training runs.**

```
                         ┌────────────────────┐
                         │  Person embedding   │
                         │    (frozen)         │
                         └──────┬──────────────┘
                                │
         ┌───────────┬──────────┼──────────────┬───────────┐
         ▼           ▼          ▼              ▼           ▼
       ReID     Attributes   Action    Natural-language   Weapon
                                        video search     detection
```

Each new product was a thin head on the frozen backbone:

| Product | What the head is | What the head does |
|---|---|---|
| Person ReID | cosine similarity over embeddings | retrieve same identity across cameras |
| Person attributes (colour, etc.) | small FC layers | classify clothing colour, other attributes |
| Natural-language video search | dual index (person + scene) | retrieve frames by natural-language query |
| Action / person-attribute | second-stage contrastive pretraining | extend embedding to encode actions |
| Weapon detection | ViT-L/14 action-person encoder + FC classifier (static), with a small temporal transformer added later | detect weapons in person crops; matches Gemini-level F1 and exceeds it on recall |

The economics changed meaningfully. A new attribute head went from a multi-week dataset-plus-training project to a few days of fitting FC layers on top of a frozen embedding. A new retrieval product went from running a fresh contrastive pretraining to just indexing the existing embedding.

This is what foundation models are *supposed* to do for a product surface. The second head is where the investment starts paying off. The third is where it compounds.

---

## 9. Extending to actions, then to weapons

ReID and attributes covered the *who* and the *what they look like*. For safety-critical products we needed the *what they're doing*. Gun detection became the flagship case — highest-stakes, most sensitive to false negatives, and the cleanest test of whether the foundation-model approach could earn its keep on a product that really matters.

### The gun-detection pipeline

The deployed gun detector isn't a single model; it's a **three-stage cascade**:

1. **Primary detector** — a fast **YOLO** model running at the edge. Low-latency, moderate accuracy; it flags candidate regions in live video.
2. **Secondary validator** — our cloud-side model (the subject of the rest of this section). More compute, richer representation, much higher accuracy. Think *"frontier-VLM-level accuracy, but proprietary and free at inference."*
3. **Human in the loop** — positives from the secondary validator go to an operator for confirmation before an alert fires.

The rest of this section is about stage 2 — the secondary validator. False positives from us burn operator trust; missed detections are the safety failure we can't afford.

### Why ViT-B/16 wasn't enough — the action-person foundation

Our first gun-classifier experiments used the ViT-B/16 person-foundation backbone from §6 — the same backbone that worked well for ReID, person attributes, and natural-language video search. On those tasks, B/16 was sufficient.

On gun detection it wasn't. The static classifier built on B/16 plateaued around **~64% accuracy** — nowhere near deployable for a safety-critical product. Gun detection needs a strictly richer representation than ReID does. Specifically, it needs the embedding to encode **person attributes and actions together** — *is this person in a pose consistent with holding a weapon-shaped object?* B/16's capacity couldn't carry that joint concept reliably.

So we scaled up. We trained a **second-stage foundation on a ViT-L/14 backbone** — a full additional pretraining run starting from the person-foundation weights, on **combined captions**: some describing the person's appearance, some describing the action, some describing both together. This is what we refer to throughout the deck as the action-person foundation.

With the action-person ViT-L/14 foundation underneath, the static gun classifier jumped to **~97%+ accuracy** on the same evaluation set — up from ~64% on B/16. Same pipeline, same training strategies; the only thing we changed was the backbone underneath. The capacity and task-alignment of the representation was the lever. This is one of the cleanest "the foundation matters" results we got — a 30+ percentage-point jump from getting the pretraining right.

### The static, single-frame classifier

The first production weapon classifier was single-frame. We did not plan to build a temporal model on top of it — the goal at the time was a strong per-frame detector, and that's what we focused on. Temporal showed up later, but we'll come to that.

**Architecture.** The ViT-L/14 action-person encoder with a **classification head of fully-connected layers** on top of the encoder's output embedding. No exotic adapter architecture — plain FC head. The design variable that actually mattered was **how much of the backbone we co-trained with the head**, which we treated as a tunable: some runs kept the backbone fully frozen and only trained the FC head, some unfroze the last few transformer blocks, and some experiments unfroze the backbone entirely.

**Training strategies that actually moved the needle:**
- **Freezing schedule experiments.** Swept from fully-frozen-except-head, through top-down unfreezing of transformer blocks, to a fully-unfrozen backbone. The right answer depended on how much labelled data the run had — with enough data, partial unfreezing beat head-only.
- **Architecture experiments.** FC head depth, hidden dimensions, with and without intermediate non-linearities.
- **Hyperparameter tuning.** Learning rate, regularisation strength, batch composition.

**Dataset strategies that mattered just as much:**
- **Mixing ratios.** Weapon-positive crops are rare relative to negatives; naive training collapsed to "no weapon." Deliberate positive-to-negative ratios and loss weighting fixed that.
- **Stratified train/val splits** by camera, lighting, and weapon type — so the validation set reflected deployment conditions rather than an IID slice of the training distribution.
- **Domain-shift analysis.** Tracked validation metrics on separate slices (day vs night, close vs far, different weapon types). Caught slice-level regressions that global metrics would have hidden.
- **Hard example mining.** Once the model was reasonable, cycled high-loss validation examples back into training and iterated.

### Benchmarking against general-purpose VLMs

Before committing to the custom stack, we compared it against the strongest off-the-shelf alternatives on the same secondary-validator task:
- **Gemini** in two configurations — sequence-based (one call over a sequence of frames) and multi-vote (separate calls, majority).
- **DeepSeek-VL2 Tiny** — a compact multimodal VLM.
- **Ours** — the action-person foundation (ViT-L/14) with an FC classifier on top.

Evaluation ran on a held-out set covering both static crops and motion-extended crops (to capture motion context).

**Headline result.** All three approaches clustered tightly on F1 (≈96–98%), but the precision / recall trade-offs were very different. Our model had the **highest F1 (~98.4%) and the highest recall (above 99%)**. Gemini ran precision-first — when it said "weapon," it was almost always right (precision ~99.5%) — but its recall sat around 97%. DeepSeek-VL2 Tiny sat in between.

**F1 ordering:**

```
   Ours (~98.4%)  >  Gemini voting (~98.2%)  >  DeepSeek-VL2 Tiny (~98.3%)  >  Gemini sequence (~96%)
```

The interpretation worth stating plainly:
- **General-purpose VLMs are genuinely strong.** DeepSeek-VL2 and Gemini both landed F1 around 96–98% zero-shot, with no task-specific training. That's not a trivial baseline — a frontier VLM doing surveillance weapon validation out of the box.
- **But on a safety-critical detector, recall is what matters.** A missed weapon is the dangerous failure; a false positive is a confirmed non-issue by a human reviewer. Gemini's 97% recall means ~3 out of every 100 real weapons slip through; ours closes most of that gap.
- **This is the foundation-model investment paying off concretely:** a small custom head on the right domain-specific embedding beats frontier VLMs on the metric that matters, because the foundation already encodes the relevant visual semantics and the head just learns the decision boundary.

### Temporal — which we didn't plan to build

Temporal reasoning wasn't on the roadmap. It came out of watching the static model's errors in deployment, where two failure classes kept showing up:

- **False positives on non-gun objects** — phones, tools, pen-shaped items — that at a specific frame, angle, and pose looked gun-like. A person holding a phone near their hip, or a worker gripping a wrench at the wrong angle, could trigger the classifier.
- **False negatives on real guns** — where pose, occlusion, or action made a real gun look non-gun-like for that single frame. Partial occlusion by clothing, hand orientation, motion blur, a gun drawn mid-motion.

What both classes had in common: the *single-frame evidence was ambiguous*. Over a small number of subsequent frames, the ambiguity usually resolved — the phone moved out to reveal its phone shape, the wrench didn't hold a pistol pose across frames, the partially-occluded gun came into view a frame later. The object identity was stable; our evaluation window was too narrow.

That observation — and only that observation — motivated the temporal layer. We put a small transformer (a few layers) on top of the per-frame embeddings produced by the static model — small, because the heavy visual work was already done by the static model's encoder; the temporal head only had to learn *"does this signal persist through pose, occlusion, and action changes, or does it disappear the moment conditions change?"*

```
  Frame t-2       Frame t-1         Frame t
     ↓               ↓                 ↓
  ┌────────────────────────────────────────┐
  │  Frozen action-person embedding        │
  └──────┬───────────┬──────────┬──────────┘
         ↓           ↓          ↓
       e(t-2)      e(t-1)      e(t)          (per-frame embedding)
         │           │          │
         └───────────┼──────────┘
                     ▼
          ┌─────────────────────┐
          │ Temporal transformer │
          │    (a few layers)    │
          └──────────┬───────────┘
                     ▼
               ┌───────────┐
               │classifier │ ──→ p(weapon)
               └───────────┘
```

Visually: the static model's encoder produces a per-frame embedding; the temporal transformer reads a small window of those embeddings and decides whether the weapon signal holds up across pose, occlusion, and motion changes.

**Why a thin head works at all — for the static classifier and for the temporal layer on top of it:** because of round 4 (§4). The pretraining captions already mentioned carried objects, including weapons, so both models inherited an embedding with weapon-relevant semantics baked in. The supervised heads — static and temporal alike — learn decision boundaries; they don't have to re-learn the visual concept of "gun" from scratch.

### Deployment

The classifier is explicitly designed as a **high-recall, reasonable-precision filter** — consistent with its position as stage 2 in the three-stage pipeline. The operator review step downstream handles the remaining precision cost; missing a real weapon at stage 2 is the failure mode we tune to avoid. This shapes how we set thresholds and monitor drift in production.

Field validation runs across varied deployment conditions — distances, lighting regimes, motion patterns — with continuous production performance monitoring. Specifics of the performance envelope are internal.

### Foundation-model gains across the product family

The foundation-model approach paid off across the product surface, not just on gun detection. Measured on **internal surveillance data** (not public benchmarks), every product that moved from its **pre-foundation baseline** to a **foundation-based head** improved — though the specific transition differed by product:

![Foundation-model gains across the product family](backbone_change_gains.png)

| Product | Pre-foundation baseline | Foundation-based (current) |
|---|---|---|
| Person ReID (surveillance eval) | 52% (standalone model) | **57%** — B/16 person foundation (+5 pp) |
| Attribute colour (upper + lower wear) | 72% (standalone model) | **92%** — B/16 person foundation (+20 pp) |
| Natural-language text search (R@1 / R@5) | — (capability didn't exist) | **73% / 94%** — B/16 person foundation |
| Gun static classifier | 64% (B/16 person foundation) | **97%+** — L/14 action-person foundation (+33 pp) |

Two distinct transitions, one conclusion:

- For **ReID, attribute colour, and NL text search**, the pre-foundation baseline was a task-specific standalone model. Moving those heads onto the **B/16 person foundation** lifted each one — and for NL search, it made the capability exist at all.
- For **gun detection**, the B/16 person foundation itself was the pre-upgrade baseline, and it wasn't sufficient. The **L/14 action-person foundation** (§9 above) was the upgrade that got it to deployable accuracy.

Different transitions, same pattern: **picking the right foundation for the task beats building standalone models per product.** Foundation-model investment is a strategy that repeats, not a one-off win.

**This is what motivated the next foundation: a vehicle model.** If foundation-model investments lifted every person-side product, we could run the same playbook on vehicles — make-and-model, vehicle colour, natural-language vehicle search — and expect the same compounding. See §11.

---

## 10. What we thought would matter vs. what actually did

Going in, we assumed the levers would be (a) more data, (b) bigger backbone, (c) more epochs.

In practice — this is a **narrative ranking, not a formal ablation** — what seemed to move the needle most, in order:

1. **Caption quality.** By a wide margin. The later prompt rounds produced notably more usable captions per image, and the *quality* of those captions mattered more than the raw count. This was the single biggest lever.
2. **Caption augmentation (sentence chunking).** Turned long, verbose captions into several positives per image. Net ~1.7× effective training pairs.
3. **Hyperparameter tuning.** The HP sweep alone lifted Market1501 Rank-1 from 0.258 → 0.439 — a bigger move than doubling the dataset would have been.
4. **Backbone size.** Mattered for specific downstream tasks — gun detection demanded ViT-L/14 over B/16 (§9) — but didn't move the base-foundation pretraining signal much once captions were right. The right framing isn't "bigger is always better" but "bigger when the task demands it."

The lesson that stuck with us: **the bottleneck is rarely what the naive story says it is.** Caption alignment with the task was the bottleneck for the base foundation. For specific downstream tasks (gun detection), capacity was the bottleneck. Which one binds depends on the task — that's the meta-lesson.

### What I'd do differently if I ran this again

The inner feedback loop (§4) used DeepSeek-VL2 to generate captions and Gemini as a stronger judge to grade them at scale — meaningfully better than manual inspection, but still subjective. Gemini is a good opinion, not ground truth. If it's consistently wrong about a clothing colour, or reads "maroon" as "red", both captioner and judge can share that failure mode and we'd never notice.

The piece that was missing: a **small labelled reference set** — a few hundred to ~2000 person crops with structured attribute labels (upper-wear type and colour from closed vocabularies, lower-wear, accessories, gender, age bucket, skin tone bucket, occlusion level). One-time cost of roughly a person-week.

With that in hand, every candidate prompt can be scored objectively in minutes:

- Generate captions for the reference set with the candidate prompt.
- Auto-extract structured attributes from each caption using a small LLM with constrained JSON output.
- Compare extracted attributes to ground truth → per-attribute precision, recall, and hallucination rate.

Three things this would have unlocked that Gemini-as-judge alone couldn't:

1. **Per-attribute attribution.** Round 4 becomes *"accessories recall jumped from 0.22 → 0.82"*, not *"the captions felt better."* Silent regressions in other attributes (e.g. a later round accidentally hurting gender recall) get caught instead of quietly polluting the training set.
2. **Stable across time.** Static ground-truth labels don't drift when a judge model updates. Scores from a year ago remain comparable; captioner or VLM swaps become a 10-minute evaluation instead of a pretraining run.
3. **Enables automated prompt search.** DSPy, OPRO, and APE need a stable, objective metric to optimise against. With this reference set, prompt iteration stops being a manual craft and becomes a search — the seven rounds compress to a handful of actually-informative decisions.

None of this invalidates what we did. We shipped a backbone that beats off-the-shelf CLIP 3.5× zero-shot, and the Gemini-graded loop was a real feedback loop, not guesswork. But the seven rounds would have been three, the prompt decisions would have been defensible per-attribute rather than holistic, and a future team picking this up wouldn't have to reproduce the prompt-engineering history to trust the captions.

**The underlying principle:** when building feedback loops for a representation-learning system, you want signal at every level you can afford to instrument. We had caption-level *subjective* (Gemini) and representation-level *objective* (Market1501 zero-shot). The missing rung was caption-level *objective* — cheap to build, would have made every downstream decision sharper.

---

## 11. Where we're going next

### What we've committed to

**A vehicle foundation model.**
Direct extension of the person-foundation playbook to the vehicle domain. Same contrastive pretraining recipe, same VLM-captioner-with-iterated-prompt approach, same inner-loop / outer-loop validation discipline. The vehicle foundation will be the shared backbone for vehicle **make-and-model recognition**, **vehicle colour**, and **natural-language vehicle search** — all products currently running on off-the-shelf backbones that suffer the same scene-dominant issues we first saw on persons (§2). If the gains look anything like the person-side cross-product table in §9, the ROI is already argued: one pretraining investment, three products lifted.

**Hard-negative contrastive.**
Current pretraining uses generic contrastive loss. Hard-negative mining — pairs of visually similar but different-identity crops — would sharpen the embedding in the confusable-identity region, similar in spirit to metric learning in face recognition.

### Other directions I'd explore

**Surveillance-aware captioner.**
Generic VLMs hallucinate on low-resolution crops. A captioner pretrained on surveillance imagery would fail gracefully instead of inventing attributes. Biggest remaining caption-quality lever if we wanted to push that axis further.

**Edge distillation.**
Much of the deployed pipeline runs on bandwidth-constrained edge devices. Distilling the heavier backbones to smaller students (ViT-Tiny, MobileViT, or a task-specific custom student) would let the foundation-model wins reach the edge without a compute tax.

**Cross-head consistency losses.**
If the attributes head says "red shirt" and the ReID head says "same person as X," the two should be consistent with X's attributes. A consistency loss across heads would couple them and force the backbone to encode attributes more robustly.

### Things I'd not do

- **Bigger backbone for its own sake.** We did scale from B/16 to L/14 for action-person (§9) because the task demanded it. Beyond task-driven scaling, capacity isn't the bottleneck on the base foundation.
- **More public-web data.** Dilutes the surveillance-specific signal.
- **Replace contrastive with masked-image modelling.** MIM is good for dense prediction, weak for retrieval, and retrieval dominates our product surface.

---

## 12. The short version

| | Off-the-shelf CLIP | Our person-centric model |
|---|---|---|
| Pretraining data | web-scale image–text pairs | person crops + VLM captions |
| Captioner | N/A | DeepSeek-VL2 with a 7-round tuned prompt |
| Zero-shot ReID Rank-1 (Market1501) | 0.124 | **0.439** (3.5× gain) |
| Downstream products shipped | N/A | 5 (ReID, attrs, action, NL search, weapon detection) |
| Per-product training cost | standalone model | lightweight head on a pretrained foundation |
| Weapon semantics in embedding | none | pre-seeded via captions (round 4) |
| Foundation-model gains on internal surveillance (per product) | — | ReID 52 (standalone) → 57 (B/16); Colour 72 (standalone) → 92 (B/16); NL search — → R@1 73 / R@5 94 (B/16); Gun 64 (B/16) → 97+ (L/14 action-person) |

If there are three things to take away:

1. **Off-the-shelf CLIP is scene-biased.** We fixed that by rewriting the *captions*, not the architecture.
2. **One foundation family powers five products.** The compounding value of foundation-model investment shows up at the second and third downstream, not the first — and the family includes both a B/16 backbone (ReID, attributes, NL search) and an L/14 action-person backbone (action, gun detection) because task demands differed.
3. **Weapon detection was a classification head, not a new model.** Matches frontier-VLM F1, exceeds it on recall, proprietary, and free at inference. That's what a working foundation looks like downstream.

---

# Backup — the details behind the slides

## Why CLIP-style contrastive, not a classifier?

A classifier needs a fixed label set. Our product surface has an open-ended label space — new attributes, new accessories, new query types appear continuously. Contrastive image–text learning gives us an embedding that takes *any* text query at inference time. That's what makes natural-language video search possible without per-query training.

## Why ViT-B/16 and not a ConvNet?

CLIP's original recipe was ViT. The pretrained OpenAI checkpoint is a strong initialiser we didn't want to throw away. ViT also handles variable-resolution crops gracefully via patch tokenisation, which matters when surveillance crops aren't canonical sizes. Inductive-bias-light architectures tend to generalise better on large, noisy training distributions — which is our regime.

## Why freeze the backbone for downstream heads?

Three reasons it's the right *default*:
- **Sample efficiency.** Supervised downstream datasets (weapon, attributes) are orders of magnitude smaller than the pretraining corpus. Fine-tuning the backbone on them risks overfitting and can destroy the foundation-model benefits.
- **Consistency.** A frozen backbone means all heads share the *same* embedding. The attributes head and the ReID head agree about who a person is, because they're reading the same representation.
- **Deployment simplicity.** One backbone checkpoint is served once; heads are cheap to hot-swap.

**When we broke this default.** The static weapon classifier (§9) swept the full freezing-depth spectrum, including fully-unfrozen backbone runs. For high-stakes tasks where we had enough labelled data to tune safely and the marginal gain from unfreezing outweighed the consistency / overfitting costs, partial or full unfreezing was the right call. "Freeze by default" is a principle; "always freeze" is a rule we explicitly didn't follow.

## Why zero-shot as the validation signal?

Zero-shot isolates *pretraining quality* from downstream fine-tuning. If the backbone is weak, no amount of fine-tuning on Market will surface that — fine-tuning smears everything toward the same supervised ceiling. The zero-shot delta over OTS CLIP is the honest measure of "did our pretraining teach the model something new."

## Why temporal for weapon detection but not ReID?

- ReID is cross-camera, cross-time — retrieval spans hours. Temporal modelling *within* a clip doesn't help; the embedding has to survive the gap between clips anyway.
- Weapon detection is a within-clip, within-seconds decision. Is a gun being held over this 2-second window? Frame-level ambiguity (angle, partial occlusion) doesn't average out at clip level. Temporal modelling resolves it.

## Why caption augmentation via sentence chunking?

VLM captions were long — one image, one 200-token caption. Directly tokenising loses information past CLIP's 77-token limit. Chunking into 2–3-sentence fragments gave us multiple text positives per image, each emphasising different attributes. The embedding learned that all fragments described the same identity, which made it robust to which attribute a query emphasised.

## How does this connect to other work I've done?

A connecting thread: in both this foundation-model work and a recent small-data classification project, the "obvious big model" lost to a domain-aligned choice, and the validation discipline was the same — cheap representation probe first, scale only if signal shows up. That's become a pattern I trust:

| | Small-data classifier project | Person foundation model |
|---|---|---|
| "Obvious big model" tried | generic time-series foundation model | off-the-shelf CLIP |
| Why it lost | domain + task mismatch | scene-biased captions |
| "Simple thing" that won | hand-crafted features + SVM | person-centric CLIP with VLM captions |
| Validation discipline | PCA-of-features sanity check | zero-shot ReID probe |
| Real bottleneck | acoustic similarity, not class size | caption quality, not model size |

Different scales, same instincts: **measure what the representation is actually doing before trusting the scale of the model to solve the problem.**

---

## Slide order

| # | Slide | Visual |
|---|---|---|
| 1 | Title | — |
| 2 | Where we started (§1) | web vs surveillance comparison |
| 3 | Why generic CLIP fell short (§2) | attention-bleed illustration |
| 4 | First idea that failed (§3) | attribute tree + wrong-colour example |
| 5 | Switching captioner + inner loop + 7 rounds (§4) | inner-loop diagram + 7-round table |
| 6 | Validating after pretraining (§5) | outer-loop / Market1501 probe flowchart |
| 7 | Training the backbone (§6) | pretraining pipeline diagram |
| 8 | First real signal (§7) | Market1501 Rank-1 bar chart |
| 9 | One model, five products (§8) | hub-and-spoke |
| 10 | Gun-detection pipeline + why B/16 wasn't enough (§9) | 3-stage cascade diagram + B/16 vs L/14 accuracy bar |
| 11 | Static classifier — architecture + training/dataset strategies (§9) | ViT-L/14 + FC head + freezing-depth sweep |
| 12 | Benchmark vs Gemini and DeepSeek-VL2 (§9) | F1 / recall bar chart |
| 13 | Temporal extension, driven by observed failures (§9) | failure-mode examples + temporal transformer diagram |
| 14 | Foundation-model gains across the product family + vehicle-foundation motivation (§9) | 5-product "pre-foundation vs foundation-based" bar chart on internal surveillance |
| 15 | What actually moved the needle (§10) | ranked-factors table |
| 16 | Where we go next (§11) | vehicle foundation + hard-negative (committed) + other directions |
| 17 | Short version (§12) | comparison table + 3 takeaways |
| 18 | Backup | — |

---

## What's held back from this version (external-safe)

- **Internal dataset specifics** — specific internal dataset names, per-source crop counts, and the aggregate source-corpus size. (Public source names and the downstream *filtered training-pair counts* are shown, since those describe our pipeline ratios rather than raw corpus scale.)
- **Caption-generation throughput** — exact GPU memory, time-to-caption-corpus, per-second throughput numbers.
- **Verbatim VLM prompts** — the 7-round table shows *what* each round fixed, not the prompt strings.
- **Weapon-detector benchmark numbers** — exact TP / FP / TN / FN counts, absolute Precision / Recall / F1 values, evaluation-set composition, and the specific crop-sampling methodology (the F1 *ordering* vs Gemini and DeepSeek-VL2 is shown; absolute numbers are held back).
- **Weapon-detection field metrics** — distance / lighting / IR performance numbers and field-test methodology specifics.
- **Deployment specifics** — inference serving infra, sampling configurations, threshold and drift-monitoring parameters.
- **Internal identifiers** — internal model code-names, colleagues' names, Jira IDs, runbook and monitoring URLs.

If the audience for this deck is fully internal to the team that built it, the first two bullets can be relaxed — the source-per-dataset counts, aggregate corpus size, and caption-generation throughput numbers from the internal source docs can be added in that case.
