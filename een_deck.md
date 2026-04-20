# EEN Person Foundation Model — how it came together

> **Note on this deck**
> This is a walkthrough of how the person-centric foundation model developed at EEN actually came together — the starting problem, the dead ends, the pivots, what ended up mattering, and where weapon detection fits into the story.
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

When we probed what CLIP was actually doing on our crops, a consistent pattern showed up: **its representations were scene-dominant, not person-dominant.**

Concretely — if we asked it to tell us what colour shirt someone was wearing, it often gave us the colour of the wall behind them. Same for vehicles: tell me about the car, get a description biased by the street colour.

Why? Because CLIP was trained on *full images* paired with captions describing *the whole scene*. "A person in a blue shirt standing in front of a red wall" — the caption referred to everything, and CLIP learned to encode everything. For a web search task this is fine. For a person-centric surveillance task it's a bug, because we need the model to isolate the person from the scene.

**This reframed the problem for us.** We didn't need a bigger model. We needed a model whose representations were centred on *the person*, trained on data where the caption described only the person.

That's what "person-centric foundation model" means in our context.

---

## 3. The first idea — and why it didn't work

Our corpus was a mix of **public person-centric image datasets** — CC3M, Visual Genome, SBU, COCO, LUP — aggregated with **internal surveillance datasets**. Several million person crops in total. None of it had usable captions for a person-only task; existing captions in the public sets described the whole scene.

Generating captions manually at this scale was not on the table. We needed to automate.

The first idea was natural: **use CLIP itself to generate captions.** The approach had three stages:

1. **Attribute phrase extraction** — break the description of a person into attribute phrases (upper-wear type / colour / texture / lower-wear / accessories / age / gender).
2. **Match attribute phrases with CLIP** — assemble each combination into a templated caption like *"A photo of a person wearing a blue patterned t-shirt,"* run it through frozen CLIP, score cosine similarity against the image.
3. **Sentence generation** — pick the highest-similarity caption per leaf node in the attribute tree; stitch the winners into a coherent sentence.

Two things broke this pipeline.

**The combinatorics blew up.** Even for a single attribute group like upper-wear, enumerating the captions meant **14 clothing types × 13 colours × 6 textures = 1,092 caption variants per image**, each requiring a CLIP forward pass. Multiplied across lower-wear, accessories, and the other attribute groups, and then across several million images, the compute was untenable. Transformer-based CLIP inference is not cheap at that scale.

**And more importantly — the captions it picked were wrong in a predictable way.** The colour predictions tracked the *background*, not the clothing. Concrete failure: on a crop of someone wearing a white hoodie standing against a purple wall, the best-matching caption came out as *"A person wearing a purple plain coloured hoodie."* When we reviewed mispredictions one by one, the pattern was unambiguous: if the background had a dominant colour (wall, vehicle, sky), the caption's shirt colour biased toward it.

We'd just rediscovered the reason CLIP wasn't good for our task in the first place. Using CLIP to caption a person was like asking a friend who can't tell left from right for directions — the same blind spot was going to corrupt everything downstream.

So we dropped the attribute-tree approach. The lesson: **a scene-biased model can't label a person-only attribute.** We needed a captioner that was *designed* to describe subjects in isolation.

---

## 4. Switching captioners and iterating the prompt

Vision–language models like **DeepSeek-VL** were designed for subject-level captioning, not image-level similarity. That was the right kind of tool for the job. We switched.

**Picking the right size mattered.** We evaluated the 7B variant (`deepseek-vl-7b-chat`) against the smaller 1.3B (`deepseek-vl-1.3b-chat`). The smaller model was faster and cheaper per image, but on complex prompts it struggled — captions became non-meaningful more often, and the failure rate on long attribute lists was noticeably higher. Since caption generation was already the rate-limiting step of the whole pipeline regardless of model size, paying the extra compute for the 7B model's reliability was the right trade.

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
- **Batch size:** 196, AMP mixed precision, 32 epochs.
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
| Weapon detection | ViT-L/14 + adapters (static), then a temporal transformer on top | detect weapons in person crops; beats Gemini / DeepSeek-VL2 on F1 |

The economics changed meaningfully. A new attribute head went from a multi-week dataset-plus-training project to a few days of fitting FC layers on top of a frozen embedding. A new retrieval product went from running a fresh contrastive pretraining to just indexing the existing embedding.

This is what foundation models are *supposed* to do for a product surface. The second head is where the investment starts paying off. The third is where it compounds.

---

## 9. Extending to actions, then to weapons

ReID and attributes covered the *who* and the *what they look like*. For safety-critical products we needed the *what they're doing*.

So we did a **second pretraining stage** — starting from the person backbone, adding action and interaction supervision on top. Same contrastive objective, captions extended with activity and object-interaction descriptions. The result is the action-person backbone that weapon detection sits on.

Weapon detection is where the whole stack had to prove itself. It's the highest-stakes downstream: false positives burn operator trust, missed detections cost safety. If the foundation-model approach was going to earn its keep on a product that matters, this was the test.

### Step 1 — a static, single-frame weapon classifier

Before we added any temporal reasoning, we built the minimum viable version: a **single-frame** classifier on top of the action-person embedding. The logic was simple — if a thin head on the foundation backbone couldn't beat strong VLM baselines on individual frames, temporal wasn't going to save us.

**Architecture.** A **ViT-L/14** vision encoder derived from the action-person foundation, with **adapter layers** inserted for parameter-efficient fine-tuning on the weapon-detection task. The main backbone stayed frozen; adapters and the classification head carried the task-specific capacity. Standard "small-tunable-capacity, large-frozen-backbone" pattern — consistent with the foundation-model reuse principle (§B3).

**Training strategies that actually moved the needle:**
- **Freezing schedule experiments.** Started with everything frozen except adapters and head. When adapter-only capacity bottlenecked, selectively unfroze the last few transformer blocks.
- **Architecture experiments.** Adapter placement (every layer vs every other vs last-K blocks), bottleneck dimensions, head depth, with and without residual connections.
- **Hyperparameter tuning.** Learning rate, adapter bottleneck size, regularisation strength, batch composition.

**Dataset strategies that mattered just as much:**
- **Mixing ratios.** Weapon-positive crops are rare relative to negatives; naive training collapsed to "no weapon." Deliberate positive-to-negative ratios and loss weighting fixed that.
- **Stratified train/val splits** by camera, lighting, and weapon type — so the validation set reflected deployment conditions rather than an IID slice of the training distribution.
- **Domain-shift analysis.** Tracked validation metrics on separate slices (day vs night, close vs far, different weapon types). Caught slice-level regressions that global metrics would have hidden.
- **Hard example mining.** Once the model was reasonable, cycled high-loss validation examples back into training and iterated.

### Benchmarking against general-purpose VLMs

Before committing to the custom stack, we compared it against the strongest off-the-shelf alternatives on the same secondary-validator task:
- **Gemini** in two configurations — sequence-based (one call over a sequence of frames) and multi-vote (separate calls, majority).
- **DeepSeek-VL2 Tiny** — a compact multimodal VLM.
- **Ours** — action-person foundation + ViT-L/14 + adapters.

Evaluation ran on a held-out set covering both static crops and motion-extended crops (to capture motion context).

**Headline result.** Our model had the **best F1 and the highest recall** of any approach tested. Gemini led on precision — it only said "weapon" when highly confident — but its recall lagged, which matters for a safety-critical detector where missed weapons are the dangerous failure mode. DeepSeek-VL2 sat between the two.

**F1 ordering:**

```
   Ours   >   Gemini (voting)   >   DeepSeek-VL2 Tiny   >   Gemini (sequence)
```

The interpretation worth stating plainly:
- General-purpose VLMs are genuinely strong. DeepSeek-VL2 and Gemini both hit F1 in the mid-to-high 90s zero-shot. That's not a trivial baseline.
- On a narrow, high-stakes domain task, a **small custom head on the right foundation backbone** beats them — because the foundation already encodes the relevant semantics and the adapter just has to learn the decision boundary, not re-learn the visual concept.
- This is the foundation-model investment paying off concretely: **beating frontier VLMs with a compact model trained on a domain-specific embedding.**

### Step 2 — adding temporal reasoning

Once the static model was strong, we added a temporal layer to resolve single-frame ambiguity. A phone, a tool, and a pistol can look similar at one angle and one moment. Persistence across frames disambiguates — if an object holds its shape relative to the hand across multiple frames, the signal is real; if it flickers in one frame and disappears, it probably isn't.

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
          │  (2 layers, small)   │
          └──────────┬───────────┘
                     ▼
               ┌───────────┐
               │classifier │ ──→ p(weapon)
               └───────────┘
```

The temporal transformer operates on sequences of per-frame embeddings from the static model. Small — 2 layers — because the heavy visual work is already done by the frozen backbone; the temporal head just needs to learn "does this signal persist."

**Why a thin head still works:** because of round 4 (§4). The pretraining captions already mentioned carried objects, including weapons, so the embedding wasn't a blank slate — it had weapon-relevant semantics baked in from the pretraining corpus. The supervised head learns the decision boundary, not the visual concept.

### Deployment

**Human-in-the-loop.** The classifier surfaces candidates; operators confirm. The model is designed to be a high-recall, reasonable-precision filter on a firehose, not an autonomous alarm. This shapes how we set thresholds and monitor drift.

Field validation runs across varied deployment conditions — distances, lighting regimes, motion patterns — with continuous production performance monitoring. Specifics of the performance envelope are internal.

---

## 10. What we thought would matter vs. what actually did

Going in, we assumed the levers would be (a) more data, (b) bigger backbone, (c) more epochs.

Ranked by measured contribution, what actually moved the needle:

1. **Caption quality.** By a wide margin. Round-7 prompts produced roughly 2× more usable captions per image than round-1 — and the *quality* of those captions mattered more than the quantity. This was the single biggest lever.
2. **Caption augmentation (sentence chunking).** Turned unusable verbose captions into several positives per image. Net ~1.7× effective training pairs.
3. **Hyperparameter tuning.** The HP sweep alone lifted Market1501 Rank-1 from 0.258 → 0.439 — a bigger move than doubling the dataset would have been.
4. **Backbone size.** In our regime, didn't matter nearly as much as 1–3.

The lesson that stuck with us: **the bottleneck is rarely what the naive story says it is.** Caption alignment with the task was the bottleneck. Capacity, data volume, and training compute were all cheaper than they looked in hindsight, because they weren't the limiting factor.

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

In rough order of expected impact to effort:

**Surveillance-aware captioner.**
Generic VLMs hallucinate on low-resolution crops. A captioner pretrained on surveillance imagery would fail gracefully instead of inventing attributes. Biggest remaining caption-quality lever.

**Hard-negative contrastive.**
Current pretraining uses generic contrastive loss. Hard-negative mining — pairs of visually similar but different-identity crops — would sharpen the embedding in the confusable-identity region. Similar in spirit to metric learning in face recognition.

**Edge distillation.**
Surveillance runs on bandwidth-constrained edge devices. ViT-B/16 is too heavy to deploy everywhere. Distilling to a smaller student (ViT-Tiny, MobileViT) would let the foundation-model wins reach production without a compute tax.

**Cross-head consistency losses.**
If the attributes head says "red shirt" and the ReID head says "same person as X," the two should be consistent with X's attributes. Adding a consistency loss couples the heads and forces the backbone to encode attributes more robustly.

**What we're not doing:**
- Bigger backbone — capacity isn't the bottleneck.
- More public-web data — dilutes the surveillance-specific signal.
- Replacing contrastive with masked-image modelling — MIM is good for dense prediction, weak for retrieval, and retrieval dominates our product surface.

---

## 12. The short version

| | Off-the-shelf CLIP | Our person-centric model |
|---|---|---|
| Pretraining data | web-scale image–text pairs | person crops + VLM captions |
| Captioner | N/A | VLM with 7-round tuned prompt |
| Zero-shot ReID Rank-1 (Market1501) | 0.124 | **0.439** (3.5× gain) |
| Downstream heads shipped | N/A | 5 (ReID, attrs, action, search, weapon) |
| Per-product training cost | standalone | thin head on frozen backbone |
| Weapon semantics in embedding | none | pre-seeded via captions |

If there are three things to take away:

1. **Off-the-shelf CLIP is scene-biased.** We fixed that by rewriting the *captions*, not the architecture.
2. **One embedding powers five products.** The compounding value of foundation-model investment shows up at the second and third downstream, not the first.
3. **Weapon detection was a thin head, not a new model.** That's what a working foundation looks like.

---

# Backup — the details behind the slides

## Why CLIP-style contrastive, not a classifier?

A classifier needs a fixed label set. Our product surface has an open-ended label space — new attributes, new accessories, new query types appear continuously. Contrastive image–text learning gives us an embedding that takes *any* text query at inference time. That's what makes natural-language video search possible without per-query training.

## Why ViT-B/16 and not a ConvNet?

CLIP's original recipe was ViT. The pretrained OpenAI checkpoint is a strong initialiser we didn't want to throw away. ViT also handles variable-resolution crops gracefully via patch tokenisation, which matters when surveillance crops aren't canonical sizes. Inductive-bias-light architectures tend to generalise better on large, noisy training distributions — which is our regime.

## Why freeze the backbone for downstream heads?

Three reasons:
- **Sample efficiency.** Supervised downstream datasets (weapon, attributes) are orders of magnitude smaller than the pretraining corpus. Fine-tuning the backbone on them overfits and destroys the foundation-model benefits.
- **Consistency.** A frozen backbone means all heads share the *same* embedding. The attributes head and the ReID head agree about who a person is, because they're reading the same representation.
- **Deployment simplicity.** One backbone checkpoint is served once; heads are cheap to hot-swap.

## Why zero-shot as the validation signal?

Zero-shot isolates *pretraining quality* from downstream fine-tuning. If the backbone is weak, no amount of fine-tuning on Market will surface that — fine-tuning smears everything toward the same supervised ceiling. The zero-shot delta over OTS CLIP is the honest measure of "did our pretraining teach the model something new."

## Why temporal for weapon detection but not ReID?

- ReID is cross-camera, cross-time — retrieval spans hours. Temporal modelling *within* a clip doesn't help; the embedding has to survive the gap between clips anyway.
- Weapon detection is a within-clip, within-seconds decision. Is a gun being held over this 2-second window? Frame-level ambiguity (angle, partial occlusion) doesn't average out at clip level. Temporal modelling resolves it.

## Why caption augmentation via sentence chunking?

VLM captions were long — one image, one 200-token caption. Directly tokenising loses information past CLIP's 77-token limit. Chunking into 2–3-sentence fragments gave us multiple text positives per image, each emphasising different attributes. The embedding learned that all fragments described the same identity, which made it robust to which attribute a query emphasised.

## How does this connect to other work I've done?

A connecting thread: in both the EEN foundation model work and a recent small-data classification project I did, the "obvious big model" lost to a domain-aligned choice, and the validation discipline was the same — cheap representation probe first, scale only if signal shows up. That's become a pattern I trust:

| | Small-data classifier project | EEN person foundation model |
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
| 10 | Action extension + static gun classifier (§9, step 1) | ViT-L/14 + adapter architecture |
| 11 | Benchmark vs Gemini and DeepSeek-VL2 (§9) | F1 ordering bar chart |
| 12 | Temporal gun model + deployment (§9, step 2) | temporal transformer diagram |
| 13 | What actually moved the needle (§10) | ranked-factors table |
| 14 | Where we go next (§11) | effort-vs-impact quadrant |
| 15 | Short version (§12) | comparison table + 3 lines |
| 16 | Backup | — |

---

## What's held back from this version (external-safe)

- **Internal dataset specifics** — specific internal dataset names, per-source crop counts, and the aggregate source-corpus size. (Public source names and the downstream *filtered training-pair counts* are shown, since those describe our pipeline ratios rather than raw corpus scale.)
- **Caption-generation throughput** — exact GPU memory, time-to-caption-corpus, per-second throughput numbers.
- **Verbatim VLM prompts** — the 7-round table shows *what* each round fixed, not the prompt strings.
- **Weapon-detector benchmark numbers** — exact TP / FP / TN / FN counts, absolute Precision / Recall / F1 values, evaluation-set composition, and the specific crop-sampling methodology (the F1 *ordering* vs Gemini and DeepSeek-VL2 is shown; absolute numbers are held back).
- **Weapon-detection field metrics** — distance / lighting / IR performance numbers and field-test methodology specifics.
- **Deployment specifics** — inference serving infra, sampling configurations, threshold and drift-monitoring parameters.
- **Internal identifiers** — internal model code-names, colleagues' names, Jira IDs, runbook and monitoring URLs.

If the audience for this deck is fully internal to EEN, the first two bullets can be relaxed — tell me and I'll add the source-per-dataset counts, aggregate corpus size, and caption-generation time numbers from the Confluence source page.
