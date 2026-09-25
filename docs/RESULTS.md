# doomLaya vs doomGemma — results

*Measurements of 25 September 2026 on soft-2 (dexmac's lab). Working document: every number comes from the files in `results/`, produced by the scripts in `src/`. Italian version: [RISULTATI.md](RISULTATI.md).*

## In short

With the same 979 examples, a 27-minute LoRA brings Gemma 4 E2B to Laya v3's level in play (6/6 vs 6/6 on MAP01) and above it on single decisions. At equal size a decoder is as fast as Laya (22 vs 23 ms); at equal result Laya is still about 3 times faster. On a new map nobody exits, not even the rule-based teacher: on this bench, generalisation cannot be measured.

## 1. Question

"System One models" (TypeSafe's Jev, closed; Convai's Laya, open, 421M, non-autoregressive encoder) promise typed decisions with calibrated probabilities in a single pass, instead of LLMs. The doomLaya project showed Laya, adapted to Doom, finishing FreeDoom MAP01 faster than Jev.

The questions:
1. does the advantage come from the "System One" **architecture**, or from having a **small model trained on the task**?
2. does a generic LLM trained on the **same data** do as well?
3. has either of them learned to **play**, or only **that level**?
4. how close are they to the **rule-based teacher** the labels come from?

## 2. Prior work (checked on 25 Sep 2026)

| work | what it does | what is missing compared with this |
|---|---|---|
| [azalio/doomLaya](https://github.com/azalio/doomLaya) | Laya v3 vs Jev on MAP01, one seed (48) | no trained LLM; they state that "the geometry of a new level was not tested" |
| [mithalouni/system-one-open](https://github.com/mithalouni/system-one-open) | open replica of Jev on Gemma 4 E2B with LoRA; Doom played zero-shot | no Doom numbers, no training on Doom, no comparison with Laya |
| [SauerkrautLM-Doom, arXiv 2604.07385](https://arxiv.org/html/2604.07385) | 1.3M model vs zero-shot LLMs on `defend_the_center` | LLMs not trained: an unbalanced comparison |
| [anth.us, "Jev vs Laya: same labels"](https://anth.us/blog/jev-vs-laya/) | trained Laya vs Jev on the same labels | sentiment classification; no generative LLMs, no Doom |
| [jwalin-shah/tensor-logic #92](https://github.com/jwalin-shah/tensor-logic/issues/92) | proposes "Laya vs fine-tuned LLMs" | proposal only, not carried out |

We found no published comparison with **Laya v3's own data, a LoRA on a generative LLM, several seeds, an unseen map, calibration, and latency on equal hardware**.

## 3. Test bench

- **Hardware**: Ryzen 5 5600X, RTX 4070 12 GB + RTX 2070 SUPER 8 GB, driver 595.91, Ubuntu 24.04. All comparable latencies measured **on the 4070**.
- **Software**:
  - doomLaya `b25edd3`, `laya` package `42626c3`;
  - checkpoint `azalio/laya-doom-v3`, sha256 of every file checked against their `SHA256SUMS`;
  - ViZDoom 1.3.1 with `freedoom2.wad`, torch 2.14 (CUDA 13.0), transformers 4.57 for Laya and 5.17 with peft 0.21 for the LoRA;
  - llama.cpp `7077abbe1`.
- **Game**: doomLaya's code unchanged. 35 tics/s, **asynchronous** requests at most every 0.5 s, the previous command stays valid for 2 s. The game **does not wait** for the model: a slow model reacts late.
- **Questions**: the agent turns the state into text (health, weapons, visible enemies with id and distance, items, doors) and asks two multiple-choice questions:
  - `command`: shoot enemy #N, pick up item #N, open the door, exit, retreat, explore, wait;
  - `weapon`.
  The executor aims, plans the path and presses the buttons. **All choices belong to the model.**
- **LLM bridge** (`serve_llm.py`): same `/predict` API as Laya's server. Each option becomes a letter, and a **grammar** allows a single token among the valid letters. Probabilities come from `top_logprobs`, renormalised over the options. It is **single-pass scoring** (prompt prefill plus one token), not free generation: no parsing errors. In `--separate` mode the two questions are sent in parallel, with the **same prompt used in training**.

## 4. Training

- **Data**: `training/v3/train.json` (979 questions) and `validation.json` (130), **the same as Laya v3**. Game seeds 42 and 43 for training, 44 for validation, **all on MAP01**. Labels assigned offline by game rules (see doomLaya's TRAINING.md).
- **LoRA** (`lora_train.py`):
  - rank 16, alpha 32, dropout 0.05, on `q,k,v,o,gate,up,down` of the language model;
  - AdamW, lr 1e-4, weight decay 0.01, accumulation 8, clip 1, seed 771;
  - **loss on the answer letter only**;
  - the best epoch by validation command + weapon is kept, as doomLaya does.

| model | trained parameters | time per epoch (4070) | best epoch | adapter |
|---|---|---|---|---|
| Gemma 4 E2B (5.1 billion total, about 2 "effective") | 24.2M | 550 s | 2 of 3 | 97 MB |
| Gemma 3 270M (268M) | 3.8M | 101 s | 5 of 5 | 15 MB |

- **Deployment**: LoRA merge, conversion to GGUF, Q8_0. doomGemma E2B is 4.9 GB and doom-270M 292 MB.
- **Laya v3** is the one published by azalio: 3 stages, training the head and then the last 3 encoder layers. We did not retrain it.
- **Two technical snags**:
  - with Gemma 4, gradient checkpointing raises a `CheckpointError`, so it is disabled;
  - E2B's `embed_tokens_per_layer` table weighs 4.4 GiB, more than the layers. It is kept in RAM, and computation stays on the 4070 alone.

## 5. Results

### 5.1 Single decisions: 130 validation questions

| model | command (80) | weapon (50) | total |
|---|---|---|---|
| Gemma 3 270M zero-shot | 0.16 | 0.02 | 0.11 |
| Laya English base (`convaiinnovations/laya`, root, @55cf4c4) | 0.41 | 0.58 | 0.48 |
| Laya typed-decisions | 0.40 | 0.60 | 0.48 |
| Gemma 4 E2B zero-shot | 0.51 | 0.78 | 0.62 |
| **Laya v3** | 0.725 | 0.98 | 0.82 |
| **doom-270M** (Gemma 3 270M + LoRA) | 0.89 | 1.00 | 0.93 |
| **doomGemma** (E2B + LoRA) | **0.96** | **1.00** | **0.98** |

Gemma E2B **without training already beats Laya without training**, and this holds for **both** Laya checkpoints. Convai's model card warns that typed-decisions is specialised on four synthetic workflows; that is why we also tested the **English base** (ModernBERT-large, 421M, sha256 `891102d3…` verified): same result (0.48). The typical error of zero-shot models is "pick up" when the right answer was "exit".

### 5.2 Calibration (130 questions)

15-bin ECE on the maximum confidence and multi-class Brier. The temperature is estimated with **2-fold cross-validation on the validation set**, so no number is measured on the data it was fitted on (`calibra.py`).

| model | raw ECE | tuned ECE | raw Brier | tuned Brier | estimated T |
|---|---|---|---|---|---|
| Gemma 3 270M zero-shot | 0.65 | 0.25 | 1.23 | 0.81 | 5–9 |
| Gemma 4 E2B zero-shot | **0.38** | 0.22 | 0.77 | 0.60 | **4.3–4.6** |
| Laya typed-decisions | 0.24 | 0.22 | 0.68 | 0.65 | 0.45–0.8 |
| Laya English base | **0.085** | 0.12 | 0.65 | 0.66 | 1.05–1.5 |
| **Laya v3** | 0.06 | 0.11 ⚠️ | 0.26 | 0.25 | 1.5 |
| doom-270M | 0.05 | 0.07 | 0.11 | 0.11 | 1.05–1.5 |
| **doomGemma** | **0.04** | **0.03** | **0.05** | **0.04** | 0.55–0.6 |

- **Laya base is well calibrated but barely discriminative.** The low ECE (0.085) comes from low, uniform confidences, not from separating right from wrong answers: accuracy is 0.48 and the Brier (0.65) is the same as typed-decisions. In play it chooses "wait" 350 times out of 350, a degenerate behaviour. **Note**: Convai's model card says the opposite, that the base "ships over-confident" (mean ECE 0.466 before tuning on their benchmark); that it is not so here is a difference to report, not a confirmation.
- **Zero-shot LLMs are very over-confident.** Gemma E2B answers with probability 1.00 even when wrong, and its confidence has to be "cooled" by a factor of about 4.4.
- **After the LoRA, doomGemma is the best-calibrated model**, even without tuning.
- ⚠️ **On Laya v3, tuning improves NLL but worsens ECE.** With 65 questions per half this is statistical noise: we report it as it is.

### 5.3 Latency on equal hardware (RTX 4070)

The same **1,416 game packets** (state plus the two real questions) are replayed in sequence. End-to-end time is measured, with 10 warm-up requests (`latenza.py`, `latenza_diretta.py`).

| model and serving method | p50 | p90 | p99 | decisions/s |
|---|---|---|---|---|
| **Laya v3**, its own server (FastAPI + PyTorch) | **23 ms** | 24 ms | 25 ms | 41.8 |
| **doom-270M, direct** PyTorch (one pass, 2 questions batched, letter logits) | **22 ms** | 23 ms | 24 ms | **44.7** |
| doom-270M via bridge + llama.cpp Q8_0 | 71 ms | 88 ms | 113 ms | 13.2 |
| doomGemma E2B via bridge + llama.cpp Q8_0 | 59 ms | 109 ms | 118 ms | 14.1 |
| Gemma E2B zero-shot, same bridge | 59 ms | 108 ms | 117 ms | 14.1 |
| doomGemma E2B, direct bf16 | 84 ms | 87 ms | 100 ms | 12.0 |

- **At equal size and equal way of computing, the decoder is as fast as the "System One" encoder** (22 vs 23 ms): the architecture gives no **intrinsic** speed advantage.
- **At equal in-game result the picture changes.** The 270M decoder that is as fast as Laya exits only 2 times out of 6; the one that matches Laya in play is doomGemma, at 59 ms through llama.cpp and 84 ms computed directly, i.e. **2.5 to 3.7 times slower**, and it weighs 4.9 GB against less than 1 GB. **With these data there is not yet a decoder that is both as fast as Laya and as good as Laya.** It remains open whether the advantage comes from the bidirectional encoder, which on this task would yield more per parameter, or simply from the fact that nobody has yet tried to shrink the decoder between 270M and 2B.
- **The bench's cadence does not stress latency.** The game asks for a decision at most every 0.5 s: between 23 and 109 ms, almost nothing changes in play. The tie does not show that latency does not matter; it says that **it is not measured here**. It would matter in a tighter control loop, for example on a robot.
- **Through llama.cpp the 270M costs as much as E2B** (71 vs 59 ms). Above a certain point the time is spent by the **serving chain** (HTTP, grammar, logprobs over a 262,000-token vocabulary), not by the model.
- **Through llama.cpp, Gemma's latency is bimodal** (about 59 and about 108 ms): it is prompt-cache reuse between similar states. In play this effect made doomGemma look faster on MAP02 (42–62 ms): **it was the cache, not the model**.
- Context note: in the first series Laya ran on the 2070S (38 ms) and looked slower than it is.

### 5.4 In play: MAP01, the training map

Skill 3, 180 s limit. **Seed 48 is not in the training data; seeds 49–53 are in neither training nor validation.**

**Seed 48, all players:**

| player | exits | kills | deaths | in-game latency (p50) |
|---|---|---|---|---|
| Laya English base | ❌ (stands still: "wait" 350 times out of 350) | 0 | 0 | 26 ms |
| Laya typed-decisions | ❌ | 8 | 1 | 39 ms (2070S) |
| Gemma 4 26B-A4B zero-shot | ❌ | 13 | 1 | 355 ms |
| Bonsai 27B ternary zero-shot | ❌ | 6 | 2 | 979 ms |
| Gemma 4 E2B zero-shot | ❌ | 10 | 0 | 93 ms |
| doom-270M | ❌ | 5 | 0 | 77 ms |
| **Laya v3** | ✅ 71.0 s | 15 | 0 | 38 ms (2070S) |
| **doomGemma** | ✅ **56.6 s** | 13 | 0 | 109 ms |
| *Jev 1.13 (doomLaya's figure, not reproduced)* | *✅ 69.0 s* | *13* | *0* | *357 ms* |
| **oracle** (the labelling rules as player) | ✅ 65.5 s | 15 | 0 | < 1 ms |

Zero-shot models have a **fixation on picking up items**: Gemma E2B chose "pick up" 271 times out of 350, Gemma 26B 274 out of 349. When they see enemies they shoot less than half of the time (Gemma 26B 41 out of 89, Bonsai 20 out of 102).

**Seeds 48–53 (6 games each):**

| player | exits | mean time | range | mean kills | deaths |
|---|---|---|---|---|---|
| **Laya v3** | 6/6 | 59.5 s | 42.9–71.0 s | 13.5 | 0 |
| **doomGemma** | 6/6 | 60.0 s | 56.6–64.2 s | 13.0 | 0 |
| doom-270M | **2/6** | 68.5 s (when it exits) | 62.8–74.1 s | 9.7 | 0 |
| **oracle** | 6/6 | **81.6 s** | 64.5–124.9 s | 16.0 | 1 |

**A tie in play between Laya v3 and doomGemma.** Laya is more variable, doomGemma more consistent. The 56.6 vs 71.0 of seed 48 was partly luck.

**The rule-based teacher as a reference.** The oracle is the `gold` function of `training/build_dataset.py`, copied verbatim, which receives the same packet and the same raw state the labels were produced from and goes through the same executor at the same cadence (`oracolo.py`). It exits 6 times out of 6, in 81.6 s on average. The oracle **does not optimise time-to-exit**: its rules include picking up useful items, and indeed it picks up more (17.7 on average, up to 31) and kills more (16.0 vs about 13). **The models exit earlier than the teacher (about 60 s vs 82) because they imitate it imperfectly and pick up less.** It is a consequence of imperfect imitation, not a merit of the models. On time-to-exit alone, therefore, 60 s is not the ceiling of the bench.

### 5.5 In play: MAP02, an unseen map (seeds 48–50, 300 s limit)

| player | exits | kills (per seed) | deaths (per seed) |
|---|---|---|---|
| Laya typed-decisions | 0/3 | 57 · 46 · 38 | 11 · 7 · 6 |
| Gemma E2B zero-shot | 0/3 | 46 · 42 · 49 | 3 · 3 · 3 |
| Laya v3 | 0/3 | 27 · 22 · 36 | 2 · 1 · 3 |
| doomGemma | 0/3 | 11 · 21 · 26 | 0 · 1 · 2 |
| doom-270M | 0/3 | 47 · 18 · 15 | 4 · 1 · 1 |
| **oracle** | **0/3** | 26 · 13 · 20 | 2 · 0 · 1 |

**Nobody exits, 0 out of 18, oracle included.** The "exit" option was **always** among the choices (the navigator knows the exit position from the map) and the models did choose it (doomGemma 69 times, Laya v3 38 in seed 48). The problem is downstream: **the oracle chooses "open the door" about 500 times out of 580** and the executor spends 7,600–9,500 tics in front of the door of **sector 37** (coordinates 508, −764). **Verified by reading `freedoom2.wad`:** all 6 linedefs of that door have special **27, "DR, yellow key"**: it opens with USE only if the player holds the yellow key. The yellow key exists on MAP02, at coordinates (−256, 800), far from the door. The chain gets stuck for three combined reasons:
  - doomLaya's sensor treats key doors (specials 26, 27, 28 and 32–34) as "openable doors" too, and offers them as the `open_door` option;
  - the teacher's rule opens any closed door within 4 m;
  - keys enter the options only when seen within 12 m, and this one is far away and out of sight.

On MAP02 **the whole chain fails** (rules, executor, state representation), not the models' ability to generalise: **here generalisation cannot be measured.** *(Kills accumulate across deaths: whoever dies often respawns and piles up more.)*

### 5.6 How often the executor rejects commands ("rails")

The share of time in which the executor **cannot** execute the model's command: path blocked or target unreachable (`binari.py`, from the telemetry).

| player | MAP01 | MAP02 |
|---|---|---|
| Bonsai zero-shot | 72% | — |
| **doom-270M** | **60%** (53% blocked) | **76%** |
| Laya typed-decisions | 43% | 25% |
| Gemma 26B zero-shot | 39% | — |
| Gemma E2B zero-shot | 37% | 26% |
| doomGemma | 22% | 8.5% |
| Laya v3 | 13% | 9.7% |
| oracle | 20.5% | 3.3% |

This explains doom-270M: **it has high validation accuracy (0.93) but keeps choosing actions that are impossible at that moment**, and in play it exits only 2 times out of 6. Questions taken one at a time do not measure consistency from one moment to the next.

## 6. What can be claimed

1. **Task training matters more than architecture.** Zero-shot everyone loses, and Gemma E2B is actually better than Laya. With the same 979 examples, a 27-minute LoRA on a consumer GPU brings a generic LLM to the specialised model's level in play, and above it on single decisions and calibration.
2. **Latency: at equal size no intrinsic advantage of the architecture** (270M decoder at 22 ms, Laya at 23). **At equal in-game result, however, Laya is about 3 times faster** than the decoder that matches it, because it gets there with a smaller model. On this bench the difference does not weigh, because decisions are requested every 0.5 s.
3. **Single-question accuracy does not predict play.** doom-270M scores 0.93 on validation and finishes 2 games out of 6; Laya v3 scores 0.82 and finishes all of them.
4. **Calibration of zero-shot LLMs is not reliable** (ECE 0.38, confidence 1.00 even when wrong). After task training it becomes the best of the group.
5. **The models exit earlier than the rule-based teacher** (about 60 s vs 82) **because they imitate it imperfectly and pick up less.** The teacher does not optimise time-to-exit: on time alone, 60 s is not the ceiling of the bench.
6. **Generalisation to new maps is not measurable on this bench.** On MAP02 nobody exits, **not even the rule-based teacher**: the rules + executor + state chain gets stuck in front of a **yellow-key door** (verified in the WAD), because the key is far away and key doors are offered as openable. To really measure generalisation one would need an action set able to look for keys and a sensor that does not offer doors that cannot be opened; only then does it make sense to ask whether the models imitate the teacher there too.

## 7. Limitations (to be stated in the article)

- Training and validation on **MAP01** only. MAP02 (3 seeds) turned out to be unsolvable by the whole chain, oracle included; other maps were not tested.
- The labels come from a **rule-based teacher**, not from human play: the models learn to imitate those rules, and the comparison measures how well they do it.
- **130** validation questions: for calibration each half has 65, and the decimals should be taken with caution.
- **Laya v3** was trained by others, in 3 stages and with a method different from ours. We did not try to improve it.
- **Jev** numbers come from doomLaya and are not reproduced (paid API).
- The 270M's **"direct" latency** is measured with our own PyTorch script, not with a production server. Laya's goes through its HTTP server: the comparison is conservative in Laya's favour.
- The bridge reads only the **top 20 logprobs**. Options outside that list receive a minimum probability (1e-4). It had no effect on accuracy and calibration, but it should be said.
- The **first zero-shot E2B game** on MAP01 used the prompt with both questions together. Later measurements use separate questions, as in training.
- **Gemma 3 270M** comes from the unsloth mirror (`model.safetensors` sha256 `700b710a…`). Google's original requires an account and we could not compare the hashes.
- All game times are **game time**; the game runs in real time.

## 8. Available material

- `media/doomLaya_vs_doomGemma.gif`: 5 s side by side, MAP01 seed 48, seconds 4.5–9.5, with the probability panel.
- `results/`: raw results (calibration, latency, MAP01/MAP02 games, oracle, Laya base, training history).
- `src/`: all scripts; `patches/doomLaya-agent.patch`: the 6 lines changed in doomLaya's `agent.py` (5 added, 1 modified) (`--model llm` and `--model oracolo`).
- LoRA adapters on Hugging Face: [dexmac/doomgemma-e2b-lora](https://huggingface.co/dexmac/doomgemma-e2b-lora) and [dexmac/doom-gemma3-270m-lora](https://huggingface.co/dexmac/doom-gemma3-270m-lora).
