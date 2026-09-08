# merc-complementarity

How complementary is a (text, audio) feature pair for emotion recognition?

Measured directly, without training a fusion model, so a pair costs seconds
instead of an hour of GPU time. The point is to sweep many pairs cheaply, then
spend the GPU only on the ones that look promising — and to know in advance
which ones cannot pay off.

It reads cached feature files. It does not extract features and does not depend
on any of the pipeline repos.

```bash
pip install numpy scikit-learn pyyaml torch   # torch only for .pt sources
python complementarity.py --spec sources.yaml
```

## What it reports

**Layer 1 — oracle gain.** The same probe on text alone, audio alone, and the
two concatenated.

```
gain = weighted_F1(concat) - weighted_F1(better single modality)
```

With a bootstrap CI and an exact McNemar test over the discordant utterances. A
CI straddling zero means this pair has nothing to offer a fusion model, however
good either modality looks on its own.

**Layer 2 — error structure.** Oracle gain hides the shape of the
complementarity. Two pairs with the same gain can differ completely:

| | meaning |
|---|---|
| `rescue` | of the utterances the better single modality gets **wrong**, the fraction the joint probe recovers |
| `damage` | of those it gets **right**, the fraction the joint probe loses |
| `headrm` | `ceiling - accuracy of the better single modality`: the room that exists |
| `realzd` | fraction of that room the joint probe actually took — **the number to compare across pairs** |
| `ceiling` | fraction at least one modality gets right — bounds *decision-level* fusion only |
| `2fault` | both wrong; fusion cannot reach these at all |
| `Q` | Kuncheva & Whitaker Q-statistic over the two correctness vectors |

Plus a per-emotion breakdown of what the joint probe fixes and breaks — often
the most informative output, e.g. audio recovering *angry* while costing
*neutral*.

**Compare pairs on `realzd`, not on raw gain.** A pair whose unimodal probes
are weak has far more room to improve, so absolute gain rewards weak features.
`realzd` divides by the room that exists.

**`realzd > 1` is a finding, not a bug.** `ceiling` bounds decision-level
fusion — any rule that picks or weights the two predictions. Feature-level
concatenation sees the representations themselves and can be right where both
unimodal probes were wrong. Exceeding the ceiling means the pair carries
synergy: information present only in the two together, which no decision-level
combiner can reach.

**`rescue` and `damage` are the ones to read alongside it.** They route through the joint
probe, so they measure information that actually combines. `rescue` high with
`damage` near zero is real complementarity; the two roughly equal is a pair
that reshuffles errors without adding anything.

**`Q` and `disagreement` measure prediction diversity, not complementary
information** — a trap worth stating because the ensemble-diversity literature
invites the confusion. Two probes carrying *identical* information still
disagree wherever each independently guesses the part neither can see, which
pushes Q towards 0 and makes a redundant pair look complementary. On a
synthetic pair built to be exactly redundant, Q reads −0.04 vs −0.02 for a
fully complementary one — useless — while rescue/damage separate them cleanly
(1.00/0.00 vs 0.28/0.33). Q is kept as description only.

## Caveats that matter more than the code

**The label sets do not match across repos.** Bi-LSTM keeps `{ang, exc, neu,
sad}` — `exc` is its own class and `hap` is dropped entirely. merits-l-text and
merits-l-llama use `{angry, happy, sad, neutral}` with `hap` and `exc` merged
into happy. These are different 4-class problems and their numbers are not
comparable. `build_manifest.py` resolves this by taking the merits convention
as canonical and emitting one `utt_id,label,split` CSV that every source is
joined onto. The `hap` utterances it asks for have no Bi-LSTM features at all —
`preprocess.py` skipped them — so `Bi-LSTM/extract_for_manifest.py` extracts
just those into a flat top-up tree, which is listed as a second `roots` entry
beside the original.

**The splits do not match either.** Bi-LSTM trains on Sessions 1-4 and tests on
Session 5. The merits manifests use their own train/val/test. Same fix: one
manifest decides, and the tool joins everything onto it by `utt_id`.

**Pooling handicaps sequence features.** Mean-pooling a (500, 300) GloVe
sequence produces a bag of words, discarding the word order the Bi-LSTM exists
to model. A low GloVe row therefore does not show that GloVe is a weak feature;
it shows that mean-pooled GloVe is. To compare fairly against a contextual
encoder, feed in the *trained* utterance embedding instead — Bi-LSTM's
`TextBiLSTM.get_features()` (512-d) and `CBLA.get_features()` (256-d) both
already return one — and dump them as a `{utt_id: vector}` `.pt`.

**Dimensionality is a confound.** A 4096-d Llama feature and a 68-d functionals
vector do not give a linear probe the same capacity, so "text has more unique
information" can be pure dimensionality. `--dim` PCAs both sides to a common
size (default 128, fit on train only). `--dim 0` turns that off, and then the
comparison across pairs is not controlled — the header says so when you do.

**`realzd` gets unstable when `headrm` is small.** It is a ratio, so a pair
whose unimodal probes are already close to the ceiling divides a small number
by a small number. Read it next to `headrm`, and treat a large `realzd` with a
tiny `headrm` as noise rather than a finding.

**Two sources of variance, and the CI only covers one.** The bootstrap CI
covers test-set sampling. It does not cover probe training — for the MLP, its
initialisation and its own early-stopping split. `--seeds N` repeats every pair
with seeds 0..N-1 and reports the spread of the gain, which is often the larger
of the two. A single-seed number is one draw; report the spread.

**One split is not evidence.** IEMOCAP test splits are ~1.2k utterances and
speaker-dependent effects are large. A 1-2 pp difference between pairs is
inside the noise. Treat a single run as a screen, and confirm anything you
would put in a table with leave-one-session-out.

## Workflow

0. Build the shared manifest, then top up whatever features it is missing:

   ```bash
   python build_manifest.py --merits-manifests ~/merits-l-llama/data/manifests/iemocap        --out manifests/iemocap_common.csv        --check-npy ~/Bi-LSTM/data/iemocap/processed_paper/audio
   ```

1. Run every pair with the linear probe to see whether the gains spread at all.
   If every pair lands within ~1 pp, there is no story here and it is worth
   knowing that on day one.
2. Re-run with `--probe mlp`. A pair whose complementarity only appears under
   the MLP is non-linearly complementary — worth saying explicitly.
3. For pairs that look interesting, run the real fusion pipeline. The
   contribution is whether these cheap numbers *predict* the fusion result. If
   they do, that prediction is a more useful result than another 0.5 pp.

## Spec format

See `sources.yaml`. Each entry is a feature source:

- `kind: pt_dict` — a `{utt_id: tensor}` `.pt` (the merits-l-* convention)
- `kind: npy_dir` — a tree of `<utt_id>.npy` under `root`, or several trees
  under `roots` (earlier ones win), indexed by filename stem; the split/label
  directories in the path are ignored, because ground truth comes from the
  manifest
- `pool` — for 2-D `(T, D)` features: `mean_nonzero` (default, ignores
  zero-padding), `mean`, `max`, or `mean_std` (mean+std functionals). 1-D
  features pass through untouched.
