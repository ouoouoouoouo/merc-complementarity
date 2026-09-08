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
| `ceiling` | fraction at least one modality gets right — a bound on *any* fusion method |
| `2fault` | both wrong; fusion cannot reach these at all |
| `Q` | Kuncheva & Whitaker Q-statistic over the two correctness vectors |

Plus a per-emotion breakdown of what the joint probe fixes and breaks — often
the most informative output, e.g. audio recovering *angry* while costing
*neutral*.

**`rescue` and `damage` are the ones to read.** They route through the joint
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
comparable. Before comparing features across repos, build one common manifest
(`utt_id,label,split`) and point every source at it.

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

**One split is not evidence.** IEMOCAP test splits are ~1.2k utterances and
speaker-dependent effects are large. A 1-2 pp difference between pairs is
inside the noise. Treat a single run as a screen, and confirm anything you
would put in a table with leave-one-session-out.

## Workflow

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
- `kind: npy_dir` — a tree of `<utt_id>.npy`, indexed by filename stem; the
  split/label directories in the path are ignored, because ground truth comes
  from the manifest
- `pool` — for 2-D `(T, D)` features: `mean_nonzero` (default, ignores
  zero-padding), `mean`, `max`, or `mean_std` (mean+std functionals). 1-D
  features pass through untouched.
