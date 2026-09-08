"""Measure how complementary a (text, audio) feature pair is for emotion.

Two layers, both cheap enough to sweep many pairs before committing GPU time to
a fusion model:

  Layer 1 - oracle gain.  Probe each modality alone and concatenated, on the
  same split with the same probe. complementarity = concat - best unimodal.
  This is the quantity that predicts whether fusion will pay off.

  Layer 2 - error-level structure.  Two pairs can share an oracle gain and
  still behave differently: one where audio rescues the utterances text gets
  wrong, one where both fail together. Reports rescue/damage rates through the
  joint probe, the ceiling that bounds decision-level fusion, how much of the
  available headroom was realized, and a per-emotion breakdown of what the
  joint probe fixes and breaks.

Neither layer touches a fusion model, so a pair costs seconds instead of an
hour. See README.md for the label-set and split caveats, which matter more than
anything in this file.

Usage:
    python complementarity.py --spec sources.yaml
    python complementarity.py --spec sources.yaml --probe mlp --dim 64
    python complementarity.py --spec sources.yaml --pairs glove:handcrafted
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------
# Feature sources
# --------------------------------------------------------------------------


@dataclass
class Source:
    name: str
    modality: str            # "text" or "audio"
    vectors: Dict[str, np.ndarray]

    @property
    def dim(self) -> int:
        return next(iter(self.vectors.values())).shape[0]


def _pool(arr: np.ndarray, how: str) -> np.ndarray:
    """Reduce a (T, D) sequence to (D,). 1-D input passes through."""
    if arr.ndim == 1:
        return arr.astype(np.float64)
    if arr.ndim != 2:
        raise ValueError(f"expected 1-D or 2-D feature, got shape {arr.shape}")
    if how == "mean":
        return arr.mean(axis=0).astype(np.float64)
    if how == "mean_nonzero":
        # Both repos pad with all-zero rows; averaging over them would shrink
        # short utterances towards the origin by a length-dependent factor.
        mask = np.any(arr != 0, axis=1)
        if not mask.any():
            return np.zeros(arr.shape[1], dtype=np.float64)
        return arr[mask].mean(axis=0).astype(np.float64)
    if how == "max":
        return arr.max(axis=0).astype(np.float64)
    if how == "mean_std":
        mask = np.any(arr != 0, axis=1)
        sub = arr[mask] if mask.any() else arr[:1]
        return np.concatenate([sub.mean(axis=0), sub.std(axis=0)]).astype(np.float64)
    raise ValueError(f"unknown pool {how!r}")


def load_pt_dict(path: Path, pool: str) -> Dict[str, np.ndarray]:
    """{utt_id: tensor} saved by the merits-l-* pipelines."""
    import torch  # only needed for this source kind

    obj = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(obj, dict):
        raise ValueError(f"{path}: expected a dict of utt_id -> tensor")
    return {str(k): _pool(np.asarray(v, dtype=np.float64), pool) for k, v in obj.items()}


def load_npy_dir(roots: List[Path], pool: str) -> Dict[str, np.ndarray]:
    """Trees of <utt_id>.npy, e.g. Bi-LSTM's <split>/<emotion>/<utt_id>.npy.

    Indexed by file stem, so the split/label directories in the path are
    ignored — ground truth comes from the manifest, never from where a feature
    file happens to sit. Several roots can be given (earlier ones win), so a
    top-up extraction lives beside the original tree instead of duplicating it.
    """
    files = [p for r in roots for p in sorted(r.rglob("*.npy"))]
    if not files:
        raise FileNotFoundError(f"no .npy under {', '.join(str(r) for r in roots)}")
    total_mb = sum(p.stat().st_size for p in files) / 1e6
    print(f"    reading {len(files)} files ({total_mb:.0f} MB) ...", flush=True)

    out: Dict[str, np.ndarray] = {}
    dupes = 0
    t0 = time.time()
    for i, p in enumerate(files, 1):
        utt = p.stem
        if utt in out:
            dupes += 1
            continue
        out[utt] = _pool(np.load(p), pool)
        if i % 1000 == 0 or i == len(files):
            print(f"      {i}/{len(files)}  ({time.time() - t0:.0f}s)", flush=True)
    if dupes:
        print(f"    [warn] {dupes} duplicate utt_ids across roots, kept the first")
    return out


def _roots_of(cfg: dict) -> List[Path]:
    if "roots" in cfg:
        return [Path(r) for r in cfg["roots"]]
    return [Path(cfg["root"])]


def _cache_key(name: str, cfg: dict) -> str:
    """Identify a pooled source by its config plus the source files' state.

    File count and newest mtime are enough to notice a re-extraction; they will
    not notice an in-place edit that preserves both, so pass --no-cache after
    one of those.
    """
    if cfg["kind"] == "pt_dict":
        target = Path(cfg["path"])
        st = target.stat()
        targets, stamp = str(target), f"{st.st_size}:{st.st_mtime_ns}"
    else:
        roots = _roots_of(cfg)
        files = [p for r in roots for p in r.rglob("*.npy")]
        targets = "+".join(str(r) for r in roots)
        stamp = f"{len(files)}:{max((p.stat().st_mtime_ns for p in files), default=0)}"
    raw = f"{cfg['kind']}|{targets}|{cfg.get('pool', 'mean_nonzero')}|{stamp}"
    return f"{name}-{hashlib.sha1(raw.encode()).hexdigest()[:12]}.npz"


def load_source(name: str, cfg: dict, cache_dir: Optional[Path]) -> Source:
    kind = cfg["kind"]
    pool = cfg.get("pool", "mean_nonzero")

    cache_path = None
    if cache_dir is not None:
        cache_path = cache_dir / _cache_key(name, cfg)
        if cache_path.exists():
            z = np.load(cache_path, allow_pickle=False)
            vecs = dict(zip((str(u) for u in z["utt_ids"]), z["mat"]))
            src = Source(name=name, modality=cfg["modality"], vectors=vecs)
            print(f"  {name:<16} {src.modality:<5} {len(vecs):>5} utts  "
                  f"{src.dim:>5}-d  pool={pool}  (cached)")
            return src

    print(f"  {name:<16} {cfg['modality']:<5} loading, pool={pool}", flush=True)
    if kind == "pt_dict":
        vecs = load_pt_dict(Path(cfg["path"]), pool)
    elif kind == "npy_dir":
        vecs = load_npy_dir(_roots_of(cfg), pool)
    else:
        raise ValueError(f"{name}: unknown kind {kind!r} (pt_dict | npy_dir)")

    src = Source(name=name, modality=cfg["modality"], vectors=vecs)
    print(f"  {name:<16} {src.modality:<5} {len(vecs):>5} utts  {src.dim:>5}-d  pool={pool}")
    if cache_path is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        utt_ids = list(vecs)
        np.savez_compressed(cache_path, utt_ids=np.array(utt_ids),
                            mat=np.stack([vecs[u] for u in utt_ids]))
        print(f"    cached -> {cache_path}")
    return src


# --------------------------------------------------------------------------
# Manifest: the single source of truth for label and split
# --------------------------------------------------------------------------


@dataclass
class Manifest:
    utt_ids: List[str]
    labels: np.ndarray          # int
    splits: List[str]           # "train" / "test"
    label_names: List[str]


def load_manifest_csv(path: Path) -> Manifest:
    import csv

    utt, lab, spl = [], [], []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            utt.append(str(row["utt_id"]))
            lab.append(str(row["label"]))
            spl.append(str(row["split"]))
    names = sorted(set(lab))
    idx = {n: i for i, n in enumerate(names)}
    return Manifest(utt, np.array([idx[x] for x in lab]), spl, names)


def manifest_from_npy_dir(root: Path, split_re: str, label_re: str) -> Manifest:
    """Derive utt_id / label / split from a Bi-LSTM-style directory tree.

    Convenient for a first look, but it means label and split come from where a
    file sits on disk. Prefer an explicit CSV as soon as more than one repo's
    features are involved.
    """
    utt, lab, spl = [], [], []
    for p in sorted(root.rglob("*.npy")):
        parts = str(p.as_posix())
        ms, ml = re.search(split_re, parts), re.search(label_re, parts)
        if not (ms and ml):
            continue
        utt.append(p.stem)
        spl.append(ms.group(1))
        lab.append(ml.group(1))
    if not utt:
        raise ValueError(f"derived no rows from {root} with {split_re!r} / {label_re!r}")
    names = sorted(set(lab))
    idx = {n: i for i, n in enumerate(names)}
    return Manifest(utt, np.array([idx[x] for x in lab]), spl, names)


def align(man: Manifest, sources: Sequence[Source]) -> Tuple[Manifest, Dict[str, np.ndarray]]:
    """Keep only utterances every source covers; report what was dropped."""
    keep = [i for i, u in enumerate(man.utt_ids)
            if all(u in s.vectors for s in sources)]
    dropped = len(man.utt_ids) - len(keep)
    if dropped:
        for s in sources:
            miss = sum(1 for u in man.utt_ids if u not in s.vectors)
            if miss:
                print(f"  [warn] {s.name}: missing {miss}/{len(man.utt_ids)} manifest utts")
    if not keep:
        raise ValueError(
            "no utt_id is covered by every source. The repos probably spell ids "
            "differently — print a few keys from each source and reconcile them."
        )
    sub = Manifest([man.utt_ids[i] for i in keep], man.labels[keep],
                   [man.splits[i] for i in keep], man.label_names)
    mats = {s.name: np.stack([s.vectors[u] for u in sub.utt_ids]) for s in sources}
    print(f"  aligned on {len(sub.utt_ids)} utterances"
          + (f" ({dropped} dropped)" if dropped else ""))
    return sub, mats


# --------------------------------------------------------------------------
# Probes
# --------------------------------------------------------------------------


def make_probe(kind: str, seed: int):
    if kind == "linear":
        return LogisticRegression(max_iter=3000, C=1.0, random_state=seed)
    if kind == "mlp":
        return MLPClassifier(hidden_layer_sizes=(256,), max_iter=800,
                             early_stopping=True, random_state=seed)
    raise ValueError(f"unknown probe {kind!r} (linear | mlp)")


def prepare(X_tr: np.ndarray, X_te: np.ndarray, dim: Optional[int], seed: int):
    """Standardize, then optionally PCA to a common dim. Fit on train only.

    The PCA is the dimensionality control: without it a 4096-d text feature and
    a 34-d audio feature are not being asked the same question, and 'text has
    more unique information' can be an artefact of probe capacity.
    """
    sc = StandardScaler().fit(X_tr)
    A, B = sc.transform(X_tr), sc.transform(X_te)
    if dim and A.shape[1] > dim:
        p = PCA(n_components=dim, random_state=seed).fit(A)
        A, B = p.transform(A), p.transform(B)
    return A, B


def fit_predict(X_tr, y_tr, X_te, kind: str, seed: int) -> np.ndarray:
    clf = make_probe(kind, seed)
    clf.fit(X_tr, y_tr)
    return clf.predict(X_te)


# --------------------------------------------------------------------------
# Layer 2 statistics
# --------------------------------------------------------------------------


def exact_mcnemar(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value on the discordant pairs (b, c)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def bootstrap_ci(fn, *arrays, n: int = 2000, seed: int = 0, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    m = len(arrays[0])
    vals = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, m, m)
        vals[i] = fn(*[a[idx] for a in arrays])
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def error_structure(y: np.ndarray, pt: np.ndarray, pa: np.ndarray, pc: np.ndarray,
                    p_best: np.ndarray, label_names: List[str]) -> dict:
    t_ok, a_ok, c_ok, u_ok = pt == y, pa == y, pc == y, p_best == y
    both = int(np.sum(t_ok & a_ok))
    only_t = int(np.sum(t_ok & ~a_ok))
    only_a = int(np.sum(~t_ok & a_ok))
    neither = int(np.sum(~t_ok & ~a_ok))
    N = len(y)

    # Kuncheva & Whitaker Q over the two correctness vectors. Careful: this
    # measures PREDICTION diversity, not complementary information. Two probes
    # carrying identical information still disagree wherever each independently
    # guesses the part neither can see, which drives Q towards 0 and makes it
    # look complementary. Kept as description; rescue/damage below are the ones
    # that answer the question, because they route through the joint probe.
    denom = both * neither + only_t * only_a
    q = ((both * neither - only_t * only_a) / denom) if denom else float("nan")

    # Fraction at least one modality gets right. This bounds DECISION-level
    # fusion — any rule that picks or weights the two predictions — but NOT
    # feature-level fusion, which sees the representations and can be right
    # where both unimodal probes were wrong.
    ceiling = (N - neither) / N
    # Absolute gain is not comparable across pairs: a pair whose unimodal
    # probes are weak has far more room to gain. Normalise by the room that
    # actually exists. Accuracy throughout, so numerator and denominator are
    # the same kind of number (the Layer 1 gain is weighted F1, deliberately —
    # that is the thesis metric — but it must not be divided by an accuracy).
    # realized > 1 is therefore not an error: it means the joint probe beat
    # every possible decision-level combiner, which is a positive finding of
    # synergy — information carried only by the two together.
    acc_best_uni = max(float(np.mean(t_ok)), float(np.mean(a_ok)))
    headroom = ceiling - acc_best_uni
    realized = ((float(np.mean(c_ok)) - acc_best_uni) / headroom
                if headroom > 1e-9 else float("nan"))

    n_uni_wrong = int(np.sum(~u_ok))
    n_uni_right = int(np.sum(u_ok))
    rescue = float(np.sum(~u_ok & c_ok) / n_uni_wrong) if n_uni_wrong else float("nan")
    damage = float(np.sum(u_ok & ~c_ok) / n_uni_right) if n_uni_right else float("nan")

    return {
        "both": both, "only_text": only_t, "only_audio": only_a, "neither": neither,
        "oracle_ceiling": ceiling,
        "headroom": headroom,
        "realized": realized,
        "double_fault": neither / N,
        "disagreement": (only_t + only_a) / N,
        "q_statistic": q,
        "rescue_rate": rescue,
        "damage_rate": damage,
        # Per-emotion: what the joint probe fixes, and what it breaks, relative
        # to the better single modality.
        "fixed": Counter(label_names[c] for c in y[~u_ok & c_ok]),
        "broken": Counter(label_names[c] for c in y[u_ok & ~c_ok]),
    }


# --------------------------------------------------------------------------
# One pair
# --------------------------------------------------------------------------


def evaluate_pair(text: Source, audio: Source, man: Manifest,
                  mats: Dict[str, np.ndarray], probe: str, dim: Optional[int],
                  seed: int, boot: int) -> dict:
    tr = np.array([s == "train" for s in man.splits])
    te = ~tr
    if not tr.any() or not te.any():
        raise ValueError("manifest has no train or no test rows")
    y_tr, y_te = man.labels[tr], man.labels[te]

    Xt_tr, Xt_te = prepare(mats[text.name][tr], mats[text.name][te], dim, seed)
    Xa_tr, Xa_te = prepare(mats[audio.name][tr], mats[audio.name][te], dim, seed)
    Xc_tr = np.hstack([Xt_tr, Xa_tr])
    Xc_te = np.hstack([Xt_te, Xa_te])

    p_t = fit_predict(Xt_tr, y_tr, Xt_te, probe, seed)
    p_a = fit_predict(Xa_tr, y_tr, Xa_te, probe, seed)
    p_c = fit_predict(Xc_tr, y_tr, Xc_te, probe, seed)

    wf1 = lambda p: f1_score(y_te, p, average="weighted", zero_division=0)
    acc = lambda p: accuracy_score(y_te, p)

    best_uni, best_name = ((wf1(p_t), "text") if wf1(p_t) >= wf1(p_a)
                           else (wf1(p_a), "audio"))
    p_best = p_t if best_name == "text" else p_a
    gain = wf1(p_c) - best_uni

    b = int(np.sum((p_c == y_te) & (p_best != y_te)))
    c = int(np.sum((p_c != y_te) & (p_best == y_te)))

    lo, hi = bootstrap_ci(
        lambda yy, cc, bb: (f1_score(yy, cc, average="weighted", zero_division=0)
                            - f1_score(yy, bb, average="weighted", zero_division=0)),
        y_te, p_c, p_best, n=boot, seed=seed,
    )
    return {
        "text": text.name, "audio": audio.name,
        "text_dim": text.dim, "audio_dim": audio.dim,
        "n_test": int(te.sum()),
        "wf1_text": wf1(p_t), "wf1_audio": wf1(p_a), "wf1_concat": wf1(p_c),
        "acc_text": acc(p_t), "acc_audio": acc(p_a), "acc_concat": acc(p_c),
        "gain": gain, "gain_lo": lo, "gain_hi": hi,
        "best_unimodal": best_name,
        "mcnemar_b": b, "mcnemar_c": c, "mcnemar_p": exact_mcnemar(b, c),
        **error_structure(y_te, p_t, p_a, p_c, p_best, man.label_names),
    }


# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, type=Path)
    ap.add_argument("--pairs", nargs="*", default=None,
                    help="text:audio names; default is every text x audio pair")
    ap.add_argument("--probe", default="linear", choices=["linear", "mlp"])
    ap.add_argument("--dim", type=int, default=128,
                    help="PCA both modalities to this dim (0 = off, and then "
                         "any dimensionality difference is uncontrolled)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--cache", type=Path, default=Path(".cache"),
                    help="where to keep pooled features so re-runs are instant")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    spec = yaml.safe_load(args.spec.read_text(encoding="utf-8"))

    print("sources:")
    cache_dir = None if args.no_cache else args.cache
    sources = {n: load_source(n, c, cache_dir) for n, c in spec["features"].items()}

    m = spec["manifest"]
    if "csv" in m:
        man = load_manifest_csv(Path(m["csv"]))
    else:
        man = manifest_from_npy_dir(Path(m["from_npy_dir"]),
                                    m.get("split_re", r"/(train|test)/"),
                                    m.get("label_re", r"/(?:train|test)/([^/]+)/"))
    print(f"manifest: {len(man.utt_ids)} utts, labels={man.label_names}, "
          f"train={sum(s == 'train' for s in man.splits)} "
          f"test={sum(s == 'test' for s in man.splits)}")

    texts = [s for s in sources.values() if s.modality == "text"]
    audios = [s for s in sources.values() if s.modality == "audio"]
    if args.pairs:
        pairs = [(sources[p.split(":")[0]], sources[p.split(":")[1]]) for p in args.pairs]
    else:
        pairs = list(itertools.product(texts, audios))
    if not pairs:
        print("no text x audio pair to evaluate")
        return 1

    rows = []
    for t, a in pairs:
        man_p, mats = align(man, [t, a])
        rows.append(evaluate_pair(t, a, man_p, mats, args.probe,
                                  args.dim or None, args.seed, args.bootstrap))

    dim_note = f"PCA->{args.dim}" if args.dim else "raw dims, UNCONTROLLED"
    print(f"\n=== Layer 1: oracle gain ({args.probe} probe, {dim_note}, "
          f"test weighted F1) ===")
    head = (f"{'text':<14} {'audio':<14} {'text':>7} {'audio':>7} {'concat':>7} "
            f"{'gain':>7} {'95% CI':>16} {'McNemar':>8}")
    print(head)
    print("-" * len(head))
    for r in rows:
        print(f"{r['text']:<14} {r['audio']:<14} {r['wf1_text']:>7.4f} "
              f"{r['wf1_audio']:>7.4f} {r['wf1_concat']:>7.4f} {r['gain']:>+7.4f} "
              f"[{r['gain_lo']:+.4f},{r['gain_hi']:+.4f}] {r['mcnemar_p']:>8.4f}")
    print("\ngain = concat - better unimodal. A CI straddling 0 means this pair "
          "gives you nothing measurable; McNemar p is over the discordant test "
          "utterances only.")

    print(f"\n=== Layer 2: error structure (n_test={rows[0]['n_test']}) ===")
    head2 = (f"{'text':<14} {'audio':<14} {'both':>6} {'onlyT':>6} {'onlyA':>6} "
             f"{'none':>6} {'ceiling':>8} {'headrm':>7} {'realzd':>7} {'rescue':>7} "
             f"{'damage':>7} {'Q':>7}")
    print(head2)
    print("-" * len(head2))
    for r in rows:
        print(f"{r['text']:<14} {r['audio']:<14} {r['both']:>6} {r['only_text']:>6} "
              f"{r['only_audio']:>6} {r['neither']:>6} {r['oracle_ceiling']:>8.4f} "
              f"{r['headroom']:>7.4f} {r['realized']:>7.4f} {r['rescue_rate']:>7.4f} "
              f"{r['damage_rate']:>7.4f} {r['q_statistic']:>7.3f}")
    print("\nrealzd = (joint probe accuracy - better single modality accuracy) / "
          "headrm, where headrm = ceiling - that same accuracy. THIS is the "
          "number to compare across pairs: raw gain rewards a pair whose "
          "unimodal probes are weak, simply because it has more room.")
    print("rescue = of the utterances the better single modality gets wrong, the "
          "fraction the joint probe recovers; damage = of those it gets right, the "
          "fraction the joint probe loses. These two route through the joint probe, "
          "so they measure combinable information.")
    print("ceiling = fraction at least one modality gets right. It bounds "
          "DECISION-level fusion (any rule over the two predictions), not "
          "feature-level fusion, so realzd > 1 is a real signal, not a bug: the "
          "joint probe beat every possible decision-level combiner, meaning the "
          "pair carries synergy that lives only in the two together.")
    print("Q is prediction diversity ONLY: two probes carrying identical "
          "information still disagree wherever each guesses the part neither can "
          "see, so a low Q is not evidence of complementarity. Read rescue/damage "
          "instead, and Q only alongside them.")

    print("\n=== Per emotion: what the joint probe fixes and breaks ===")
    for r in rows:
        fx = " ".join(f"{k}:{v}" for k, v in sorted(r["fixed"].items()))
        bk = " ".join(f"{k}:{v}" for k, v in sorted(r["broken"].items()))
        print(f"{r['text']} x {r['audio']}  (vs {r['best_unimodal']} alone)")
        print(f"    fixed:  {fx or '(none)'}")
        print(f"    broken: {bk or '(none)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
