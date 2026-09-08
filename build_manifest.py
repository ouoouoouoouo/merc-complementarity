"""Build the one manifest every feature source is joined onto.

The repos disagree on both label set and split:

  merits-l-text / merits-l-llama : {angry, happy(hap+exc), sad, neutral},
                                   their own train/val/test CSVs
  Bi-LSTM                        : {ang, exc, neu, sad} -- exc is its own
                                   class and hap was dropped at extraction
                                   time -- split by session (1-4 / 5)

This takes the merits manifests as canonical, since that is the convention the
thesis reports in, and emits utt_id,label,split. merits' val is folded into
train because the probes here fit on train and report on test only (the MLP
probe carves its own validation out of what it is given).

Anything in the output that Bi-LSTM has no .npy for is listed at the end --
those are the hap utterances, and Bi-LSTM/extract_for_manifest.py extracts them.

Usage:
    python build_manifest.py \
        --merits-manifests ~/merits-l-llama/data/manifests/iemocap \
        --out manifests/iemocap_common.csv \
        --check-npy ~/Bi-LSTM/data/iemocap/processed_paper/audio
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import List

DEFAULT_LABEL_NAMES = ["angry", "happy", "sad", "neutral"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merits-manifests", required=True, type=Path,
                    help="directory holding train.csv / val.csv / test.csv")
    ap.add_argument("--out", default=Path("manifests/iemocap_common.csv"), type=Path)
    ap.add_argument("--label-names", nargs="*", default=DEFAULT_LABEL_NAMES,
                    help="index -> name, matching dataset.label_names in the "
                         "merits config that produced these manifests")
    ap.add_argument("--val-into", default="train", choices=["train", "test", "drop"],
                    help="what to do with merits' val split (default: fold into train)")
    ap.add_argument("--check-npy", type=Path, default=None,
                    help="a Bi-LSTM feature root; reports which manifest utts "
                         "have no .npy there yet")
    args = ap.parse_args()

    rows: List[dict] = []
    for split in ("train", "val", "test"):
        p = args.merits_manifests / f"{split}.csv"
        if not p.exists():
            print(f"  {split}.csv not found, skipping")
            continue
        with p.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                lab = r["label"]
                # merits manifests store the label as an int index.
                name = args.label_names[int(lab)] if lab.strip().isdigit() else lab
                out_split = split if split != "val" else args.val_into
                if out_split == "drop":
                    continue
                rows.append({"utt_id": str(r["utt_id"]), "label": name,
                             "split": out_split, "orig_split": split})
        print(f"  {split}.csv: {sum(1 for x in rows)} rows so far")

    if not rows:
        raise SystemExit(f"no rows read from {args.merits_manifests}")

    dupes = len(rows) - len({r['utt_id'] for r in rows})
    if dupes:
        raise SystemExit(f"{dupes} duplicate utt_ids across splits — refusing to "
                         "write a manifest where an utterance is in two splits")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["utt_id", "label", "split", "orig_split"])
        w.writeheader()
        w.writerows(rows)

    print(f"\nwrote {args.out}  ({len(rows)} utterances)")
    print("  split:", dict(Counter(r["split"] for r in rows)))
    print("  label:", dict(Counter(r["label"] for r in rows)))

    if args.check_npy:
        have = {p.stem for p in args.check_npy.rglob("*.npy")}
        missing = [r for r in rows if r["utt_id"] not in have]
        print(f"\nagainst {args.check_npy}: {len(rows) - len(missing)}/{len(rows)} "
              f"already extracted")
        if missing:
            by_label = Counter(r["label"] for r in missing)
            print(f"  {len(missing)} missing: {dict(by_label)}")
            print("  -> run Bi-LSTM/extract_for_manifest.py to fill these in")
            print("  (expected to be the hap utterances, which Bi-LSTM's "
                  "preprocess.py skipped)")
        extra = len(have) - (len(rows) - len(missing))
        if extra > 0:
            print(f"  {extra} extracted utts are not in the manifest; they are "
                  "simply unused (IEMOCAP labels outside the 4-class set)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
