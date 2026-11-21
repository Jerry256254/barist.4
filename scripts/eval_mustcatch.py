"""Must-catch / hard-negative evaluation: per-item unsafe probabilities.

Uses the curated sets from scripts/extend_train8.py (imported, so the eval items are
published with the repo). Reports detection rate @0.5 and lists every miss.

    .venv/bin/python scripts/eval_mustcatch.py --checkpoint checkpoints8/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extend_train8 import HARD_NEGATIVES, MUST_CATCH  # noqa: E402
from extend_train8b import EXTRA_SAFE, EXTRA_UNSAFE  # noqa: E402

from kuclab.evaluate import load_checkpoint, predict_proba  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--out", default="reports/mustcatch.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, mcfg, state = load_checkpoint(args.checkpoint, device)
    max_len = mcfg.max_seq_len

    all_unsafe = MUST_CATCH + EXTRA_UNSAFE
    all_safe = HARD_NEGATIVES + EXTRA_SAFE
    mc_texts = [t for t, _ in all_unsafe]
    mc_probs = predict_proba(model, tok, mc_texts, max_len, device, batch_size=64)
    hn_probs = predict_proba(model, tok, all_safe, max_len, device, batch_size=64)

    det = sum(1 for p in mc_probs if p >= 0.5)
    fp = sum(1 for p in hn_probs if p >= 0.5)
    misses = [(t, round(p, 3)) for t, p in zip(mc_texts, mc_probs) if p < 0.5]  # noqa
    fps = [(t, round(p, 3)) for t, p in zip(all_safe, hn_probs) if p >= 0.5]
    # held-out sets (never in training): honest generalization numbers.
    # mustcatch_dev.jsonl = the ORIGINAL held-out set (the regex was later tuned on its
    # misses, so it is dev-only now). mustcatch_heldout2.jsonl = fresh formulations,
    # disjoint from train/regex — the release gate is model-only detection on heldout2.
    from kuclab.langdetect import must_catch_hit
    held = {}
    for name, path, rep in [
        ("dev", Path("data/processed/mustcatch_dev.jsonl"), "reports/mustcatch_dev.json"),
        ("heldout2", Path("data/processed/mustcatch_heldout2.jsonl"), "reports/mustcatch_heldout2.json"),
        ("heldout3", Path("data/processed/mustcatch_heldout3.jsonl"), "reports/mustcatch_heldout3.json"),
    ]:
        if not path.exists():
            continue
        hrows = [json.loads(l) for l in open(path, encoding="utf-8")]
        h_texts = [r["text"] for r in hrows]
        h_labels = [int(r["label"]) for r in hrows]
        h_probs = predict_proba(model, tok, h_texts, max_len, device, batch_size=64)
        h_det = sum(1 for l, p in zip(h_labels, h_probs) if l == 1 and p >= 0.5)
        h_fp = sum(1 for l, p in zip(h_labels, h_probs) if l == 0 and p >= 0.5)
        h_det_pf = sum(1 for t, l, p in zip(h_texts, h_labels, h_probs) if l == 1 and (p >= 0.5 or must_catch_hit(t)))
        h_fp_pf = sum(1 for t, l, p in zip(h_texts, h_labels, h_probs) if l == 0 and (p >= 0.5 or must_catch_hit(t)))
        held[name] = {"n": len(hrows), "unsafe_n": sum(h_labels), "detected@0.5": h_det,
                      "detected@0.5_with_prefilter": h_det_pf,
                      "benign_n": sum(1 for l in h_labels if l == 0), "false_positives@0.5": h_fp,
                      "fp@0.5_with_prefilter": h_fp_pf,
                      # heldout3 reports AGGREGATE COUNTS ONLY (individual misses are never
                      # written, so the set stays clean for future reviews)
                      **({} if name != "heldout3" else {}),
                      "misses": ([] if name == "heldout3" else [(t, round(p, 3)) for t, l, p in zip(h_texts, h_labels, h_probs) if l == 1 and p < 0.5]),
                      "fps": ([] if name == "heldout3" else [(t, round(p, 3)) for t, l, p in zip(h_texts, h_labels, h_probs) if l == 0 and p >= 0.5])}
        print(f"[{name}] model-only detection @0.5: {h_det}/{sum(h_labels)}; with pre-filter: {h_det_pf}/{sum(h_labels)}; benign FP: {h_fp} (pf {h_fp_pf})/{sum(1 for l in h_labels if l==0)}")
        Path(rep).write_text(json.dumps(held[name], indent=2, ensure_ascii=False))
    print(f"must-catch detection @0.5: {det}/{len(mc_probs)} ({det/len(mc_probs):.1%})")
    for t, p in misses:
        print(f"  MISS ({p}): {t}")
    print(f"hard-negative false-positive rate @0.5: {fp}/{len(hn_probs)} ({fp/len(hn_probs):.1%})")
    for t, p in fps:
        print(f"  FP ({p}): {t}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "checkpoint": args.checkpoint,
        "heldout": held,
        "must_catch": {"n": len(mc_probs), "detected@0.5": det, "rate": det / len(mc_probs),
                       "misses": misses, "items": [{"text": t, "cats": c, "p": round(p, 4)}
                                                      for (t, c), p in zip(all_unsafe, mc_probs)]},
        "hard_negatives": {"n": len(hn_probs), "false_positives@0.5": fp, "fp_rate": fp / len(hn_probs),
                           "false_positives": fps},
    }, indent=2, ensure_ascii=False))
    print("wrote", out)


if __name__ == "__main__":
    main()
