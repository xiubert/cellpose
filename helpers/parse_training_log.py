#!/usr/bin/env python3
"""Parse cellpose training .out log file, write losses to CSV, and plot metrics."""

import re
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# matches lines like: 0, train_loss=2.9191, test_loss=0.6794, LR=0.000000, time 12.27s
PATTERN = re.compile(
    r"\[INFO\]\s+(\d+),\s+train_loss=([\d.]+),\s+test_loss=([\d.]+),\s+LR=([\d.]+),\s+time\s+([\d.]+)s"
)


def parse(log_path: Path) -> list[dict]:
    rows = []
    with open(log_path) as f:
        for line in f:
            m = PATTERN.search(line)
            if m:
                rows.append({
                    "epoch":      int(m.group(1)),
                    "train_loss": float(m.group(2)),
                    "test_loss":  float(m.group(3)),
                    "lr":         float(m.group(4)),
                    "time_s":     float(m.group(5)),
                })
    return rows


def plot(rows: list[dict], out_path: Path) -> None:
    epochs     = [r["epoch"]      for r in rows]
    train_loss = [r["train_loss"] for r in rows]
    test_loss  = [r["test_loss"]  for r in rows]
    lr         = [r["lr"]         for r in rows]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1.plot(epochs, train_loss, marker="o", label="train loss")
    ax1.plot(epochs, test_loss,  marker="o", label="test loss")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, lr, marker="o", color="tab:orange", label="learning rate")
    ax2.set_ylabel("Learning Rate")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Cellpose Training Metrics")
    fig.tight_layout()

    plot_path = out_path.with_suffix(".png")
    fig.savefig(str(plot_path), dpi=150)
    print(f"Saved plot to {plot_path}")
    plt.show()


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <logfile.out> [output.csv]")
        sys.exit(1)

    log_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else log_path.with_suffix(".csv")

    rows = parse(log_path)
    if not rows:
        print("No training loss lines found in log.")
        sys.exit(1)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "test_loss", "lr", "time_s"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")
    plot(rows, out_path)


if __name__ == "__main__":
    main()
