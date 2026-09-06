"""Training entry point for EfficientDet-D3.

This script trains the EfficientDet branch using the shared
dataset and evaluation protocol.
"""


def main() -> None:
    """Delegate to the canonical comparison runner."""

    import sys
    from src.training.train_comparison import main as run
    raise SystemExit(run(["--model", "efficientdet_d3", *sys.argv[1:]]))


if __name__ == "__main__":
    main()
