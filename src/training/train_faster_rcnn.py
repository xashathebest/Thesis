"""Training entry point for Faster R-CNN with ResNet50.

This script trains the Faster R-CNN branch using the shared
dataset and evaluation protocol.
"""


def main() -> None:
    """Delegate to the canonical comparison runner."""

    import sys
    from src.training.train_comparison import main as run
    raise SystemExit(run(["--model", "faster_rcnn", *sys.argv[1:]]))


if __name__ == "__main__":
    main()
