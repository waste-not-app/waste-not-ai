"""Run controlled COCO80 fine-tuning with validation-only selection."""

from pathlib import Path

from wastenot_ai.taxonomy import COCO_LABELS as COCO_LABELS

ROOT = Path(__file__).resolve().parents[1]


def validate_dataset(data, expected_labels):
    from training_experiments import validate_dataset as validate

    return validate(data, expected_labels)


def main():
    from training_experiments import cli

    cli("general", ROOT)


if __name__ == "__main__":
    main()
