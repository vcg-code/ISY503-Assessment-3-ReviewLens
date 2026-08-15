"""Run one prediction with the trained PyTorch sentiment model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from train_model import SentimentNetwork, encode_review


def load_trained_model(
    config_path: Path = Path("public/model/config.json"),
    checkpoint_path: Path = Path("ml/artifacts/best_model.pt"),
) -> tuple[SentimentNetwork, dict, dict]:
    configuration = json.loads(config_path.read_text(encoding="utf-8"))
    vocabulary = configuration["vocabulary"]
    model = SentimentNetwork(
        vocabulary_size=len(vocabulary),
        embedding_dim=configuration["embedding_dim"],
        hidden_dim=configuration["hidden_dim"],
        dropout=0.0,
    )
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    model.eval()
    return model, vocabulary, configuration


@torch.no_grad()
def predict_review(text: str) -> tuple[str, float]:
    model, vocabulary, configuration = load_trained_model()
    sequence = encode_review(
        text, vocabulary, configuration["max_length"]
    ).unsqueeze(0)
    probability = torch.sigmoid(model(sequence)).item()
    label = "Positive review" if probability >= 0.5 else "Negative review"
    confidence = probability if probability >= 0.5 else 1 - probability
    return label, confidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("text", help="English Amazon-style product review")
    args = parser.parse_args()
    label, confidence = predict_review(args.text)
    print(f"{label} ({confidence:.1%} confidence)")


if __name__ == "__main__":
    main()
