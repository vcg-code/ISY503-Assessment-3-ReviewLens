"""Train and export the Assessment 3 Amazon review sentiment model.

The script uses the official Johns Hopkins Multi-Domain Sentiment Dataset.
It creates a balanced, stratified train/validation/test split, trains a small
PyTorch neural network, evaluates it, and exports browser-readable weights.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import random
import re
import tarfile
import urllib.request
from array import array
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset


SEED = 503
DATA_URL = (
    "https://www.cs.jhu.edu/~mdredze/datasets/sentiment/"
    "domain_sentiment_data.tar.gz"
)
DOMAINS = ("books", "dvd", "electronics", "kitchen_&_housewares")
TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+")


@dataclass(frozen=True)
class Review:
    text: str
    label: int
    domain: str


def clean_text(text: str) -> str:
    text = html.unescape(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " url ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"n['’]t\b", " not", text)
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def unigram_tokens(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(clean_text(text))


def model_tokens(text: str) -> list[str]:
    words = unigram_tokens(text)
    bigrams = [f"{left}__{right}" for left, right in zip(words, words[1:])]
    return words + bigrams


def download_dataset(data_root: Path) -> Path:
    archive = data_root / "domain_sentiment_data.tar.gz"
    extracted = data_root / "sorted_data_acl"
    if extracted.exists():
        return extracted

    data_root.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        print("Downloading the official Johns Hopkins dataset...")
        urllib.request.urlretrieve(DATA_URL, archive)
    print("Extracting dataset...")
    with tarfile.open(archive, "r:gz") as bundle:
        bundle.extractall(data_root, filter="data")
    return extracted


def parse_review_file(path: Path, label: int, domain: str) -> list[Review]:
    source = path.read_text(encoding="latin-1")
    blocks = re.findall(r"<review>(.*?)</review>", source, flags=re.DOTALL)
    reviews: list[Review] = []
    for block in blocks:
        match = re.search(
            r"<review_text>(.*?)</review_text>", block, flags=re.DOTALL
        )
        if match:
            text = clean_text(match.group(1))
            if text:
                reviews.append(Review(text=text, label=label, domain=domain))
    return reviews


def load_reviews(dataset_root: Path) -> tuple[list[Review], int]:
    reviews: list[Review] = []
    for domain in DOMAINS:
        folder = dataset_root / domain
        reviews.extend(parse_review_file(folder / "negative.review", 0, domain))
        reviews.extend(parse_review_file(folder / "positive.review", 1, domain))

    # Exact duplicate removal prevents the same text appearing in two splits.
    unique: dict[tuple[str, int], Review] = {}
    for review in reviews:
        unique[(review.text, review.label)] = review
    return list(unique.values()), len(reviews) - len(unique)


def remove_length_outliers(reviews: list[Review]) -> tuple[list[Review], dict]:
    lengths = sorted(len(unigram_tokens(item.text)) for item in reviews)
    lower = max(5, lengths[int(0.01 * (len(lengths) - 1))])
    upper = lengths[int(0.99 * (len(lengths) - 1))]
    filtered = [
        item for item in reviews if lower <= len(unigram_tokens(item.text)) <= upper
    ]
    return filtered, {
        "method": "Removed reviews outside the 1st-99th token-length percentiles",
        "minimum_tokens": lower,
        "maximum_tokens": upper,
        "removed": len(reviews) - len(filtered),
    }


def stratified_split(
    reviews: list[Review], seed: int = SEED
) -> tuple[list[Review], list[Review], list[Review]]:
    rng = random.Random(seed)
    train: list[Review] = []
    validation: list[Review] = []
    test: list[Review] = []
    for label in (0, 1):
        group = [item for item in reviews if item.label == label]
        rng.shuffle(group)
        train_end = int(0.70 * len(group))
        validation_end = train_end + int(0.15 * len(group))
        train.extend(group[:train_end])
        validation.extend(group[train_end:validation_end])
        test.extend(group[validation_end:])
    rng.shuffle(train)
    rng.shuffle(validation)
    rng.shuffle(test)
    return train, validation, test


def build_vocabulary(
    reviews: list[Review], maximum_size: int, minimum_frequency: int = 2
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for review in reviews:
        counts.update(model_tokens(review.text))
    vocabulary = {"<PAD>": 0, "<UNK>": 1}
    candidates = [
        pair for pair in counts.most_common() if pair[1] >= minimum_frequency
    ]
    for token, _ in candidates[: maximum_size - len(vocabulary)]:
        vocabulary[token] = len(vocabulary)
    return vocabulary


def encode_review(text: str, vocabulary: dict[str, int], max_length: int) -> Tensor:
    unknown = vocabulary["<UNK>"]
    encoded = [vocabulary.get(token, unknown) for token in model_tokens(text)]
    encoded = encoded[:max_length]
    output = torch.zeros(max_length, dtype=torch.long)
    if encoded:
        output[: len(encoded)] = torch.tensor(encoded, dtype=torch.long)
    return output


class ReviewDataset(Dataset):
    def __init__(
        self, reviews: list[Review], vocabulary: dict[str, int], max_length: int
    ) -> None:
        self.reviews = reviews
        self.sequences = torch.stack(
            [encode_review(item.text, vocabulary, max_length) for item in reviews]
        )
        self.labels = torch.tensor(
            [item.label for item in reviews], dtype=torch.float32
        )

    def __len__(self) -> int:
        return len(self.reviews)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.sequences[index], self.labels[index]


class SentimentNetwork(nn.Module):
    """Embedding-average neural classifier with a nonlinear hidden layer."""

    def __init__(
        self, vocabulary_size: int, embedding_dim: int, hidden_dim: int, dropout: float
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        self.hidden = nn.Linear(embedding_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, tokens: Tensor) -> Tensor:
        mask = tokens.ne(0).unsqueeze(-1)
        embedded = self.embedding(tokens) * mask
        lengths = mask.sum(dim=1).clamp(min=1)
        pooled = embedded.sum(dim=1) / lengths
        hidden = torch.relu(self.hidden(pooled))
        return self.output(self.dropout(hidden)).squeeze(1)


def binary_metrics(labels: list[int], probabilities: list[float]) -> dict:
    predictions = [int(score >= 0.5) for score in probabilities]
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    accuracy = (tp + tn) / len(labels)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    negative_recall = tn / max(tn + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    positives = sum(labels)
    negatives = len(labels) - positives
    ordered = sorted(zip(probabilities, labels), reverse=True)
    previous_fpr = previous_tpr = auc = 0.0
    true_positives = false_positives = 0
    for _, label in ordered:
        if label == 1:
            true_positives += 1
        else:
            false_positives += 1
        tpr = true_positives / max(positives, 1)
        fpr = false_positives / max(negatives, 1)
        auc += (fpr - previous_fpr) * (tpr + previous_tpr) / 2
        previous_fpr, previous_tpr = fpr, tpr

    return {
        "accuracy": round(accuracy, 4),
        "precision_positive": round(precision, 4),
        "recall_positive": round(recall, 4),
        "recall_negative": round(negative_recall, 4),
        "macro_f1": round(
            (
                f1
                + 2
                * (tn / max(tn + fn, 1))
                * negative_recall
                / max((tn / max(tn + fn, 1)) + negative_recall, 1e-12)
            )
            / 2,
            4,
        ),
        "roc_auc": round(auc, 4),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, loss_function: nn.Module
) -> tuple[float, dict, list[float]]:
    model.eval()
    total_loss = 0.0
    labels: list[int] = []
    probabilities: list[float] = []
    for tokens, target in loader:
        logits = model(tokens)
        total_loss += loss_function(logits, target).item() * len(target)
        probabilities.extend(torch.sigmoid(logits).tolist())
        labels.extend(target.int().tolist())
    return total_loss / len(labels), binary_metrics(labels, probabilities), probabilities


def train(
    model: SentimentNetwork,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    checkpoint: Path,
) -> list[dict]:
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    history: list[dict] = []
    best_loss = math.inf
    patience = 4
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for tokens, labels in train_loader:
            optimizer.zero_grad()
            logits = model(tokens)
            loss = loss_function(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * len(labels)

        train_loss = running_loss / len(train_loader.dataset)
        validation_loss, validation_metrics, _ = evaluate(
            model, validation_loader, loss_function
        )
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "validation_loss": round(validation_loss, 5),
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_macro_f1": validation_metrics["macro_f1"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d} | train loss {train_loss:.4f} | "
            f"validation loss {validation_loss:.4f} | "
            f"validation accuracy {validation_metrics['accuracy']:.2%}"
        )

        if validation_loss < best_loss - 1e-4:
            best_loss = validation_loss
            epochs_without_improvement = 0
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stopping: validation loss stopped improving.")
                break

    model.load_state_dict(torch.load(checkpoint, weights_only=True))
    return history


def export_browser_model(
    model: SentimentNetwork,
    vocabulary: dict[str, int],
    max_length: int,
    metrics: dict,
    output_folder: Path,
) -> None:
    output_folder.mkdir(parents=True, exist_ok=True)
    tensors = [
        ("embedding", model.embedding.weight.detach().cpu()),
        ("hidden_weight", model.hidden.weight.detach().cpu()),
        ("hidden_bias", model.hidden.bias.detach().cpu()),
        ("output_weight", model.output.weight.detach().cpu()),
        ("output_bias", model.output.bias.detach().cpu()),
    ]
    weights = array("f")
    layout: dict[str, dict] = {}
    offset = 0
    for name, tensor in tensors:
        values = tensor.contiguous().view(-1).tolist()
        layout[name] = {
            "offset": offset,
            "length": len(values),
            "shape": list(tensor.shape),
        }
        weights.extend(values)
        offset += len(values)
    (output_folder / "weights.bin").write_bytes(weights.tobytes())

    configuration = {
        "model_type": "Embedding average + dense ReLU neural network",
        "classes": {"0": "Negative review", "1": "Positive review"},
        "vocabulary": vocabulary,
        "max_length": max_length,
        "embedding_dim": model.embedding.embedding_dim,
        "hidden_dim": model.hidden.out_features,
        "decision_threshold": 0.5,
        "tokenisation": "lowercase unigrams plus adjacent bigrams",
        "weights": layout,
        "test_metrics": metrics,
    }
    (output_folder / "config.json").write_text(
        json.dumps(configuration, separators=(",", ":")), encoding="utf-8"
    )


def class_counts(reviews: list[Review]) -> dict[str, int]:
    return {
        "negative": sum(item.label == 0 for item in reviews),
        "positive": sum(item.label == 1 for item in reviews),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("ml/data/raw"))
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--vocabulary-size", type=int, default=12000)
    parser.add_argument("--max-length", type=int, default=260)
    parser.add_argument("--embedding-dim", type=int, default=48)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    args = parser.parse_args()

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(min(8, max(1, torch.get_num_threads())))

    dataset_root = download_dataset(args.data_root)
    raw_reviews, duplicates_removed = load_reviews(dataset_root)
    reviews, outlier_summary = remove_length_outliers(raw_reviews)
    train_reviews, validation_reviews, test_reviews = stratified_split(reviews)
    vocabulary = build_vocabulary(train_reviews, args.vocabulary_size)

    print(
        f"Loaded {len(raw_reviews) + duplicates_removed:,} labelled reviews; "
        f"removed {duplicates_removed:,} exact duplicates before splitting."
    )
    print(f"After outlier removal: {len(reviews):,} reviews.")
    print(
        f"Split sizes: train={len(train_reviews):,}, "
        f"validation={len(validation_reviews):,}, test={len(test_reviews):,}."
    )
    print(f"Vocabulary size: {len(vocabulary):,} tokens.")

    train_dataset = ReviewDataset(train_reviews, vocabulary, args.max_length)
    validation_dataset = ReviewDataset(
        validation_reviews, vocabulary, args.max_length
    )
    test_dataset = ReviewDataset(test_reviews, vocabulary, args.max_length)
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    model = SentimentNetwork(
        vocabulary_size=len(vocabulary),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        dropout=0.35,
    )
    checkpoint = Path("ml/artifacts/best_model.pt")
    history = train(
        model,
        train_loader,
        validation_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        checkpoint=checkpoint,
    )
    test_loss, test_metrics, probabilities = evaluate(
        model, test_loader, nn.BCEWithLogitsLoss()
    )

    by_domain: dict[str, dict] = {}
    for domain in DOMAINS:
        indices = [
            index for index, item in enumerate(test_reviews) if item.domain == domain
        ]
        by_domain[domain] = binary_metrics(
            [test_reviews[index].label for index in indices],
            [probabilities[index] for index in indices],
        )

    test_metrics["loss"] = round(test_loss, 5)
    test_metrics["by_domain"] = by_domain
    report = {
        "seed": SEED,
        "source": DATA_URL,
        "source_review_count": len(raw_reviews) + duplicates_removed,
        "duplicates_removed": duplicates_removed,
        "outlier_removal": outlier_summary,
        "retained_review_count": len(reviews),
        "split": {
            "train": {"size": len(train_reviews), **class_counts(train_reviews)},
            "validation": {
                "size": len(validation_reviews),
                **class_counts(validation_reviews),
            },
            "test": {"size": len(test_reviews), **class_counts(test_reviews)},
        },
        "model": {
            "architecture": "Embedding average -> Dense ReLU -> Dropout -> Sigmoid",
            "vocabulary_size": len(vocabulary),
            "max_length": args.max_length,
            "embedding_dim": args.embedding_dim,
            "hidden_dim": args.hidden_dim,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
        },
        "training_history": history,
        "test_metrics": test_metrics,
    }
    artifacts = Path("ml/artifacts")
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "training_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    export_browser_model(
        model, vocabulary, args.max_length, test_metrics, Path("public/model")
    )
    print("\nTest metrics")
    print(json.dumps(test_metrics, indent=2))
    print("Browser model exported to public/model/.")


if __name__ == "__main__":
    main()
