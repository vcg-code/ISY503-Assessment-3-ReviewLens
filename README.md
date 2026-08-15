# ReviewLens — ISY503 Assessment 3

ReviewLens is an English Amazon review sentiment classifier. It trains a neural
network on the official Johns Hopkins Multi-Domain Sentiment Dataset and exposes
the trained model through a simple website. The required interface accepts review
text and returns exactly **Positive review** or **Negative review**.

## Dataset

Source: [Multi-Domain Sentiment Dataset, Johns Hopkins University](https://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html)

The source contains 8,000 labelled reviews: 1,000 positive and 1,000 negative
reviews for each of four Amazon domains (books, DVDs, electronics, and kitchen
products). The preparation pipeline:

1. loads the positive and negative files from all four domains;
2. cleans case, HTML, URLs, punctuation and whitespace;
3. removes 148 exact duplicates before splitting to reduce data leakage;
4. removes 154 token-length outliers outside the 1st–99th percentiles;
5. performs a reproducible stratified 70/15/15 train/validation/test split;
6. fits the vocabulary on the training data only;
7. encodes unigrams and adjacent bigrams, then pads or truncates each sequence.

## Neural network

The model is implemented in PyTorch:

```text
Token IDs
  → 48-dimensional embedding
  → masked average pooling
  → 64-unit dense layer with ReLU
  → dropout (35%)
  → single sigmoid output
```

Training uses binary cross-entropy, AdamW, batches of 128, gradient clipping and
early stopping based on validation loss. Random seed 503 makes the split and
training reproducible.

## Current test results

| Metric | Result |
|---|---:|
| Accuracy | 81.94% |
| Macro F1 | 0.8193 |
| ROC-AUC | 0.8894 |
| Positive recall | 79.45% |
| Negative recall | 84.47% |

Confusion matrix (`actual × predicted`):

|  | Negative | Positive |
|---|---:|---:|
| Negative | 484 | 89 |
| Positive | 120 | 464 |

The full training history and per-domain results are saved in
`ml/artifacts/training_report.json`.

## Project structure

```text
app/                         Website interface and browser inference
ml/train_model.py            Data preparation, training, evaluation and export
ml/predict.py                Command-line inference example
ml/artifacts/                Training report and local checkpoint
public/model/                Browser-readable trained model
tests/model-inference.test.mjs  Exported-model tests
```

## Run the model training

Create a Python environment and install the requirements, then run:

```bash
python ml/train_model.py
```

The script downloads the official dataset automatically when it is not already
available. It saves the best checkpoint, a JSON report, and the model files used
by the website.

Test a single review:

```bash
python ml/predict.py "This product works perfectly and I love it."
```

## Run the website

```bash
npm install
npm run dev
```

The trained weights run directly in the visitor's browser; review text is not
sent to an external service.

## Ethics and limitations

- The training data is limited to older English Amazon reviews and may not
  represent current language or other platforms.
- Sarcasm, mixed sentiment, spelling variation and cultural context can cause
  incorrect predictions.
- Removing duplicates and fitting the vocabulary only on training data reduces
  leakage, but does not eliminate sampling bias.
- Model confidence is not certainty and should support, rather than replace,
  human judgement.
- Review text is processed locally in the browser, which avoids collecting it on
  a server in this implementation.

## References

Blitzer, J., Dredze, M., & Pereira, F. (2007). Biographies, Bollywood, boom-boxes
and blenders: Domain adaptation for sentiment classification. *Proceedings of the
45th Annual Meeting of the Association of Computational Linguistics*, 440–447.
https://aclanthology.org/P07-1056/

Paszke, A., Gross, S., Massa, F., et al. (2019). PyTorch: An imperative style,
high-performance deep learning library. *Advances in Neural Information Processing
Systems, 32*. https://proceedings.neurips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html
