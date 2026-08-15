"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";

type WeightPart = {
  offset: number;
  length: number;
  shape: number[];
};

type ModelConfig = {
  model_type: string;
  vocabulary: Record<string, number>;
  max_length: number;
  embedding_dim: number;
  hidden_dim: number;
  decision_threshold: number;
  weights: Record<string, WeightPart>;
  test_metrics: {
    accuracy: number;
    macro_f1: number;
    roc_auc: number;
  };
};

type LoadedModel = {
  config: ModelConfig;
  values: Float32Array;
};

type Prediction = {
  label: "Positive review" | "Negative review";
  probability: number;
  confidence: number;
};

const examples = {
  positive:
    "The product arrived quickly, works perfectly and feels much better than I expected.",
  negative:
    "This stopped working after two days. The quality is poor and I would not recommend it.",
};

let modelPromise: Promise<LoadedModel> | null = null;

function loadModel(): Promise<LoadedModel> {
  if (!modelPromise) {
    modelPromise = Promise.all([
      fetch("/model/config.json").then((response) => {
        if (!response.ok) throw new Error("Model configuration was not found.");
        return response.json() as Promise<ModelConfig>;
      }),
      fetch("/model/weights.bin").then((response) => {
        if (!response.ok) throw new Error("Model weights were not found.");
        return response.arrayBuffer();
      }),
    ]).then(([config, buffer]) => ({
      config,
      values: new Float32Array(buffer),
    }));
  }
  return modelPromise;
}

function tokenise(text: string): string[] {
  const cleaned = text
    .toLowerCase()
    .replace(/https?:\/\/\S+|www\.\S+/g, " url ")
    .replace(/<[^>]+>/g, " ")
    .replace(/n['’]t\b/g, " not")
    .replace(/[^a-z0-9'\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  const words = cleaned.match(/[a-z]+(?:'[a-z]+)?|\d+/g) ?? [];
  const bigrams = words
    .slice(0, -1)
    .map((word, index) => `${word}__${words[index + 1]}`);
  return [...words, ...bigrams];
}

function predict(text: string, model: LoadedModel): Prediction {
  const { config, values } = model;
  const tokens = tokenise(text).slice(0, config.max_length);
  const ids = tokens.map((token) => config.vocabulary[token] ?? 1);
  const embedding = config.weights.embedding;
  const hiddenWeight = config.weights.hidden_weight;
  const hiddenBias = config.weights.hidden_bias;
  const outputWeight = config.weights.output_weight;
  const outputBias = config.weights.output_bias;

  const pooled = new Float32Array(config.embedding_dim);
  for (const id of ids) {
    const start = embedding.offset + id * config.embedding_dim;
    for (let dimension = 0; dimension < config.embedding_dim; dimension += 1) {
      pooled[dimension] += values[start + dimension];
    }
  }
  const divisor = Math.max(ids.length, 1);
  for (let dimension = 0; dimension < config.embedding_dim; dimension += 1) {
    pooled[dimension] /= divisor;
  }

  const hidden = new Float32Array(config.hidden_dim);
  for (let neuron = 0; neuron < config.hidden_dim; neuron += 1) {
    let activation = values[hiddenBias.offset + neuron];
    const row = hiddenWeight.offset + neuron * config.embedding_dim;
    for (let dimension = 0; dimension < config.embedding_dim; dimension += 1) {
      activation += pooled[dimension] * values[row + dimension];
    }
    hidden[neuron] = Math.max(0, activation);
  }

  let logit = values[outputBias.offset];
  for (let neuron = 0; neuron < config.hidden_dim; neuron += 1) {
    logit += hidden[neuron] * values[outputWeight.offset + neuron];
  }
  const probability = 1 / (1 + Math.exp(-logit));
  const positive = probability >= config.decision_threshold;
  return {
    label: positive ? "Positive review" : "Negative review",
    probability,
    confidence: positive ? probability : 1 - probability,
  };
}

export default function Home() {
  const [review, setReview] = useState("");
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [metrics, setMetrics] = useState<ModelConfig["test_metrics"] | null>(null);
  const [status, setStatus] = useState<"idle" | "loading" | "error">("idle");

  useEffect(() => {
    loadModel()
      .then(({ config }) => setMetrics(config.test_metrics))
      .catch(() => setStatus("error"));
  }, []);

  const wordCount = useMemo(
    () => review.trim().split(/\s+/).filter(Boolean).length,
    [review],
  );

  async function analyse(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (review.trim().length < 3) return;
    setStatus("loading");
    try {
      const model = await loadModel();
      setPrediction(predict(review, model));
      setStatus("idle");
    } catch {
      setPrediction(null);
      setStatus("error");
    }
  }

  function selectExample(value: string) {
    setReview(value);
    setPrediction(null);
  }

  return (
    <main>
      <header className="site-header">
        <a className="brand" href="#top" aria-label="ReviewLens home">
          <span className="brand-mark" aria-hidden="true">R</span>
          <span>ReviewLens</span>
        </a>
        <span className="course-tag">ISY503 · Assessment 3</span>
      </header>

      <section className="hero" id="top">
        <div className="hero-copy">
          <p className="eyebrow">Neural sentiment analysis</p>
          <h1>What does your Amazon review really say?</h1>
          <p className="hero-intro">
            Enter a product review and a trained neural network will classify its
            sentiment as positive or negative.
          </p>
          <div className="metric-row" aria-label="Model test results">
            <div>
              <strong>{metrics ? `${(metrics.accuracy * 100).toFixed(1)}%` : "—"}</strong>
              <span>test accuracy</span>
            </div>
            <div>
              <strong>{metrics ? metrics.macro_f1.toFixed(3) : "—"}</strong>
              <span>macro F1</span>
            </div>
            <div>
              <strong>{metrics ? metrics.roc_auc.toFixed(3) : "—"}</strong>
              <span>ROC-AUC</span>
            </div>
          </div>
        </div>

        <div className="analyser-panel">
          <form onSubmit={analyse}>
            <div className="field-heading">
              <label htmlFor="review">Review text</label>
              <span>{wordCount} words</span>
            </div>
            <textarea
              id="review"
              value={review}
              onChange={(event) => {
                setReview(event.target.value);
                setPrediction(null);
              }}
              placeholder="Example: The product arrived quickly and works exactly as described..."
              rows={8}
              maxLength={3000}
            />
            <div className="example-row">
              <span>Try an example:</span>
              <button type="button" onClick={() => selectExample(examples.positive)}>
                Positive
              </button>
              <button type="button" onClick={() => selectExample(examples.negative)}>
                Negative
              </button>
            </div>
            <button
              className="analyse-button"
              type="submit"
              disabled={review.trim().length < 3 || status === "loading"}
            >
              {status === "loading" ? "Analysing…" : "Analyse review"}
              <span aria-hidden="true">→</span>
            </button>
          </form>

          <div className={`result ${prediction ? "is-visible" : ""}`} aria-live="polite">
            {prediction ? (
              <>
                <div className={prediction.probability >= 0.5 ? "result-icon positive" : "result-icon negative"}>
                  {prediction.probability >= 0.5 ? "+" : "−"}
                </div>
                <div>
                  <span className="result-kicker">Model prediction</span>
                  <strong>{prediction.label}</strong>
                  <small>{(prediction.confidence * 100).toFixed(1)}% confidence</small>
                </div>
              </>
            ) : (
              <p>
                {status === "error"
                  ? "The model could not be loaded. Please refresh and try again."
                  : "Your classification will appear here."}
              </p>
            )}
          </div>
        </div>
      </section>

      <section className="process" aria-labelledby="process-title">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Inside the model</p>
            <h2 id="process-title">From raw text to one clear prediction</h2>
          </div>
        </div>
        <ol className="process-grid">
          <li>
            <span>01</span>
            <h3>Clean</h3>
            <p>Normalise case, punctuation, links and common contractions.</p>
          </li>
          <li>
            <span>02</span>
            <h3>Encode</h3>
            <p>Convert words and adjacent word pairs into padded numeric sequences.</p>
          </li>
          <li>
            <span>03</span>
            <h3>Learn</h3>
            <p>Pass the sequence through embeddings and a nonlinear dense layer.</p>
          </li>
          <li>
            <span>04</span>
            <h3>Classify</h3>
            <p>Return “Positive review” or “Negative review” with confidence.</p>
          </li>
        </ol>
      </section>

      <section className="responsible-use">
        <div>
          <p className="eyebrow">Responsible use</p>
          <h2>A prediction is not a fact.</h2>
        </div>
        <p>
          The model was trained on English Amazon reviews. Sarcasm, unusual spelling,
          cultural context and reviews from other settings can reduce reliability.
          Confidence should support human judgement, not replace it.
        </p>
      </section>

      <footer>
        <span>ReviewLens</span>
        <span>Multi-Domain Sentiment Dataset · Johns Hopkins University</span>
      </footer>
    </main>
  );
}
