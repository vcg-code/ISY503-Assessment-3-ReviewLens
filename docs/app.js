const examples = {
  positive: "The product arrived quickly, works perfectly and feels much better than I expected.",
  negative: "This stopped working after two days. The quality is poor and I would not recommend it.",
};

let modelPromise;

function loadModel() {
  if (!modelPromise) {
    modelPromise = Promise.all([
      fetch("./model/config.json").then((response) => {
        if (!response.ok) throw new Error("Model configuration was not found.");
        return response.json();
      }),
      fetch("./model/weights.bin").then((response) => {
        if (!response.ok) throw new Error("Model weights were not found.");
        return response.arrayBuffer();
      }),
    ]).then(([config, buffer]) => ({ config, values: new Float32Array(buffer) }));
  }
  return modelPromise;
}

function tokenise(text) {
  const cleaned = text
    .toLowerCase()
    .replace(/https?:\/\/\S+|www\.\S+/g, " url ")
    .replace(/<[^>]+>/g, " ")
    .replace(/n['’]t\b/g, " not")
    .replace(/[^a-z0-9'\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  const words = cleaned.match(/[a-z]+(?:'[a-z]+)?|\d+/g) ?? [];
  const bigrams = words.slice(0, -1).map((word, index) => `${word}__${words[index + 1]}`);
  return [...words, ...bigrams];
}

function predict(text, model) {
  const { config, values } = model;
  const ids = tokenise(text)
    .slice(0, config.max_length)
    .map((token) => config.vocabulary[token] ?? 1);
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

const form = document.querySelector("#review-form");
const review = document.querySelector("#review");
const wordCount = document.querySelector("#word-count");
const analyseButton = document.querySelector("#analyse-button");
const buttonLabel = document.querySelector("#button-label");
const result = document.querySelector("#result");

function clearResult() {
  result.className = "result";
  result.innerHTML = '<p id="result-placeholder">Your classification will appear here.</p>';
}

function updateReview(value) {
  review.value = value;
  const count = value.trim().split(/\s+/).filter(Boolean).length;
  wordCount.textContent = `${count} ${count === 1 ? "word" : "words"}`;
  analyseButton.disabled = value.trim().length < 3;
  clearResult();
}

review.addEventListener("input", () => updateReview(review.value));
document.querySelectorAll("[data-example]").forEach((button) => {
  button.addEventListener("click", () => updateReview(examples[button.dataset.example]));
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (review.value.trim().length < 3) return;
  analyseButton.disabled = true;
  buttonLabel.textContent = "Analysing…";
  try {
    const prediction = predict(review.value, await loadModel());
    const positive = prediction.probability >= 0.5;
    result.className = `result is-visible ${positive ? "positive-result" : "negative-result"}`;
    result.innerHTML = `
      <div class="result-icon ${positive ? "positive" : "negative"}">${positive ? "+" : "−"}</div>
      <div>
        <span class="result-kicker">Model prediction</span>
        <strong>${prediction.label}</strong>
        <small>${(prediction.confidence * 100).toFixed(1)}% confidence</small>
      </div>`;
  } catch {
    result.className = "result";
    result.innerHTML = "<p>The model could not be loaded. Please refresh and try again.</p>";
  } finally {
    analyseButton.disabled = review.value.trim().length < 3;
    buttonLabel.textContent = "Analyse review";
  }
});

loadModel()
  .then(({ config }) => {
    document.querySelector("#accuracy").textContent = `${(config.test_metrics.accuracy * 100).toFixed(1)}%`;
    document.querySelector("#macro-f1").textContent = config.test_metrics.macro_f1.toFixed(3);
    document.querySelector("#roc-auc").textContent = config.test_metrics.roc_auc.toFixed(3);
  })
  .catch(() => {
    result.innerHTML = "<p>The model could not be loaded. Please refresh and try again.</p>";
  });
