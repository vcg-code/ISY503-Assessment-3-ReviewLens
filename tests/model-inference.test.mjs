import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const projectRoot = new URL("../", import.meta.url);
const config = JSON.parse(
  await readFile(new URL("public/model/config.json", projectRoot), "utf8"),
);
const buffer = await readFile(new URL("public/model/weights.bin", projectRoot));
const values = new Float32Array(
  buffer.buffer,
  buffer.byteOffset,
  buffer.byteLength / Float32Array.BYTES_PER_ELEMENT,
);

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
  const bigrams = words
    .slice(0, -1)
    .map((word, index) => `${word}__${words[index + 1]}`);
  return [...words, ...bigrams];
}

function probability(text) {
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
  for (let dimension = 0; dimension < config.embedding_dim; dimension += 1) {
    pooled[dimension] /= Math.max(ids.length, 1);
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
  return 1 / (1 + Math.exp(-logit));
}

test("positive review example is classified as positive", () => {
  const score = probability(
    "The product arrived quickly, works perfectly and feels much better than I expected.",
  );
  assert.ok(score >= 0.5, `Expected positive probability, received ${score}`);
});

test("negative review example is classified as negative", () => {
  const score = probability(
    "This stopped working after two days. The quality is poor and I would not recommend it.",
  );
  assert.ok(score < 0.5, `Expected negative probability, received ${score}`);
});

test("exported model has the expected dimensions", () => {
  assert.equal(config.weights.embedding.shape[0], 12000);
  assert.equal(config.weights.embedding.shape[1], 48);
  assert.equal(config.weights.hidden_weight.shape[0], 64);
});
