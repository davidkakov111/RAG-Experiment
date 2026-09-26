import { pipeline } from '@xenova/transformers';

// Load a small local embedding model (384-dim, same family as MiniLM)
const embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');

// Helper: get a plain float array embedding for a string
async function embed(text) {
  const output = await embedder(text, { pooling: 'mean', normalize: true });
  return Array.from(output.data); // 384-length array
}

// --- your chunks ---
const chunks = [
  // 1. Unrelated
  "Arctic terns undertake the longest migration of any animal, travelling from their Arctic breeding grounds to Antarctica and back each year.",

  // 2. Perfect match, paraphrased differently
  "Orion's auth service was rewritten last year; the team migrated it from Python to Rust to eliminate an entire class of memory-related security bugs.",

  // 3. Same product, DIFFERENT component (confusable - same system, wrong answer)
  "The Orion dashboard frontend is built with TypeScript and React, communicating with backend services over a REST API.",

  // 4. Related but vague (mentions the module, no language specifics)
  "Orion's architecture consists of five core modules: authentication, payments, notifications, the dashboard, and the audit logger. Each was built by a different sub-team.",

  // 5. Confusing - same domain word "authentication" but generic/policy content, no tech
  "Company policy requires two-factor authentication for all employees accessing internal systems, regardless of which application they use.",

  // 6. Confusing - mentions "Rust" but totally different context (keyword collision)
  "Rust is a common problem in older industrial equipment; regular inspection and anti-corrosion coating are recommended for outdoor machinery.",

  // 7. Unrelated
  "To make a good chocolate cake, cream the butter and sugar first, then alternate adding the dry ingredients and milk to avoid overmixing the batter.",

  // 8. Perfect match
  "The Orion authentication module is implemented in Rust, chosen for its memory safety guarantees and performance under concurrent login load.",

  // 9. Unrelated
  "The central bank raised interest rates by a quarter point on Thursday, citing persistent inflation pressure in the services sector.",

  // 10. Same LANGUAGE, different component (confusable - right keyword, wrong subject)
  "The Orion payment processing service is also written in Rust, sharing several internal crates with other backend modules.",
];

// Embed all chunks
const chunkEmbeddings = [];
for (const chunk of chunks) {
  chunkEmbeddings.push(await embed(chunk));
}

// --- your question ---
const question = "What programming language is used to build the Orion authentication module?";
const questionEmbedding = await embed(question);

// Cosine similarity.
// Result interpretation:
// 1 → vectors point in exactly the same direction (maximally similar meaning)
// 0 → vectors are orthogonal (unrelated)
// -1 → vectors point in exactly opposite directions
function cosineSim(a, b) {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

// Score every chunk against the question
const scored = chunks.map((chunk, i) => ({
  chunk,
  score: cosineSim(questionEmbedding, chunkEmbeddings[i]),
}));

// Sort descending, take top-k
const topK = 10;
const topChunks = scored
  .sort((a, b) => b.score - a.score)
  .slice(0, topK);

// Print scores so you can see the ranking behavior
for (const { chunk, score } of topChunks) {
  console.log(`score=${score.toFixed(3)} | ${chunk.slice(0, 80)}...`);
}

// Build the final prompt to paste into browser AI
const prompt = `Answer only using the context below. If not found, say so.

Context:
${topChunks.map(c => `- ${c.chunk}`).join('\n')}

Question: ${question}
`;

console.log('\n--- PROMPT TO PASTE ---\n');
console.log(prompt);
