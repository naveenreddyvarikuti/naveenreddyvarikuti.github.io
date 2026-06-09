---
layout: post
title: "Positional Encoding Explained: Sinusoidal Embeddings and RoPE (Part 2)"
date: 2026-06-06 08:00:00 +0000
---

## Introduction

In [Part 1](https://naveenreddyvarikuti.github.io/2026/05/23/positional-encoding-transformers-explained.html), we explored three simple approaches to positional encoding. Direct integers, normalized values, and
binary vectors.

The most important lesson came from binary encoding. It revealed a multi
frequency structure hidden inside position representations. Each bit
oscillates at a different rate. Low bits flip rapidly, capturing fine
grained position differences. High bits flip slowly, capturing coarse
position in the sequence.

The structure was right. The problem was the shape. Square waves jump
between 0 and 1 with no in between. Adjacent positions sometimes looked
very different in vector space. Neural networks need smooth inputs to
learn smooth functions.

The fix is simple. Replace the square waves with smooth waves but keeping the
multi frequency structure and making it continuous.

In this post, we will cover two approaches:

- **Sinusoidal positional encoding**, introduced in the original
  Transformer paper (Vaswani et al., 2017). It uses sine and cosine waves
  at geometrically spaced frequencies to give each position a unique,
  smooth, bounded vector.

- **Rotary Position Embeddings (RoPE)**, introduced by Su et al. (2021).
  Instead of adding position to the embedding, it rotates the query and
  key vectors by their position. This makes relative position fall out of
  the dot product naturally, with no learning required.

Sinusoidal encoding was a significant step. But it has a structural
limitation in how it mixes position with meaning. RoPE fixes that
limitation, and is the method used in nearly every modern large language
model today, including LLaMA, Mistral, Gemma, and Phi.

We will build both ideas step by step. Every formula will be derived from
scratch. Every design choice will be motivated by a specific problem.

Let us start from exactly where Part 1 left off: the square waves of
binary encoding and the two functions that make them smooth.

