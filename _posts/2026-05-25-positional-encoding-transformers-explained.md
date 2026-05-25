---
layout: post
title: "Positional Encoding Transformers Explained: Deep Dive"
date: 2026-05-25 08:00:00 +0000
---

### Introduction

Language models process text as a sequence of tokens. While token embeddings can represent the meaning of individual words, they do not inherently represent where those words appear in the sequence.

For example, the words in:

> Dog bites man

and

> Man bites dog

are identical, but their meanings are completely different because the order of the words is different.

Positional encodings are techniques that allow Transformer models to incorporate information about token positions. They help the model distinguish between different token orders and reason about relationships between tokens across a sequence.

Over the years, several approaches have been proposed for representing positional information. Some are simple and intuitive, while others are mathematically elegant and widely used in modern large language models.

In this blog, we will build these ideas step by step, starting from the simplest possible approaches and gradually moving toward the methods used in modern Transformers.

We will cover:

- Direct position values
- Normalized position representations
- Binary position encodings
- Sinusoidal positional embeddings
- Rotary Position Embeddings (RoPE)

For each approach, we will understand the intuition behind it, see how it works mathematically, identify its limitations, and understand how those limitations naturally motivate the next idea.

By the end of this blog, you should have a solid understanding of how positional information is represented in Transformers, why different positional encoding methods were developed, and why modern language models rely heavily on techniques such as RoPE.