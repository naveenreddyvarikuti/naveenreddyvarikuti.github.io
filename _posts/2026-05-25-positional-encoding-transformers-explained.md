---
layout: post
title: "Positional Encoding Explained: Deep Dive"
date: 2026-05-25 08:00:00 +0000
---

## Introduction

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

## The Bag of Words Problem

Consider two sentences: "Dog bites man" and "Man bites dog."

Same three words. Completely different meanings. Ideally language model must
tell these apart.

A Transformer without positional encoding cannot distinguish between these two
sentences. The math makes it impossible.

### How Self Attention Computes Scores

Each token is first converted into an embedding vector. The model then projects
these embeddings into queries and keys using learned weight matrices $$W_q$$ and
$$W_k$$:

$$Q_i = W_q \cdot e_i$$

$$K_j = W_k \cdot e_j$$

The attention score between token $$i$$ and token $$j$$ is their dot product:

$$\text{score}(i, j) = Q_i \cdot K_j = (W_q \cdot e_i) \cdot (W_k \cdot e_j)$$

Based on the above formula.

The score depends on the embedding of token $$i$$ and the embedding of token $$j$$.
Nothing else. The positions $$i$$ and $$j$$ do not appear anywhere in the formula.

### Why Order Becomes Invisible

The word "dog" gets the same embedding vector whether it appears at position 1
or position 3. So the attention score between "dog" and "bites" is identical
in both sentences.

In "Dog bites man":

$$\text{score}(\text{dog}, \text{bites}) = (W_q \cdot e_{\text{dog}}) \cdot (W_k \cdot e_{\text{bites}})$$

In "Man bites dog":

$$\text{score}(\text{dog}, \text{bites}) = (W_q \cdot e_{\text{dog}}) \cdot (W_k \cdot e_{\text{bites}})$$

Identical. This holds for every pair of tokens.

The full attention matrix for "Dog bites man" is identical to the full attention
matrix for "Man bites dog." Every value, every row, every column will be same.
 
![Bag of Words animation](/assets/animations/BagOfWordsProblem.gif)

### Permutation Invariance

This property is called **permutation invariance**.

Shuffle the tokens in any order and the attention scores do not change. "Dog
bites man" and "Man bites dog" and "Bites dog man" all produce the same
attention pattern.

Embeddings are looked up by token identity, not by position. Position simply
does not exist in the computation.

Without a mechanism to inject position, the Transformer is a bag of words
model. It knows which words are present. It does not know where they are.

This is the problem that positional encodings exist to solve.


