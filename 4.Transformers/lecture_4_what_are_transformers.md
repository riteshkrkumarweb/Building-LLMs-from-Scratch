#  What Are Transformers?

## 1. Main Idea

A **Transformer** is a deep neural-network architecture introduced in the 2017 research paper **"Attention Is All You Need"**.

Transformers became a major building block for modern language models such as **GPT** and **BERT**.

The original Transformer was designed mainly for **machine translation**, such as translating English text into German or French.

> Important: This lecture is an introduction. Detailed mathematics and coding are covered in later lectures.

---

# 2. Why Are Transformers Important?

Before Transformers, sequence-processing tasks commonly used architectures such as:

* RNNs
* LSTMs
* GRUs

These models process sequence information step by step.

Transformers introduced a different approach based heavily on **attention**, allowing the model to consider relationships between different tokens in a sequence.

This became especially useful for understanding context and long-range relationships.

---

# 3. The Transformer Paper

## Paper

**Attention Is All You Need**

## Year

**2017**

## Main contribution

The paper introduced the Transformer architecture and showed how attention could be used as the main mechanism for sequence modeling.

The original Transformer had two major parts:

```text
Input Text
    ↓
Encoder
    ↓
Context / Embeddings
    ↓
Decoder
    ↓
Output Text
```

---

# 4. Simplified Transformer Architecture

The lecture explains the Transformer using an **8-step simplified process**.

```text
1. Input Text
      ↓
2. Tokenization + Token IDs
      ↓
3. Encoder
      ↓
4. Vector Embeddings
      ↓
5. Decoder + Partial Output
      ↓
6. Predict Next Token
      ↓
7. Generate Output One Token at a Time
      ↓
8. Final Output
```

The example used in the lecture is a translation task.

For example:

```text
English → German
```

---

# 5. Step 1 — Input Text

The Transformer receives text as input.

Example:

```text
I like machine learning.
```

A neural network cannot directly understand text as words.

So the text must first be converted into numerical representations.

---

# 6. Step 2 — Tokenization

**Tokenization** means breaking text into smaller units called **tokens**.

For simple understanding:

```text
"I love AI"
```

can be represented as:

```text
["I", "love", "AI"]
```

Each token is then assigned a numerical ID.

Example:

```text
"I"     → 101
"love"  → 205
"AI"    → 87
```

> Important: A token is not always exactly one word. Modern tokenizers can split words into subwords or other pieces.

---

# 7. Step 3 — Encoder

The token IDs are passed into the **encoder**.

The encoder processes the input and creates useful representations of the input information.

A simplified view:

```text
Token IDs
   ↓
Encoder
   ↓
Meaningful representation
```

---

# 8. Step 4 — Vector Embedding

Token IDs themselves do not contain semantic meaning.

They are converted into **vector embeddings**.

Example:

```text
"king" → [0.21, 0.83, -0.14, ...]
"queen" → [0.19, 0.79, -0.11, ...]
```

These vectors exist in a high-dimensional space.

The important idea is:

```text
Similar meaning
      ↓
Similar position in vector space
```

Embeddings allow the model to represent relationships between words/tokens numerically.

---

# 9. Step 5 — Decoder

The decoder receives information from the encoder.

It also receives the output generated so far.

For example, while translating:

```text
Input:
English sentence

Already generated:
German word 1
German word 2
...
```

The decoder uses this information to predict the next token.

---

# 10. Step 6 — Predict the Next Token

The decoder predicts what token should come next.

Conceptually:

```text
Previous tokens + Encoder information
                 ↓
              Decoder
                 ↓
           Next token
```

For example:

```text
"The cat is"
       ↓
Prediction
       ↓
"sleeping"
```

During training, the model compares its prediction with the correct target and updates its parameters using a loss function.

---

# 11. Step 7 — Generate One Token at a Time

The decoder generates the output sequentially.

Example:

```text
Token 1
   ↓
Token 2
   ↓
Token 3
   ↓
Token 4
```

The generated output becomes part of the context used to generate the next token.

---

# 12. Step 8 — Final Output

After enough tokens are generated, we obtain the final output.

For translation:

```text
English sentence
      ↓
Transformer
      ↓
German sentence
```

---

# 13. Encoder and Decoder

The original Transformer architecture contains:

```text
Encoder + Decoder
```

### Encoder

Main purpose:

```text
Input text
    ↓
Useful representation
```

### Decoder

Main purpose:

```text
Representation + previous output
            ↓
       Next token
```

---

# 14. Self-Attention

One of the most important ideas behind Transformers is **self-attention**.

Self-attention allows the model to determine which other tokens are important when processing a particular token.

Example:

```text
"The animal didn't cross the road because it was tired."
```

To understand what **"it"** refers to, the model needs to consider other words in the sentence.

Self-attention helps the model examine relationships between tokens.

### Simple idea

```text
Token A
  ↓
Which other tokens are important?
  ↓
Assign different importance
  ↓
Build a better representation
```

This helps Transformers capture **long-range dependencies**.

---

# 15. Why Is It Called "Attention"?

The name comes from the idea that the model can assign different levels of importance to different parts of the input.

For a particular token:

```text
Token A
  ↓
Token B → high importance
Token C → low importance
Token D → medium importance
```

The model can therefore focus more on relevant context.

---

# 16. BERT vs GPT

Two important Transformer-based architectures discussed in the lecture are:

```text
BERT
GPT
```

They are related to the Transformer architecture but use different parts and objectives.

---

## BERT

BERT stands for:

**Bidirectional Encoder Representations from Transformers**

BERT uses the **encoder** side of the Transformer.

```text
BERT
  ↓
Encoder
```

BERT can look at context from both directions.

```text
Left context ← WORD → Right context
```

This helps it understand the meaning of a word using surrounding context.

Example:

```text
I deposited money in the bank.
```

versus:

```text
I sat beside the river bank.
```

The surrounding words help determine the meaning of **bank**.

BERT is commonly associated with understanding tasks such as classification and sentiment analysis.

---

## GPT

GPT stands for:

**Generative Pre-trained Transformer**

GPT uses a **decoder-only** Transformer architecture.

```text
GPT
 ↓
Decoder
```

GPT generates text by predicting the next token.

Example:

```text
The weather today is
          ↓
       sunny
```

Then:

```text
The weather today is sunny
                         ↓
                      and...
```

So GPT is fundamentally based on **next-token prediction**.

---

# 17. BERT vs GPT — Simple Comparison

| Feature | BERT | GPT |
|---|---|---|
| Meaning | Bidirectional Encoder Representations from Transformers | Generative Pre-trained Transformer |
| Transformer part | Encoder | Decoder |
| Main style | Bidirectional understanding | Autoregressive generation |
| Context | Looks at both sides during its pretraining objective | Uses previous context to predict the next token |
| Common use | Language understanding/classification | Text generation |

---

# 18. Transformer ≠ LLM

This is one of the most important points from the lecture.

Do **not** treat these terms as identical:

```text
Transformer ≠ LLM
```

They are related, but they are not the same thing.

---

# 19. Not All Transformers Are LLMs

Transformers can be used outside language.

For example:

```text
Transformer
   ├── NLP
   ├── Computer Vision
   └── Other sequence/data tasks
```

A major example is the **Vision Transformer (ViT)**.

Vision Transformers can be used for image-related tasks such as image classification.

Therefore:

```text
Not all Transformers are LLMs.
```

---

# 20. Not All LLMs Are Transformers

Language models existed before Transformers.

Examples include:

```text
RNN
LSTM
CNN-based architectures
Transformer
```

Therefore:

```text
Not all LLMs must be Transformers.
```

Historically, recurrent architectures such as RNNs and LSTMs were used for language modeling and sequence prediction.

Modern LLMs, however, are predominantly Transformer-based.

---

# 21. Transformer vs LLM

Think of them this way:

### Transformer

An **architecture**.

```text
Transformer
= Neural network architecture
```

### LLM

A **large language model** trained to work with language.

```text
LLM
= Large model trained for language-related tasks
```

A Transformer can be used to build an LLM, but the terms are not synonyms.

---

# 22. Important Relationships

```text
Transformer
    │
    ├── Encoder + Decoder
    │
    ├── Encoder-only → BERT-style
    │
    └── Decoder-only → GPT-style
```

And:

```text
Transformer
    ├── Language
    ├── Vision
    └── Other applications
```

---

# 23. Key Terms

## Token

A small unit of text processed by a model.

```text
Text → Tokens
```

---

## Token ID

A numerical ID assigned to a token.

```text
"hello" → 1532
```

---

## Embedding

A numerical vector representing a token in a continuous vector space.

```text
Token → Vector
```

---

## Encoder

Processes the input and creates representations containing useful information about it.

---

## Decoder

Uses available context to generate output tokens.

---

## Attention

A mechanism that helps the model determine which parts of the context are important.

---

## Self-Attention

Attention where tokens consider relationships with other tokens within the same sequence.

---

## Transformer

A neural-network architecture built around attention mechanisms.

---

## LLM

A large language model trained on large amounts of text/data to perform language-related tasks.

---

# 24. Most Important Points to Remember

* The Transformer architecture was introduced in the **2017 "Attention Is All You Need" paper**.
* The original Transformer was designed for **machine translation**.
* The original architecture contains an **encoder and decoder**.
* Text must be converted into tokens and numerical representations before being processed.
* Token IDs are converted into **vector embeddings**.
* **Self-attention** helps the model understand relationships between tokens.
* GPT is a **decoder-only** Transformer architecture.
* BERT is an **encoder-only** Transformer architecture.
* GPT is strongly associated with **next-token generation**.
* BERT uses bidirectional context for its language-understanding objective.
* **Not all Transformers are LLMs.**
* **Not all LLMs are Transformers.**
* Transformers can also be used for tasks such as computer vision.

---

# 25. One-Minute Revision

```text
Transformer
    ↓
Neural-network architecture
    ↓
Introduced in 2017
    ↓
Attention is All You Need
    ↓
Original architecture
    ↓
Encoder + Decoder
    ↓
Tokens → Embeddings
    ↓
Self-Attention
    ↓
Understand relationships between tokens
```

Modern variants:

```text
BERT
  ↓
Encoder-only
  ↓
Language understanding

GPT
  ↓
Decoder-only
  ↓
Next-token generation
```

Remember:

```text
Transformer ≠ LLM

Not all Transformers are LLMs.
Not all LLMs are Transformers.
```

---

# 26. Connection to the Next Topics

This lecture gives the high-level picture. The deeper LLM-building path is approximately:

```text
Transformer Basics
       ↓
Tokenization
       ↓
Token Embeddings
       ↓
Positional Embeddings
       ↓
Attention
       ↓
Self-Attention
       ↓
Causal Self-Attention
       ↓
Multi-Head Attention
       ↓
Transformer Block
       ↓
GPT Architecture
       ↓
Training
       ↓
Fine-Tuning
```

The next important concept to understand deeply is **attention**, followed by **Query, Key, and Value (Q, K, V)**.

---
