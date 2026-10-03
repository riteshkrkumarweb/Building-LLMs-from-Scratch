# GPT (Generative Pre-trained Transformer)

## 1. What is GPT?

**GPT (Generative Pre-trained Transformer)** is a **decoder-only Transformer model** designed to generate text by predicting the next token based on the tokens that came before it.

```text
Text
 ↓
Tokenizer
 ↓
Token IDs
 ↓
Embedding
 ↓
GPT Decoder Blocks
 ↓
Next-token prediction
 ↓
Next token
 ↓
Next-token prediction
 ↓
...
```

---

## 2. GPT is Decoder-Only

The original Transformer architecture has two main parts:

```text
Encoder → Decoder
```

GPT uses only the **decoder side**:

```text
GPT

Input
  ↓
Embedding
  ↓
Decoder
  ↓
Next-token prediction
```

This does **not** mean that GPT cannot process or understand the input.

The GPT decoder uses **causal self-attention** to process the context and generate the next token.

---

## 3. Encoder vs Decoder

### Encoder

> **Encoder understands the relationships and context between input tokens.**

It does **not** simply mean:

```text
Words → Numbers
```

### Decoder

> **Decoder uses the available context to generate tokens step by step.**

It does **not** simply mean:

```text
Numbers → Words
```

The conversion from text into numerical form happens mainly through **tokenization and embeddings**.

---

# 4. How GPT Handles Raw Text

GPT cannot directly perform mathematical operations on raw words.

The text first goes through several steps.

```text
"I love AI"
      ↓
   Tokenizer
      ↓
["I", " love", " AI"]
      ↓
   Token IDs
      ↓
[101, 245, 789]     ← example IDs
      ↓
   Embedding
      ↓
Numerical vectors
      ↓
GPT Decoder Blocks
```

---

# 5. Tokenization

A tokenizer breaks text into **tokens**.

Example:

```text
"I love AI"
```

might become:

```text
["I", " love", " AI"]
```

The exact tokens depend on the tokenizer.

Each token is assigned an integer ID:

```text
"I"     → 101
" love" → 245
" AI"   → 789
```

These numbers are called **token IDs**.

### Important

Token IDs are just identifiers.

For example:

```text
101 = "I"
245 = " love"
789 = " AI"
```

The number `245` does not mathematically mean that "love" is bigger or smaller than another word.

---

# 6. Embedding

The token IDs are converted into vectors using an **embedding layer**.

Conceptually:

```text
Token ID
   ↓
Embedding
   ↓
Vector
```

Example:

```text
101 → [0.21, 0.53, -0.17, ...]
245 → [0.72, 0.14,  0.63, ...]
789 → [0.31, 0.87, -0.31, ...]
```

These vectors are the numerical representations that are processed by the Transformer.

The embedding values are **learned during training**.

---

# 7. GPT Pipeline

The complete simplified GPT pipeline is:

```text
                 GPT
                  │
                  ▼
                Text
                  │
                  ▼
              Tokenizer
                  │
                  ▼
              Token IDs
                  │
                  ▼
              Embedding
                  │
                  ▼
          Numerical Vectors
                  │
                  ▼
       ┌────────────────────┐
       │    GPT DECODER     │
       │                    │
       │ Causal Attention   │
       │        ↓           │
       │ Feed Forward       │
       │        ↓           │
       │ Add & Normalize    │
       └─────────┬──────────┘
                 │
                 ▼
        Contextual Information
                 │
                 ▼
          Output Probabilities
                 │
                 ▼
           Next Token
```

---

# 8. What Does the GPT Decoder Do?

The decoder does not simply convert numbers back into words.

Its main job is to process the token representations while respecting the **causal/left-to-right constraint**, then use the resulting information to predict the next token.

For example:

```text
"The cat is"
```

GPT might predict:

```text
sleeping → 0.42
running  → 0.18
eating   → 0.11
...
```

The actual probabilities depend on the model and context.

After selecting/generating a token:

```text
"The cat is sleeping"
```

GPT processes the new sequence again and predicts the next token.

This continues until generation stops.

---

# 9. Causal Self-Attention

GPT uses **causal self-attention**.

This means a token can use information from the tokens before it, but it cannot use future tokens.

Example:

```text
The cat is sleeping
```

While processing:

```text
The
```

it cannot see:

```text
cat is sleeping
```

While processing:

```text
The cat
```

it can see:

```text
The cat
```

But not:

```text
is sleeping
```

Visualized:

```text
          Can Look At

The       ✓
cat       ✓ ✓
is        ✓ ✓ ✓
sleeping  ✓ ✓ ✓ ✓
```

The future is masked.

---

# 10. Why Does GPT Use Causal Attention?

Because GPT is trained to predict the **next token**.

Example:

```text
Input:

"The cat is"

Target:

"sleeping"
```

During training, GPT learns:

```text
The → predict cat

The cat → predict is

The cat is → predict sleeping
```

It must not be allowed to look at the answer while learning to predict it.

That is why future tokens are masked.

---

# 11. Next-Token Prediction

The central idea of GPT is:

> **Given the previous tokens, predict the next token.**

Example:

```text
"The"
 ↓
"cat"

"The cat"
 ↓
"is"

"The cat is"
 ↓
"sleeping"
```

Mathematically, GPT models:

```text
P(next token | previous tokens)
```

For example:

```text
P("sleeping" | "The cat is")
```

The model calculates probabilities for many possible next tokens.

---

# 12. Repeated Generation

GPT generates text one token at a time.

```text
Input:
"The cat"

       ↓

Predict:
"is"

       ↓

"The cat is"

       ↓

Predict:
"sleeping"

       ↓

"The cat is sleeping"

       ↓

Predict:
"on"

       ↓

"The cat is sleeping on"

       ↓

...
```

This is called **autoregressive generation**.

The generated token becomes part of the context used for the next prediction.

---

# 13. GPT vs Encoder-Decoder Transformer

## Encoder-Decoder

```text
Input
  ↓
Tokenizer
  ↓
Embedding
  ↓
Encoder
  ↓
Encoder representation
  ↓
Decoder
  ↓
Output
```

The decoder can use the encoder's information through **cross-attention**.

---

## GPT

```text
Input
  ↓
Tokenizer
  ↓
Embedding
  ↓
GPT Decoder
  ↓
Next Token
  ↓
Next Token
  ↓
Next Token
  ↓
...
```

GPT does not have a separate encoder.

Its decoder blocks use **causal self-attention** to process the context and support next-token generation.

---

# 14. Why Can GPT Work Without a Separate Encoder?

A separate encoder is not required for GPT's objective.

GPT is designed around:

```text
Previous tokens
      ↓
Causal self-attention
      ↓
Contextual information
      ↓
Next-token prediction
```

The decoder blocks themselves process the input context.

Therefore:

```text
Encoder-Decoder Transformer:

Encoder + Decoder


GPT:

Decoder only
```

---

# 15. Important Correction

Do not memorize:

```text
Encoder = Text → Numbers
Decoder = Numbers → Text
```

That is an oversimplification and is not the actual meaning of encoder and decoder in Transformers.

A better understanding is:

```text
Tokenizer
    ↓
Text → Token IDs

Embedding
    ↓
Token IDs → Vectors

Encoder
    ↓
Builds contextual information from input tokens

Decoder
    ↓
Uses context to generate tokens
```

---

# 16. GPT Architecture in One Diagram

```text
                     GPT
                      │
                      ▼
                  Raw Text
                      │
                      ▼
                  Tokenizer
                      │
                      ▼
                  Token IDs
                      │
                      ▼
                  Embedding
                      │
                      ▼
              Token Representations
                      │
                      ▼
        ┌─────────────────────────────┐
        │       Decoder Block         │
        │                             │
        │  Causal Self-Attention      │
        │             ↓               │
        │  Feed-Forward Network       │
        │             ↓               │
        │  Normalization              │
        └──────────────┬──────────────┘
                       │
                       ▼
                  More Decoder
                    Blocks
                       │
                       ▼
             Final Representation
                       │
                       ▼
                Linear Layer
                       │
                       ▼
                   Softmax
                       │
                       ▼
              Token Probabilities
                       │
                       ▼
                 Next Token
                       │
                       ▼
              Added to Context
                       │
                       ▼
                 Next Token
                       │
                       ▼
                      ...
```

---

# 17. Key Terms

### Token
A piece of text processed by the model.

```text
"hello"
"ing"
" AI"
```

### Token ID
An integer assigned to a token.

```text
"hello" → 1532
```

### Embedding
A learned numerical vector representing a token.

```text
1532 → [0.2, -0.4, 0.7, ...]
```

### Self-Attention
A mechanism that allows tokens to consider other tokens in the sequence.

### Causal Self-Attention
Self-attention where each position cannot use information from future positions.

### Decoder Block
A Transformer block used by GPT to process the sequence using causal self-attention and a feed-forward network.

### Next-Token Prediction
Predicting what token should come next based on the previous tokens.

### Autoregressive Generation
Generating one token, adding it to the context, and using the expanded context to generate the next token.

---

# 18. Final Mental Model

Remember GPT like this:

```text
TEXT
 ↓
TOKENIZE
 ↓
TOKEN IDs
 ↓
EMBED
 ↓
VECTORS
 ↓
GPT DECODER
 ↓
UNDERSTAND CONTEXT
 ↓
PREDICT NEXT TOKEN
 ↓
ADD TOKEN TO CONTEXT
 ↓
PREDICT NEXT TOKEN
 ↓
...
```

### One-line definition

> **GPT is a decoder-only Transformer that uses causal self-attention to process previous tokens and predict the next token repeatedly.**
