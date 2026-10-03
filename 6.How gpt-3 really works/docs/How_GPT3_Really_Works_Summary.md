# How Does GPT-3 Really Work?


## 1. GPT

**GPT = Generative Pre-trained Transformer**

- **Generative** → generates text.
- **Pre-trained** → first learns general patterns from a large text corpus.
- **Transformer** → uses the Transformer architecture.

GPT is an **autoregressive language model**: it predicts the next token using the tokens that came before it.

Example:

```text
The cat is sitting on the ___
                         ↓
                       mat
```

The predicted token is then added to the sequence and the process continues.

---

## 2. GPT Is Decoder-Only

The original Transformer architecture contains:

```text
Encoder + Decoder
```

GPT uses:

```text
Decoder only
```

GPT does not use a separate Transformer encoder.

The decoder blocks use **causal/masked self-attention**, which prevents a position from using future tokens.

---

## 3. GPT Evolution

The lecture follows the progression from the original Transformer idea to GPT models.

### GPT-1

The GPT approach combined:

```text
Transformer decoder
+
Unsupervised / self-supervised language-model pretraining
+
Task-specific learning
```

The main idea was that a language model could first learn general language representations and then be adapted to downstream tasks.

### GPT-2

GPT-2 increased model size and training data.

Important idea:

> Scaling a language model can improve its ability to perform different language tasks without training a separate model for every task.

GPT-2 also demonstrated stronger **zero-shot** behavior.

### GPT-3

GPT-3 scaled the same general idea much further.

The original GPT-3 paper reported:

```text
175 billion parameters
```

GPT-3 became especially well known for **few-shot learning** and **in-context learning**.

---

# 4. GPT-3 Architecture

A simplified GPT-3 pipeline:

```text
Text
 ↓
Tokenizer
 ↓
Token IDs
 ↓
Token Embeddings
 ↓
Positional Information
 ↓
Transformer Blocks
 ↓
Vocabulary Logits
 ↓
Probability Distribution
 ↓
Next Token
```

The model repeatedly predicts the next token.

---

# 5. Tokenization

Text is converted into smaller units called **tokens**.

A token can be:

- a word
- part of a word
- punctuation
- another piece of text

Conceptually:

```text
"I love AI"
      ↓
["I", "love", "AI"]
```

The tokenizer then maps tokens to integer IDs.

```text
"I"    → 101
"love" → 245
"AI"   → 782
```

The exact IDs depend on the tokenizer.

---

# 6. Token IDs Are Not Meaning

A token ID is mainly an identifier.

For example:

```text
AI → 782
```

The number `782` does not inherently mean "AI".

The model learns useful numerical representations through embeddings.

---

# 7. Token Embeddings

Token IDs are converted into vectors.

```text
Token ID
   ↓
Embedding lookup
   ↓
Vector
```

Conceptually:

```text
AI
↓
[0.21, -0.48, 0.73, ...]
```

The embedding values are learned during training.

---

# 8. Positional Information

Self-attention needs information about the order of tokens.

Compare:

```text
Dog bites man.
Man bites dog.
```

The same words appear, but the order changes the meaning.

Therefore, Transformer models include information about token positions.

---

# 9. Self-Attention

Self-attention allows a token representation to use information from other tokens in the same sequence.

Example:

```text
The animal didn't cross the road because it was tired.
```

Understanding relationships such as what **"it"** refers to requires information from other parts of the sentence.

Self-attention provides a mechanism for learning such relationships.

---

# 10. Query, Key, and Value

Attention uses three representations:

```text
Query (Q)
Key   (K)
Value (V)
```

Simple intuition:

- **Query** → what information am I looking for?
- **Key** → what information can I be matched with?
- **Value** → what information should be passed forward?

The attention mechanism calculates how strongly tokens should use information from other tokens.

---

# 11. Attention Formula

The standard scaled dot-product attention formula is:

```text
Attention(Q, K, V)
=
softmax(QKᵀ / √dₖ)V
```

Where:

```text
Q  = Query
K  = Key
V  = Value
dₖ = dimension of the key vectors
```

The softmax produces attention weights.

---

# 12. Causal / Masked Self-Attention

GPT uses causal attention.

When predicting a token, the model can use:

```text
Previous tokens
+
Current position
```

but not future tokens.

Example:

```text
Position 1 → sees 1
Position 2 → sees 1, 2
Position 3 → sees 1, 2, 3
Position 4 → sees 1, 2, 3, 4
```

The future is masked.

This prevents the model from seeing the answer during next-token training.

---

# 13. Multi-Head Attention

Transformers use multiple attention heads.

Instead of one attention operation:

```text
Input
 ↓
Attention
```

there are multiple attention operations:

```text
             ┌→ Head 1 ─┐
Input ────────┼→ Head 2 ─┼→ Combine
             ├→ Head 3 ─┤
             └→ ... ────┘
```

Different heads can learn different relationships between tokens.

---

# 14. Transformer Block

A simplified GPT Transformer block contains:

```text
Input
 ↓
Masked Multi-Head Self-Attention
 ↓
Residual Connection + Normalization
 ↓
Feed-Forward Network
 ↓
Residual Connection + Normalization
 ↓
Output
```

Many Transformer blocks are stacked together.

---

# 15. Feed-Forward Network

After attention, each token representation passes through a feed-forward neural network.

Conceptually:

```text
Input
 ↓
Linear layer
 ↓
Activation
 ↓
Linear layer
 ↓
Output
```

It performs additional nonlinear transformations on the representations.

---

# 16. Residual / Shortcut Connections

Residual connections allow information to skip over a sub-layer.

Conceptually:

```text
Input ──────────────────┐
  ↓                     │
Transformation          │
  ↓                     │
  └──────── Add ◄───────┘
```

They help optimization and information flow in deep neural networks.

---

# 17. Layer Normalization

Layer normalization normalizes activations within the representation.

It helps make Transformer training more stable.

The exact placement of normalization depends on the architecture. GPT-style implementations commonly use normalization around Transformer sub-layers.

---

# 18. Vocabulary Logits

After all Transformer blocks, the model produces a representation for the current position.

A final linear transformation maps that representation to scores for every token in the vocabulary.

These raw scores are called **logits**.

Conceptually:

```text
Transformer output
       ↓
Linear layer
       ↓
Vocabulary logits
       ↓
Softmax
       ↓
Token probabilities
```

---

# 19. Softmax

Softmax converts logits into probabilities.

Example:

```text
Token A → 0.10
Token B → 0.70
Token C → 0.20
```

The probabilities add up to:

```text
1.0
```

The model can then select or sample the next token.

---

# 20. Next-Token Prediction

The central language-modeling objective is:

```text
Predict the next token.
```

Example:

```text
Input:
The sky is

Target:
blue
```

For a longer sequence:

```text
The → next token = sky
The sky → next token = is
The sky is → next token = blue
```

The same sequence can provide many training examples.

---

# 21. Training

A simplified training loop:

```text
Text
 ↓
Tokenization
 ↓
Input / Target sequences
 ↓
GPT
 ↓
Next-token predictions
 ↓
Loss
 ↓
Backpropagation
 ↓
Gradient update
 ↓
Repeat
```

The model gradually adjusts its parameters to improve next-token predictions.

---

# 22. Loss Function

The model produces a probability distribution for the next token.

The correct next token is compared with the prediction.

Language models commonly use **cross-entropy loss**.

If the correct token gets high probability:

```text
Loss ↓
```

If the correct token gets low probability:

```text
Loss ↑
```

The training objective is to minimize this loss.

---

# 23. Parameters

Parameters are learned numerical values inside the neural network.

They include the weights used by:

```text
Embeddings
Attention
Feed-forward networks
Output layers
```

GPT-3 was reported to contain:

```text
175 billion parameters
```

---

# 24. Scaling

One of the central ideas behind GPT-3 is **scaling**.

Researchers increased:

```text
Model size
+
Training data
+
Compute
```

and observed improvements in language-model performance.

The broader idea is that sufficiently large Transformer language models can acquire increasingly capable behavior from the same basic next-token prediction objective.

---

# 25. GPT-3 Model Sizes

The GPT-3 paper described several model sizes.

From smallest to largest:

```text
125M
350M
760M
1.3B
2.7B
6.7B
13B
175B
```

The largest GPT-3 model had approximately:

```text
175 billion parameters
```

---

# 26. Zero-Shot Learning

**Zero-shot** means performing a task without providing examples of the task in the prompt.

Example:

```text
Translate English to French:

Hello →
```

The model may produce:

```text
Bonjour
```

without a demonstration example.

---

# 27. One-Shot Learning

**One-shot** means providing one example.

Example:

```text
English: Hello
French: Bonjour

English: Thank you
French:
```

The model uses the example to infer the task.

---

# 28. Few-Shot Learning

**Few-shot** means providing several examples.

Example:

```text
English: Hello
French: Bonjour

English: Thank you
French: Merci

English: Goodbye
French:
```

The model can infer the pattern from the examples.

---

# 29. In-Context Learning

Few-shot behavior is commonly described as **in-context learning**.

The model receives examples in the prompt and uses them to perform the task.

Important distinction:

```text
Few-shot prompting
        ≠
Fine-tuning
```

With normal prompting, the model's parameters are not permanently changed.

---

# 30. Fine-Tuning vs In-Context Learning

### In-context learning

```text
Examples
 ↓
Prompt
 ↓
Pre-trained model
 ↓
Answer
```

Parameters:

```text
Not updated
```

### Fine-tuning

```text
Training dataset
 ↓
Training
 ↓
Parameter updates
 ↓
Fine-tuned model
```

Parameters:

```text
Updated
```

---

# 31. GPT Can Perform Many Tasks

A single large language model can be prompted for different tasks.

Examples include:

```text
Translation
Summarization
Question answering
Classification
Text completion
Code generation
```

This is possible because the model learns broad patterns from large-scale text.

---

# 32. Why Next-Token Prediction Can Produce Complex Abilities

The training objective looks simple:

```text
Predict the next token.
```

But doing this well requires learning many patterns in language.

The model can learn relationships involving:

```text
Grammar
Facts present in training data
Writing patterns
Language relationships
Code patterns
Task formats
Longer-range context
```

Therefore, a model trained with next-token prediction can later be prompted to perform many tasks.

---

# 33. GPT Generation / Inference

During inference, the model receives a prompt.

Example:

```text
The capital of France is
```

The model calculates probabilities for the next token:

```text
Paris → high probability
London → lower probability
Berlin → lower probability
...
```

A token is selected.

Then:

```text
The capital of France is Paris
```

is fed back into the model to predict the next token.

This repeats.

---

# 34. Training vs Inference

## Training

```text
Input
 ↓
Prediction
 ↓
Loss
 ↓
Backpropagation
 ↓
Update weights
```

The model learns.

## Inference

```text
Prompt
 ↓
Prediction
 ↓
Select token
 ↓
Add token
 ↓
Predict again
 ↓
Repeat
```

The model generates text.

Ordinary inference does not update the model's parameters.

---

# 35. GPT-3's Important Contribution

GPT-3 demonstrated that increasing the scale of an autoregressive Transformer language model could produce strong performance across many tasks with little or no task-specific training.

Its particularly important result was the strength of **zero-shot, one-shot, and few-shot** performance.

---

# 36. Complete GPT Mental Model

```text
                 GPT
                  │
          Decoder-only Transformer
                  │
          ┌───────┴───────┐
          │               │
       Tokenizer       Parameters
          │               │
       Token IDs       Learned during
          │              training
          ↓
      Embeddings
          ↓
 Positional Information
          ↓
 ┌─────────────────────┐
 │ Transformer Block   │
 │                     │
 │ Masked Attention    │
 │ Feed-Forward        │
 │ Residual Connections│
 │ Normalization       │
 └─────────────────────┘
          ↓
     Repeat blocks
          ↓
       Logits
          ↓
       Softmax
          ↓
  Next-token probability
          ↓
     Select token
          ↓
   Add to sequence
          ↓
        Repeat
```

---

# 37. Key Definitions

| Concept | Meaning |
|---|---|
| GPT | Generative Pre-trained Transformer |
| Autoregressive | Predicts the next token using previous context |
| Decoder-only | Transformer architecture using decoder blocks without an encoder |
| Token | Small unit of text processed by the model |
| Token ID | Integer identifier for a token |
| Embedding | Learned vector representation |
| Self-attention | Allows token representations to use information from other tokens |
| Causal attention | Attention that prevents access to future tokens |
| Query | Representation used to find relevant information |
| Key | Representation used for matching |
| Value | Information passed through attention |
| Multi-head attention | Several attention mechanisms operating in parallel |
| Logits | Raw scores produced for vocabulary tokens |
| Softmax | Converts logits into probabilities |
| Parameter | Learned numerical value in the model |
| Pretraining | Broad initial training on large amounts of data |
| Fine-tuning | Further training that changes model parameters |
| Zero-shot | Task with no examples in the prompt |
| One-shot | Task with one example |
| Few-shot | Task with several examples |
| In-context learning | Using information/examples in the prompt without updating weights |

---

# 38. Most Important Points to Remember

- GPT stands for **Generative Pre-trained Transformer**.
- GPT is an **autoregressive, decoder-only Transformer**.
- GPT predicts the **next token**.
- Text is tokenized before entering the model.
- Token IDs are converted into learned embeddings.
- Positional information tells the model about token order.
- GPT uses **causal/masked self-attention**.
- Multi-head attention lets the model learn different relationships.
- Transformer blocks contain attention and feed-forward sub-layers with normalization and residual connections.
- The output layer produces vocabulary logits.
- Softmax converts logits into token probabilities.
- Cross-entropy is used to train next-token prediction.
- Backpropagation updates the model's parameters.
- GPT-3's largest model had **175 billion parameters**.
- GPT-3 showed strong **zero-shot, one-shot, and few-shot** capabilities.
- Few-shot prompting does not permanently change model weights.
- Fine-tuning changes model weights.
- During inference, GPT generates text **one token at a time**.

---

# 39. One-Sentence Summary

> **GPT is a decoder-only Transformer trained mainly by next-token prediction; by scaling the model, data, and computation, GPT-3 demonstrated that one pretrained model could perform many tasks through zero-shot, one-shot, and few-shot prompting.**
