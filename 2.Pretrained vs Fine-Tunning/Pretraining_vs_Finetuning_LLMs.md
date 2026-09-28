# Pretraining LLMs vs Fine-tuning LLMs

## 1. The Big Picture

A modern LLM is generally developed in stages rather than being trained for every task from zero.

```text
Large Unlabeled Dataset
          ↓
     PRETRAINING
          ↓
   Base / Foundation Model
          ↓
      FINE-TUNING
          ↓
Task / Domain-Specific Model
```

The two important stages are:

* **Pretraining**
* **Fine-tuning**

---

## 2. What Is Pretraining?

**Pretraining** is the process of training a language model on a large and diverse dataset before adapting it to a particular task.

The model learns general patterns of language from a large collection of text.

```text
Large Text Dataset
        ↓
    Tokenization
        ↓
      Batches
        ↓
      LLM
        ↓
Prediction
        ↓
      Loss
        ↓
Backpropagation
        ↓
Updated Parameters
```

The data used for pretraining is commonly **unlabeled**.

---

## 3. Next-Token Prediction

A common objective for GPT-style models is **next-token prediction**.

Example:

```text
Input:
"The cat is"

Target:
"sleeping"
```

The model predicts a probability distribution for the next token.

```text
"The cat is"

      ↓

LLM

      ↓

sleeping → 0.45
running  → 0.20
eating   → 0.15
...
```

The prediction is compared with the target and a loss is calculated.

```text
Input
 ↓
LLM
 ↓
Prediction
 ↓
Loss
 ↓
Backpropagation
 ↓
Parameter update
```

Repeating this over huge amounts of text teaches the model general language patterns.

---

## 4. What Does the Model Learn During Pretraining?

Pretraining gives the model a broad language foundation.

It can learn patterns related to:

* Grammar
* Vocabulary
* Syntax
* Common linguistic structures
* Relationships between words and tokens
* Patterns in written text
* Some factual and world knowledge present in the training data
* Code or other modalities if included in the dataset

The model is not simply memorizing a dictionary. Its parameters are adjusted through training so the network can represent useful patterns in the data.

---

## 5. Why Is Pretraining Expensive?

Pretraining a large LLM can require:

```text
Huge Dataset
     +
Large Model
     +
Many Training Steps
     +
Powerful GPUs / Accelerators
     ↓
Large Compute Requirement
```

For learning, it is much more practical to implement and train a **small language model** to understand the same concepts.

---

## 6. What Is a Pretrained / Foundation Model?

After pretraining, the resulting model is often called a:

* **Base model**
* **Pretrained model**
* **Foundation model**

Conceptually:

```text
General Language Training
          ↓
     Base Model
          ↓
   Can be adapted to
   many applications
```

The base model has learned general language patterns but may not yet behave like a polished assistant.

---

## 7. What Is Fine-Tuning?

**Fine-tuning** means continuing the training of a pretrained model on a narrower dataset for a particular task, behavior, or domain.

```text
Pretrained Model
       ↓
Task-Specific Dataset
       ↓
Fine-Tuning
       ↓
Adapted Model
```

Instead of starting with random parameters, fine-tuning starts with parameters that have already learned useful representations.

---

## 8. Why Fine-Tune?

Suppose a model has learned general language:

```text
Pretraining
     ↓
General language capability
```

You may then want it to perform a particular task:

```text
General Model
     ↓
Fine-Tuning
     ↓
Specific Capability
```

Possible applications include:

* Classification
* Instruction following
* Domain-specific language
* Question answering
* Specialized text generation
* Task-specific behavior

---

## 9. Pretraining vs Fine-Tuning

| Aspect | Pretraining | Fine-Tuning |
|---|---|---|
| Main goal | Learn general language patterns | Adapt the model |
| Dataset | Large and diverse | Narrower / task-specific |
| Labels | Often unlabeled text | Often labeled or instruction data |
| Starting point | Usually randomly initialized model | Pretrained model |
| Compute | Usually very large | Usually much smaller |
| Scope | General | Specific |
| Output | Base/foundation model | Adapted model |

The exact setup can vary depending on the model and objective.

---

## 10. Example: Email Classification

### Train from scratch

```text
Random Model
     ↓
Spam Dataset
     ↓
Train
     ↓
Spam Classifier
```

### Fine-tune a pretrained model

```text
Pretrained Language Model
          ↓
Spam / Not-Spam Dataset
          ↓
      Fine-Tuning
          ↓
Spam Classifier
```

The pretrained model already contains useful language representations, so task-specific training can focus on adapting them.

---

## 11. Instruction Fine-Tuning

One important type of fine-tuning is **instruction fine-tuning**.

The dataset can contain examples of:

```text
Instruction → Response
```

For example:

```text
Instruction:
"Explain photosynthesis in simple words."

Response:
"Photosynthesis is the process by which plants..."
```

The model is trained on many such examples to improve instruction-following behavior.

---

## 12. Fine-Tuning for Classification

Fine-tuning can also be used for classification.

Example:

```text
Email:
"Congratulations! You won a prize."

Label:
Spam
```

Another example:

```text
Email:
"Your electricity bill is ready."

Label:
Not Spam
```

A model can be adapted using examples containing text and corresponding labels.

```text
Text + Label
     ↓
Fine-Tuning
     ↓
Classification Model
```

---

## 13. Zero-Shot, One-Shot, and Few-Shot Learning

These describe how examples can be provided to a model at inference time.

### Zero-Shot

The model receives a task without an example.

```text
Task:
"Classify this sentence as positive or negative."

Sentence:
"I love this movie."

Output:
Positive
```

### One-Shot

The model receives **one example**.

```text
Example:
"I love this phone." → Positive

New input:
"The camera is terrible."

Model predicts:
Negative
```

### Few-Shot

The model receives several examples.

```text
Example 1 → Label
Example 2 → Label
Example 3 → Label
Example 4 → Label

New input → Prediction
```

---

## 14. Zero-Shot vs Fine-Tuning

These concepts should not be confused.

### Zero-shot prompting

The model's parameters are **not updated**.

```text
Prompt
  ↓
Existing Model
  ↓
Answer
```

### Fine-tuning

The model's parameters **are updated using training data**.

```text
Dataset
  ↓
Pretrained Model
  ↓
Training
  ↓
Updated Parameters
  ↓
Fine-tuned Model
```

---

## 15. Complete Development Pipeline

```text
             LARGE TEXT DATA
                    ↓
               TOKENIZATION
                    ↓
                PRETRAINING
                    ↓
              BASE MODEL
                    ↓
          ┌─────────┴─────────┐
          ↓                   ↓
     Fine-Tuning        Direct Prompting
          ↓                   ↓
 Specialized Model      Zero/Few-Shot Use
          ↓                   ↓
          └─────────┬─────────┘
                    ↓
                Application
```

---

## 16. Relation to Building an LLM From Scratch

A useful learning structure is:

```text
Stage 1 — Architecture & Fundamentals
        ↓
Data preparation
        ↓
Tokenization
        ↓
Embeddings
        ↓
Attention
        ↓
Transformer architecture

Stage 2 — Pretraining
        ↓
Training loop
        ↓
Model evaluation
        ↓
Pretrained / base model

Stage 3 — Fine-Tuning
        ↓
Classification
        ↓
Instruction following
        ↓
Task-specific applications
```

---

## 17. Key Takeaways

* **Pretraining** teaches a model general patterns from a large dataset.
* Pretraining commonly uses large amounts of unlabeled text.
* GPT-style language models can use next-token prediction as the training objective.
* A pretrained model becomes a base/foundation model.
* **Fine-tuning** adapts a pretrained model to a narrower task, domain, or behavior.
* Fine-tuning generally starts from an already pretrained model.
* Instruction fine-tuning uses instruction-response examples.
* Classification fine-tuning can use text-label examples.
* Zero-shot, one-shot, and few-shot learning refer to how examples are provided at inference time.
* Zero-shot prompting does not update model parameters.
* Fine-tuning updates model parameters through additional training.
* Pretraining is generally much more computationally expensive than task-specific fine-tuning.

---

## 18. Mental Model

```text
                PRETRAINING
                     ↓
        "Learn language generally"
                     ↓
              BASE MODEL
                     ↓
               FINE-TUNING
                     ↓
         "Adapt to a specific goal"
                     ↓
          SPECIALIZED MODEL
```

Or simply:

```text
Pretraining = Learn broadly
Fine-tuning = Adapt specifically
Prompting   = Guide without updating parameters
```

---