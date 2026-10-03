# Lecture 6 --- Stages of Building an LLM from Scratch

## 1. Purpose of This Lecture

roadmap for the complete process of building a Large Language Model (LLM) from scratch.


1.  **Stage 1 --- Build the basic LLM**
2.  **Stage 2 --- Pretrain the LLM and create a foundation model**
3.  **Stage 3 --- Fine-tune the pretrained model for specific tasks**

------------------------------------------------------------------------

# 2. The Three Stages

``` text
STAGE 1
Data Preparation & Sampling
        ↓
Attention Mechanism
        ↓
LLM Architecture
        ↓
Building an LLM
        ↓
STAGE 2
Pretraining
        ↓
Training Loop
        ↓
Model Evaluation
        ↓
Load Pretrained Weights
        ↓
Foundation Model
        ↓
STAGE 3
Fine-tuning
        ↓
Specific Applications
        ├── Classifier
        ├── Personal Assistant / Chat Model
        └── Other specialized applications
```

------------------------------------------------------------------------

# 3. Stage 1 --- Building the LLM

The first stage focuses on understanding and implementing the **basic
mechanisms that make an LLM work**.

It contains three major areas:

-   Data preparation and sampling
-   Attention mechanism
-   LLM architecture

The goal is not yet to train a huge production-scale model.

The goal is to understand the components and implement a small GPT-style
LLM.

------------------------------------------------------------------------

## 3.1 Data Preparation and Sampling

Before an LLM can learn from text, raw text must be converted into a
form that a neural network can process.

### Main steps

-   Collect text data
-   Clean and prepare the text
-   Tokenize the text
-   Convert tokens into numerical IDs
-   Create input-target sequences
-   Create batches for training
-   Sample useful training examples

### Tokenization

Tokenization breaks text into smaller units called **tokens**.

For example:

``` text
"I love AI"
       ↓
["I", "love", "AI"]
```

A real tokenizer may use words, subwords, characters, or other token
units.

The tokens are then represented by numerical IDs:

``` text
["I", "love", "AI"]
        ↓
[25, 431, 92]
```

The exact IDs depend on the tokenizer vocabulary.

### Vector Embeddings

Token IDs are converted into vectors called **embeddings**.

``` text
Token ID
   ↓
Embedding layer
   ↓
Vector
```

The vector gives the neural network a numerical representation that can
be processed by the model.

------------------------------------------------------------------------

# 4. Stage 1 --- Attention Mechanism

The second major part of Stage 1 is the **attention mechanism**.

Attention allows the model to determine which parts of the input are
important when processing a particular token.

A major topic is **self-attention**.

------------------------------------------------------------------------

## 4.1 Query, Key, and Value

Self-attention uses three representations:

``` text
Query (Q)
Key   (K)
Value (V)
```

Conceptually:

-   **Query** represents what the current token is looking for.
-   **Key** represents what information each token contains for
    matching.
-   **Value** contains the information that can be passed forward.

The attention mechanism compares queries with keys to calculate how
strongly tokens should attend to one another.

A simplified attention calculation is:

``` text
Attention(Q, K, V)
        =
softmax(QKᵀ / √dₖ)V
```

Where:

-   `Q` = Query matrix
-   `K` = Key matrix
-   `V` = Value matrix
-   `dₖ` = dimension of the key vectors
-   `softmax` = converts scores into normalized attention weights

------------------------------------------------------------------------

## 4.2 Other Attention Topics

The Stage 1 attention section includes concepts such as:

-   Self-attention
-   Query, Key, Value
-   Attention scores
-   Attention weights
-   Positional information
-   Positional encoding / embeddings
-   Vector embeddings
-   Causal attention
-   Multi-head attention

These concepts form the core of the Transformer architecture used by
GPT-style models.

------------------------------------------------------------------------

# 5. Stage 1 --- LLM Architecture

After understanding data preparation and attention, the next step is to
understand how these components are combined into an LLM.

The course focuses on a **GPT-style Transformer architecture**.

Important components include:

-   Token embeddings
-   Positional embeddings
-   Transformer blocks
-   Self-attention
-   Multi-head attention
-   Feed-forward neural networks
-   Layer normalization
-   Activation functions
-   Shortcut / residual connections
-   Output layer

A simplified view:

``` text
Input Text
    ↓
Tokenization
    ↓
Token IDs
    ↓
Token Embeddings
    +
Positional Information
    ↓
Transformer Blocks
    ├── Multi-Head Self-Attention
    ├── Feed-Forward Network
    ├── Layer Normalization
    └── Residual Connections
    ↓
Output Representation
    ↓
Next-Token Prediction
```

------------------------------------------------------------------------

# 6. What Is the Goal of Stage 1?

The main goal is:

> Understand the basic mechanism of an LLM and implement a small LLM
> from scratch.

This means understanding **how the individual pieces work**, rather than
treating an LLM as a black box.

At the end of this stage, we should understand how:

``` text
Text
 ↓
Tokens
 ↓
Embeddings
 ↓
Attention
 ↓
Transformer blocks
 ↓
LLM
 ↓
Next-token prediction
```

works.

------------------------------------------------------------------------

# 7. Stage 2 --- Pretraining

Once the LLM architecture has been built, the next step is to train it.

This is called **pretraining**.

The model is trained on a large amount of **unlabeled text data**.

``` text
Large unlabeled text dataset
          ↓
      LLM training
          ↓
   Foundation model
```

The model learns general patterns of language by predicting the next
token.

------------------------------------------------------------------------

## 7.1 Next-Token Prediction

GPT-style language models are trained primarily using next-token
prediction.

Example:

``` text
Input:
"The cat is"

Target:
"sleeping"
```

The model tries to predict the next token.

Another example:

``` text
"The capital of France is"
                    ↓
                  "Paris"
```

During training, the model repeatedly makes predictions and adjusts its
parameters based on the error.

------------------------------------------------------------------------

# 8. Training Loop

Stage 2 includes implementing the training loop.

A simplified training loop looks like:

``` text
Input batch
    ↓
Forward pass
    ↓
Predictions
    ↓
Calculate loss
    ↓
Backpropagation
    ↓
Update model weights
    ↓
Repeat
```

### Important parts

-   Forward pass
-   Loss calculation
-   Backpropagation
-   Gradient calculation
-   Parameter updates
-   Repeating over many batches and epochs

------------------------------------------------------------------------

# 9. Model Evaluation

Training alone is not enough.

We also need to evaluate how well the model is learning.

Important ideas include:

-   Training loss
-   Validation loss
-   Loss curves
-   Perplexity
-   Generated text samples
-   Qualitative evaluation

### Perplexity

Perplexity is commonly used to evaluate language models.

A lower perplexity generally indicates that the model assigns higher
probability to the observed text, although interpretation depends on the
dataset and tokenizer.

------------------------------------------------------------------------

# 10. Loading Pretrained Weights

Training a large LLM from scratch can require enormous computational
resources.

Instead of always training a large model from zero, we can use an
already pretrained model.

The process is:

``` text
Existing pretrained model
        ↓
Load pretrained weights
        ↓
Use the model
        ↓
Fine-tune for a specific task
```

The lecture discusses loading pretrained weights as part of Stage 2.

------------------------------------------------------------------------

# 11. Foundation Model

After pretraining, we obtain a **foundation model**.

A foundation model has learned general patterns from large-scale data
and can serve as the starting point for further adaptation.

``` text
Large unlabeled dataset
        ↓
     Pretraining
        ↓
 Foundation model
        ↓
    Fine-tuning
        ↓
Specific application
```

The foundation model is not necessarily the final application.

It is a base model that can be adapted to different tasks.

------------------------------------------------------------------------

# 12. Stage 3 --- Fine-tuning

The third stage is about adapting the pretrained model to a specific
purpose.

``` text
Pretrained LLM
      ↓
 Fine-tuning
      ↓
Specific task
```

Fine-tuning uses additional data designed for the target task.

------------------------------------------------------------------------

# 13. Fine-tuning for Classification

One possible application is text classification.

Example:

``` text
Text
 ↓
Pretrained LLM
 ↓
Fine-tuning
 ↓
Classifier
```

Possible tasks include:

-   Spam detection
-   Sentiment classification
-   Topic classification
-   Other text classification tasks

For classification, the model is adapted so that its output can
represent the desired class labels.

------------------------------------------------------------------------

# 14. Fine-tuning for a Personal Assistant

Another application is creating a personal assistant or chat model.

This generally involves an **instruction dataset**.

Conceptually:

``` text
Instruction dataset
        ↓
Pretrained LLM
        ↓
Instruction fine-tuning
        ↓
Assistant / Chat model
```

The model learns to respond to instructions rather than simply
continuing arbitrary text.

------------------------------------------------------------------------

# 15. The Complete LLM Development Pipeline

The complete roadmap can be remembered as:

``` text
                 BUILDING AN LLM

STAGE 1
Data Preparation & Sampling
          ↓
Attention Mechanism
          ↓
LLM Architecture
          ↓
Basic LLM
          │
          ↓
STAGE 2
Pretraining
          ↓
Training Loop
          ↓
Model Evaluation
          ↓
Load / Use Pretrained Weights
          ↓
Foundation Model
          │
          ↓
STAGE 3
Fine-tuning
          ↓
     ┌────┴──────────────┐
     ↓                   ↓
Classifier        Personal Assistant
                     / Chat Model
```

------------------------------------------------------------------------

# 16. Important Difference: Pretraining vs Fine-tuning

  -----------------------------------------------------------------------
  Pretraining                         Fine-tuning
  ----------------------------------- -----------------------------------
  Comes before task-specific          Comes after a pretrained model
  adaptation                          

  Uses large-scale data               Uses task-specific data

  Usually uses unlabeled text for     Uses data designed for a particular
  GPT-style next-token training       task

  Learns general language patterns    Adapts the model to a particular
                                      behavior/task

  Produces a foundation model         Produces a task-specific
                                      model/application
  -----------------------------------------------------------------------

### Simple way to remember

``` text
Pretraining
= Learn general language patterns

Fine-tuning
= Adapt the learned model for a specific purpose
```

------------------------------------------------------------------------

# 17. Why LLMs Can Perform Different Tasks

An interesting point discussed in the lecture is that GPT-style LLMs are
trained primarily to predict the next token.

Yet, after large-scale pretraining, they can exhibit capabilities such
as:

-   Text classification
-   Translation
-   Summarization
-   Question answering
-   Text generation

The important idea is that the model is not necessarily trained with a
separate complete model for every one of these abilities.

Large-scale pretraining can produce a general language model that can
later be adapted to many tasks.

------------------------------------------------------------------------

# 18. GPT Model Progression Mentioned in the Lecture

The lecture briefly recaps the progression of GPT-style models:

``` text
GPT
 ↓
GPT-2
 ↓
GPT-3
 ↓
GPT-4
```

The lecture highlights GPT-3's scale, including its **175 billion
parameters**.

It also discusses the very high computational cost associated with
pretraining large models.

------------------------------------------------------------------------

# 19. What the Playlist Will Cover Next

The lecture marks the transition from introductory theory to hands-on
implementation.

The upcoming practical work begins with **Stage 1: Data Preparation and
Sampling**.

The next topics include:

-   Loading a text dataset
-   Counting characters
-   Tokenization
-   Creating tokens from text
-   Preparing data for training
-   Creating input-target pairs
-   Building the data pipeline
-   Starting the implementation in Jupyter notebooks

The lecture explains that the series will combine:

``` text
Theory
  +
Practical implementation
  =
Understanding how LLMs work
```

------------------------------------------------------------------------

# 20. Key Takeaways

-   An LLM can be understood through a sequence of major development
    stages.
-   The roadmap is divided into three main stages.
-   **Stage 1** focuses on data preparation, attention, and the LLM
    architecture.
-   **Stage 2** focuses on pretraining, training, evaluation, and
    pretrained weights.
-   Pretraining produces a **foundation model**.
-   **Stage 3** focuses on fine-tuning the foundation model for specific
    applications.
-   Fine-tuning can be used for tasks such as classification and
    instruction-following assistants.
-   GPT-style models learn through next-token prediction during their
    core pretraining process.
-   Understanding the components from scratch helps avoid treating LLMs
    as a black box.
-   The next practical step is working with text data and building the
    data-preparation pipeline.

------------------------------------------------------------------------

# 21. One-Line Summary

> **Build the basic LLM → pretrain it to create a foundation model →
> fine-tune it for a specific application.**

------------------------------------------------------------------------

## Learning Roadmap

``` text
STAGE 1
├── Data preparation & sampling
├── Tokenization
├── Embeddings
├── Attention
├── Query / Key / Value
├── Positional information
├── Multi-head attention
└── LLM / Transformer architecture

STAGE 2
├── Pretraining
├── Training loop
├── Loss
├── Backpropagation
├── Model evaluation
├── Perplexity
└── Loading pretrained weights
        ↓
   Foundation Model

STAGE 3
├── Fine-tuning
├── Classification
└── Instruction fine-tuning
        ↓
   Personal Assistant / Chat Model
```
