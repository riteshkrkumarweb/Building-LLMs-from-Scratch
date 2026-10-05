# LLM Tokenizer from Scratch in Python

## 1. What is Tokenization?

**Tokenization** is the process of breaking raw text into smaller pieces called **tokens**.

A token can be:

- a word
- part of a word
- punctuation
- a special token

Example:

```text
"Hello, world!"
```

can be split into:

```text
["Hello", ",", "world", "!"]
```

The exact way text is split depends on the tokenizer.

### Why do we need tokenization?

Neural networks do not directly process normal text such as:

```text
"I love AI"
```

The text must first be represented numerically.

The basic pipeline is:

```text
Raw Text
   ↓
Tokens
   ↓
Token IDs
   ↓
Token Embeddings
   ↓
LLM
```

Tokenization is therefore one of the first preprocessing steps in an LLM.

---

# 2. Two Main Steps

The lecture explains tokenization as two important steps:

```text
Step 1 → Split text into tokens
Step 2 → Convert tokens into token IDs
```

For example:

```text
Text:
"Hello world"

Tokens:
["Hello", "world"]

Token IDs:
[12, 57]
```

The actual numbers depend on the vocabulary created by the tokenizer.

---

# 3. Loading the Text Dataset

A text dataset is loaded into Python as a string.

Example:

```python
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
```

### What happens here?

- `open()` opens the file.
- `"r"` means read mode.
- `encoding="utf-8"` specifies the text encoding.
- `f.read()` reads the complete file.
- The result is stored in `raw_text`.

So:

```python
raw_text
```

contains the complete text as one Python string.

---

# 4. Splitting Text into Tokens

A simple tokenizer can use regular expressions (`re`) to split text.

A typical pattern separates words, punctuation, and whitespace.

Example:

```python
import re

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
```

The regular expression looks for:

- punctuation
- `--`
- whitespace

After splitting, empty strings can be removed:

```python
preprocessed = [
    item.strip()
    for item in preprocessed
    if item.strip()
]
```

### Why use `strip()`?

`strip()` removes whitespace from the beginning and end of a string.

Example:

```text
"  hello  "
```

becomes:

```text
"hello"
```

---

# 5. Example of Tokenization

Consider:

```text
"Hello, world!"
```

A simple tokenizer may produce:

```text
["Hello", ",", "world", "!"]
```

Notice that punctuation can become separate tokens.

This is useful because punctuation can carry information for the model.

---

# 6. Why Not Just Split on Spaces?

A very simple approach would be:

```python
text.split()
```

For:

```text
"Hello, world!"
```

the result would be:

```text
["Hello,", "world!"]
```

This keeps punctuation attached to words.

A more useful tokenizer can separate them:

```text
["Hello", ",", "world", "!"]
```

This gives the tokenizer more control over individual pieces of text.

---

# 7. Creating a Vocabulary

After tokenizing the text, we need a vocabulary.

A **vocabulary** is a collection of unique tokens known by the tokenizer.

Example:

```text
Tokens:

["Hello", ",", "world", "!", "Hello"]
```

Unique vocabulary:

```text
["!", ",", "Hello", "world"]
```

Each unique token receives a unique integer ID.

For example:

```text
!       → 0
,       → 1
Hello   → 2
world   → 3
```

The exact IDs are arbitrary; they simply provide a numerical representation for the tokens.

---

# 8. Sorting the Vocabulary

A common implementation sorts the unique tokens:

```python
all_words = sorted(set(preprocessed))
```

### What does `set()` do?

It removes duplicates.

Example:

```python
["cat", "dog", "cat"]
```

becomes:

```python
{"cat", "dog"}
```

### What does `sorted()` do?

It sorts the values into a predictable order.

Then:

```python
vocab = {token: integer for integer, token in enumerate(all_words)}
```

creates a dictionary mapping tokens to IDs.

---

# 9. Token → Token ID

Suppose the vocabulary is:

```python
{
    "!": 0,
    ",": 1,
    "Hello": 2,
    "world": 3
}
```

Then:

```text
Hello → 2
,     → 1
world → 3
!     → 0
```

So:

```text
"Hello, world!"
```

becomes:

```text
[2, 1, 3, 0]
```

The important idea is:

```text
Token → Integer ID
```

---

# 10. Why Token IDs Are Needed

A neural network works with numerical data.

It cannot directly receive:

```text
["Hello", ",", "world", "!"]
```

Instead, the tokenizer converts the tokens into integers:

```text
[2, 1, 3, 0]
```

These IDs are later converted into vectors through an **embedding layer**.

The complete flow is:

```text
Text
 ↓
Tokens
 ↓
Token IDs
 ↓
Embedding Vectors
 ↓
Transformer
```

---

# 11. Building a Simple Tokenizer Class

Instead of writing separate code every time, we can create a class.

Example structure:

```python
class SimpleTokenizerV1:

    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {
            i: s for s, i in vocab.items()
        }

    def encode(self, text):
        # text → token IDs
        pass

    def decode(self, ids):
        # token IDs → text
        pass
```

The class maintains two mappings:

```text
Token → ID
```

and:

```text
ID → Token
```

---

# 12. Why Do We Need Two Dictionaries?

### Encoding

Encoding means:

```text
Text → Token IDs
```

So we need:

```python
str_to_int
```

Example:

```text
"Hello" → 2
```

### Decoding

Decoding means:

```text
Token IDs → Text
```

So we need:

```python
int_to_str
```

Example:

```text
2 → "Hello"
```

Therefore:

```text
str_to_int
    ↓
Text → IDs

int_to_str
    ↓
IDs → Text
```

---

# 13. Encoding

A simple encoding process can be thought of as:

```python
def encode(self, text):
    preprocessed = re.split(
        r'([,.:;?_!"()\']|--|\s)',
        text
    )

    preprocessed = [
        item.strip()
        for item in preprocessed
        if item.strip()
    ]

    ids = [self.str_to_int[s] for s in preprocessed]

    return ids
```

The process is:

```text
Input text
   ↓
Split into tokens
   ↓
Remove unnecessary whitespace
   ↓
Look up each token in vocabulary
   ↓
Return token IDs
```

---

# 14. Decoding

Decoding performs the reverse operation.

```python
def decode(self, ids):
    text = " ".join(
        self.int_to_str[i]
        for i in ids
    )

    return text
```

The basic idea is:

```text
IDs
 ↓
Tokens
 ↓
Text
```

For example:

```text
[2, 1, 3, 0]
```

could become:

```text
["Hello", ",", "world", "!"]
```

and then:

```text
"Hello , world !"
```

A production tokenizer needs additional handling to reconstruct spacing and punctuation correctly.

---

# 15. The Encode → Decode Cycle

A tokenizer should support both directions:

```text
Text
 ↓
encode()
 ↓
Token IDs
 ↓
decode()
 ↓
Text
```

Example:

```text
"Hello world"
      ↓
[12, 57]
      ↓
"Hello world"
```

This makes it possible to move between human-readable text and the numerical representation used by the model.

---

# 16. Special Tokens

Real LLM tokenizers need special tokens in addition to normal text tokens.

Special tokens are reserved tokens that provide information about the structure or boundaries of the input.

Examples include:

```text
<|endoftext|>
<|unk|>
```

Depending on the tokenizer or model, different special tokens may be used.

---

# 17. Unknown Token

An unknown-token marker can be used when a token is not present in the vocabulary.

For example:

```text
<|unk|>
```

means approximately:

```text
unknown token
```

Suppose the vocabulary does not contain:

```text
"supercalifragilistic"
```

The tokenizer could represent it using an unknown-token token if that tokenizer uses an `<|unk|>` mechanism.

---

# 18. End-of-Text Token

A token such as:

```text
<|endoftext|>
```

can indicate a boundary or the end of a text sequence.

For example:

```text
Document A <|endoftext|> Document B
```

The special token helps the model distinguish separate pieces of text.

The exact meaning and usage of special tokens depends on the model/tokenizer design.

---

# 19. Adding Special Tokens to the Vocabulary

Special tokens must have their own IDs.

Conceptually:

```python
special_tokens = [
    "<|endoftext|>",
    "<|unk|>"
]
```

Then they can be added to the vocabulary so that they can also be encoded into integers.

The vocabulary therefore contains:

```text
Normal tokens
+
Special tokens
```

---

# 20. Important Difference: Tokenization vs Embedding

These two concepts should not be confused.

### Tokenization

Converts:

```text
Text → Token IDs
```

Example:

```text
"Hello" → 42
```

### Embedding

Converts:

```text
Token ID → Vector
```

Example:

```text
42 → [0.21, -0.73, 0.14, ...]
```

So the complete pipeline is:

```text
"Hello"
   ↓
Tokenizer
   ↓
42
   ↓
Embedding Layer
   ↓
[0.21, -0.73, 0.14, ...]
   ↓
Transformer
```

The tokenizer itself does **not** create semantic embedding vectors.

---

# 21. Complete LLM Input Pipeline

The lecture fits into the larger LLM pipeline:

```text
                 RAW TEXT
                    │
                    ▼
              TOKENIZATION
                    │
                    ▼
                TOKENS
                    │
                    ▼
               TOKEN IDs
                    │
                    ▼
             TOKEN EMBEDDINGS
                    │
                    ▼
          POSITIONAL INFORMATION
                    │
                    ▼
              TRANSFORMER
                    │
                    ▼
             MODEL OUTPUT
                    │
                    ▼
            PREDICTED TOKEN ID
                    │
                    ▼
              DECODING
                    │
                    ▼
              HUMAN TEXT
```

---

# 22. Key Python Concepts Used

## `re.split()`

Used to split text according to a regular expression.

```python
re.split(pattern, text)
```

## `set()`

Removes duplicate values.

```python
set(["a", "b", "a"])
```

Result:

```text
{"a", "b"}
```

## `sorted()`

Sorts values.

```python
sorted(["dog", "cat"])
```

Result:

```text
["cat", "dog"]
```

## `enumerate()`

Provides both an index and a value.

Example:

```python
for i, token in enumerate(tokens):
    print(i, token)
```

## Dictionary comprehension

Used to construct mappings compactly.

```python
{token: i for i, token in enumerate(tokens)}
```

---

# 23. Simple Mental Model

Remember the tokenizer using this simple chain:

```text
SENTENCE
   ↓
TOKENS
   ↓
TOKEN IDs
   ↓
EMBEDDINGS
   ↓
LLM
```

For example:

```text
"I love AI."
```

might become:

```text
["I", "love", "AI", "."]
```

then:

```text
[15, 82, 41, 7]
```

then embedding vectors:

```text
[
  [ ... ],
  [ ... ],
  [ ... ],
  [ ... ]
]
```

The exact IDs and vectors depend on the tokenizer and model.

---

# 24. What This Lecture Teaches

The major concepts are:

- What tokenization is
- Why LLMs need tokenization
- Splitting raw text into tokens
- Handling punctuation and whitespace
- Creating a vocabulary
- Mapping tokens to integer IDs
- Mapping IDs back to tokens
- Building a simple tokenizer class
- Encoding text
- Decoding token IDs
- Understanding special tokens
- Understanding unknown tokens
- Understanding end-of-text tokens
- Connecting tokenization to the larger LLM pipeline

---

# 25. Important Takeaways

### 1. Text cannot be directly processed as normal strings

The text must first be converted into a numerical representation.

### 2. Tokenization comes before embeddings

```text
Text → Tokens → IDs → Embeddings
```

### 3. Token IDs are just identifiers

A token ID such as `42` does not itself mean that the token has a particular semantic meaning.

The embedding layer learns a vector representation associated with that ID.

### 4. A vocabulary defines the token-to-ID mapping

```text
Vocabulary
    ↓
Token → ID
```

### 5. Encoding and decoding are opposite directions

```text
Encoding:
Text → IDs

Decoding:
IDs → Text
```

### 6. Special tokens provide additional structure

They can represent concepts such as text boundaries or unknown tokens, depending on the tokenizer.

---

# 26. One-Line Summary

> **An LLM tokenizer converts human-readable text into tokens and then into token IDs that can be passed to the model's embedding layer.**

---

# 27. Connection to the Next Lecture

This lecture demonstrates a **simple tokenizer built from scratch**.

Modern GPT-style tokenizers use more advanced approaches, especially **Byte Pair Encoding (BPE)**.

The next lecture in the series is:

**Lecture 8: The GPT Tokenizer — Byte Pair Encoding**

The progression is:

```text
Simple Tokenizer
       ↓
Byte Pair Encoding (BPE)
       ↓
More efficient subword tokenization
       ↓
LLM input representation
```

---

## Final Revision Diagram

```text
                HUMAN TEXT
                     │
                     ▼
              ┌─────────────┐
              │  TOKENIZER  │
              └─────────────┘
                     │
                     ▼
              ["I", "love", "AI"]
                     │
                     ▼
              TOKEN IDs
              [15, 82, 41]
                     │
                     ▼
             EMBEDDING LAYER
                     │
                     ▼
             VECTOR REPRESENTATION
                     │
                     ▼
                TRANSFORMER
                     │
                     ▼
             NEXT TOKEN PREDICTION
                     │
                     ▼
                TOKEN ID
                     │
                     ▼
                 DECODER
                     │
                     ▼
                HUMAN TEXT
```