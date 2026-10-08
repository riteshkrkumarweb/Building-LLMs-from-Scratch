#  The GPT Tokenizer — Byte Pair Encoding (BPE)


##  What is the main topic?

explains **Byte Pair Encoding (BPE)**, a subword tokenization method used by GPT-family language models and other LLM tokenizers.

The goal is to understand:

- Why simple word-level tokenization is not enough
- Why subword tokenization is useful
- What Byte Pair Encoding means
- How BPE repeatedly merges common pairs
- How BPE helps with unknown/out-of-vocabulary words
- How GPT-style tokenizers convert text into token IDs
- How `tiktoken` can be used in Python

---

#  Why do we need a better tokenizer?

A language model cannot directly process raw text.

For example:

```text
I love machine learning
```

must eventually become something like:

```text
tokens → token IDs → embeddings → neural network
```

A tokenizer decides how the text is split into tokens.

There are three important approaches:

1. Word-level tokenization
2. Character-level tokenization
3. Subword-level tokenization

---

#  Word-Level Tokenization

In word-level tokenization, a complete word is generally treated as one token.

Example:

```text
The fox chased the dog.
```

could become:

```text
["The", "fox", "chased", "the", "dog", "."]
```

Each token receives an integer ID.

### Problem

Consider a word that was not present in the tokenizer's vocabulary:

```text
electrification
```

If the tokenizer has never seen it, it may need an unknown-token representation such as:

```text
<UNK>
```

This creates an **out-of-vocabulary (OOV)** problem.

Another problem is vocabulary size. If every possible word is a separate token, the vocabulary can become very large.

---

#  Character-Level Tokenization

Character-level tokenization breaks text into individual characters.

Example:

```text
hello
```

becomes:

```text
["h", "e", "l", "l", "o"]
```

### Advantage

Almost any word can be represented because the tokenizer can fall back to individual characters.

### Problem

Sequences become much longer.

For example:

```text
machine
```

could require 7 character tokens instead of potentially fewer subword tokens.

Longer sequences mean more positions for the model to process.

---

#  Subword Tokenization

Subword tokenization sits between word-level and character-level tokenization.

A word can be represented as:

```text
whole word
```

or, when useful:

```text
subword + subword
```

For example, a word may be split into meaningful pieces such as:

```text
un + happy
```

or:

```text
token + ization
```

The exact split depends on the tokenizer.

### Main idea

Common words or common pieces can become single tokens, while uncommon words can be represented using smaller pieces.

---

#  What is Byte Pair Encoding (BPE)?

**Byte Pair Encoding is an algorithm that repeatedly merges frequently occurring adjacent pairs.**

The original BPE algorithm was introduced as a data-compression algorithm in 1994.

For LLM tokenization, the idea is adapted to create a useful vocabulary of subword tokens.

The basic process is:

```text
Find the most frequent pair
        ↓
Merge the pair
        ↓
Add the merged piece to the vocabulary
        ↓
Repeat
```

---

#  Simple BPE Example

Suppose our data contains:

```text
a a a b d a a a b a c
```

We look for the most frequent adjacent pair.

Suppose:

```text
a a
```

is the most frequent pair.

We merge it:

```text
aa
```

Now the sequence contains the new piece `aa`.

We then search again for the most frequent pair and continue merging.

Conceptually:

```text
individual pieces
       ↓
frequent pair
       ↓
merged piece
       ↓
another frequent pair
       ↓
another merged piece
```

The process continues until the chosen stopping condition is reached.

---

#  How BPE is Used for LLMs

For language models, BPE is used to create a vocabulary containing useful subword tokens.

A simplified intuition is:

### Common text

```text
the
```

may become one token because it occurs frequently.

### Less common text

A rare word may be represented using multiple pieces:

```text
rareword
   ↓
rare + word
```

The exact tokenization is determined by the tokenizer's learned vocabulary and merge rules.

---

#  Why Subwords Are Useful

BPE combines useful properties of word-level and character-level tokenization.

### Word-level problem

Vocabulary can become very large and unknown words are difficult to represent.

### Character-level problem

Sequences become very long.

### Subword solution

BPE can represent:

```text
common words → larger tokens
rare words   → smaller subword tokens
```

This gives a manageable vocabulary while still allowing unfamiliar words to be represented.

---

#   Out-of-Vocabulary Problem

One major advantage of subword tokenization is that an unfamiliar word does not necessarily need to become `<UNK>`.

For example, suppose the tokenizer knows:

```text
learn
ing
```

but encounters:

```text
learning
```

It can potentially represent it using known pieces:

```text
learn + ing
```

The exact split depends on the trained tokenizer.

This is why subword tokenization can handle many words that were not stored as complete vocabulary entries.

---

#  BPE and Root/Subword Information

Subword tokenization can preserve recurring pieces across different words.

For example, words may share pieces such as:

```text
play
playing
played
player
```

A tokenizer may learn reusable pieces that occur across these words.

This can make the vocabulary more efficient than treating every complete word as unrelated.

---

#  GPT Tokenization

GPT-style models use a form of BPE-based tokenization.

The lecture demonstrates GPT-2 tokenization and explains how text is converted into token IDs.

For example:

```text
"Hello, ..."
       ↓
Tokenizer
       ↓
[50256, ...]
```

The integer values are **token IDs**.

A token ID is simply an integer that identifies a particular token in the tokenizer's vocabulary.

---

#  Token ≠ Word

This is extremely important.

A token is **not necessarily one complete word**.

A token can represent:

- a complete word
- part of a word
- punctuation
- a space plus a word/piece
- other byte-level pieces

For example, a sentence might be tokenized conceptually as:

```text
"My name is Ritesh"
```

into:

```text
"My"
" name"
" is"
" R"
"ites"
"h"
```

The exact result depends on the encoding.

---

#  Token IDs

After tokenization, every token is mapped to an integer.

Example:

```text
"My"    → 5159
" name" → 836
" is"   → 374
" R"    → 432
"ites"  → 3695
"h"     → 71
```

So:

```text
Text
 ↓
Tokens
 ↓
Token IDs
```

The neural network works with these numerical representations rather than raw words.

---

#  `encode()` and `decode()`

A tokenizer generally needs two important operations.

### Encode

```text
text → token IDs
```

### Decode

```text
token IDs → text
```

Conceptually:

```text
"I love AI"
     ↓ encode
[40, ...]
     ↓ decode
"I love AI"
```

The goal is that decoding the encoded text reconstructs the original text.

---

#  `tiktoken`

The lecture introduces **`tiktoken`**, an OpenAI tokenizer library.

Install it with:

```bash
pip install tiktoken
```

Then:

```python
import tiktoken

encoding = tiktoken.get_encoding("gpt2")

text = "Hello, how are you?"

tokens = encoding.encode(text)

print(tokens)
```

This converts the text into token IDs.

You can decode them:

```python
text_again = encoding.decode(tokens)

print(text_again)
```

---

#  Seeing Individual Tokens

You can inspect what each token ID represents:

```python
for token in tokens:
    print(
        token,
        "->",
        encoding.decode_single_token_bytes(token)
    )
```

You may see output such as:

```text
5159 -> b'My'
836  -> b' name'
374  -> b' is'
```

The `b` means Python is displaying a **bytes object**.

To convert the bytes to normal text:

```python
piece = encoding.decode_single_token_bytes(token).decode("utf-8")
```

---

#  Encoding Name vs Model Name

This distinction is important when using `tiktoken`.

### Directly choose an encoding

```python
encoding = tiktoken.get_encoding("cl100k_base")
```

Here:

```text
"cl100k_base"
```

is an **encoding name**.

You are explicitly asking for that encoding.

### Choose by model

```python
encoding = tiktoken.encoding_for_model("MODEL_NAME")
```

Here you provide a **model name**, and `tiktoken` looks up the encoding associated with that model.

So:

```text
get_encoding()
        ↓
"I want this exact encoding."

encoding_for_model()
        ↓
"I am using this model; give me its encoding."
```

---

#  Important Vocabulary Concepts

### Vocabulary

The tokenizer's collection of available tokens.

Example:

```text
["the", "ing", "hello", ",", ...]
```

Each token has an ID.

### Token ID

An integer representing a token.

```text
"hello" → 15339
```

### Tokenizer

The system that converts text into tokens/token IDs and can convert token IDs back into text.

### Encoding

A specific set of tokenizer rules/vocabulary used to perform that conversion.

---

#  Why BPE Is Important for LLMs

BPE provides a useful compromise:

```text
Word-level
    ↓
Huge vocabulary + OOV problems

Character-level
    ↓
Very long sequences

BPE / Subword
    ↓
Manageable vocabulary + flexible representation
```

This makes subword tokenization highly useful for large language models.

---

#  Complete Pipeline

The lecture fits into the larger LLM pipeline:

```text
Raw Text
   ↓
Tokenizer
   ↓
Tokens
   ↓
Token IDs
   ↓
Token Embeddings
   ↓
Transformer
   ↓
Predictions
```

At this stage, the important thing is to understand that **tokenization happens before embeddings**.

---

#  Key Takeaways

1. Tokenization converts text into pieces that a model can work with.
2. Word-level tokenization treats words as tokens but has vocabulary and OOV problems.
3. Character-level tokenization avoids many OOV problems but creates long sequences.
4. Subword tokenization provides a middle ground.
5. BPE repeatedly merges frequently occurring pairs.
6. GPT-style tokenizers use BPE-based approaches.
7. A token is not necessarily a complete word.
8. Token IDs are integers representing tokens.
9. `encode()` converts text to token IDs.
10. `decode()` converts token IDs back to text.
11. `tiktoken` provides fast GPT-style tokenization implementations.
12. `get_encoding()` selects an encoding directly.
13. `encoding_for_model()` selects an encoding based on a model name.
14. BPE helps represent rare or unseen words using smaller known pieces.

---

## Simple Mental Model

Remember this:

```text
"I am learning AI"
          ↓
       Tokenizer
          ↓
   ["I", " am", " learning", " AI"]
          ↓
      Token IDs
          ↓
     Embeddings
          ↓
    Neural Network
```

And for BPE:

```text
Characters / bytes
        ↓
frequent pairs
        ↓
merge pairs
        ↓
subwords
        ↓
build vocabulary
        ↓
token IDs
```

**Core idea:**

> BPE learns useful subword pieces by repeatedly merging frequently occurring pairs, allowing a tokenizer to represent common text efficiently while still handling uncommon words.

---

