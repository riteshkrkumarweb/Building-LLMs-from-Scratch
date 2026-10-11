# GPT Tokenization and Data Preparation for Autoregressive Training


## 1. The big picture

Before a GPT-style model can learn from text, the text must be converted into numbers and organized into examples.

**Pipeline:** raw text → tokenization → token IDs → input-target pairs → PyTorch `Dataset` → `DataLoader` batches → GPT training loop.

This lesson prepares data. It does **not**, by itself, train the GPT model.

## 2. What is tokenization?

Tokenization converts text into units called **tokens**. Depending on the tokenizer, a token may be a whole word, part of a word, punctuation, or another text unit.

For example, a simple word-level tokenizer might split:

```text
I love machine learning.
```

into:

```python
["I", "love", "machine", "learning", "."]
```

A real GPT tokenizer may split the same text differently. Tokens are not necessarily complete words.

Neural networks operate on numerical representations, so tokenizers map tokens to integer IDs. Example IDs below are illustrative:

```text
Tokens:    ["I", "love", "AI"]
Token IDs: [  10,   25,   87]
```

## 3. A simple vocabulary and tokenizer

A simple tokenizer can build a vocabulary from tokens found in a text corpus:

```python
all_words = sorted(set(preprocessed))
vocab = {token: i for i, token in enumerate(all_words)}
```

- `set(preprocessed)` removes duplicates.
- `sorted(...)` gives a consistent order.
- `enumerate(...)` assigns an integer ID to each token.
- `vocab` maps each token to its ID.

A tokenizer needs two directions:

- **Encoding:** text/tokens → token IDs.
- **Decoding:** token IDs → text/tokens.

If a word does not exist in a word-level vocabulary, direct lookup can raise a `KeyError`. A simple solution is to add a special unknown token such as `<|unk|>` and map unseen tokens to it.

A special token such as `<|endoftext|>` can mark the end of a document. Special tokens have meanings defined by the tokenizer/model setup.

## 4. Word-level, character-level, and subword tokenization

### Word-level
Each token is usually a complete word.

- Advantage: tokens can be easy to interpret and sequences may be shorter.
- Disadvantage: the vocabulary can become very large, and unseen words cause problems.

### Character-level
Each token is a character.

- Advantage: a small vocabulary can represent new words.
- Disadvantage: sequences become long and require more steps to process.

### Subword-level
Words can be represented as common whole words or smaller pieces.

- Balances vocabulary size and sequence length.
- Helps represent rare or previously unseen words.
- Is commonly used in modern language models.

## 5. Byte Pair Encoding (BPE)

**BPE** is a subword-tokenization method. In a simplified description, it starts with small units and repeatedly merges frequent adjacent pairs to create larger units.

1. Start with small units.
2. Count adjacent pairs.
3. Merge a frequent pair.
4. Update the counts.
5. Repeat until the desired vocabulary/merge limit is reached.

GPT-2's tokenizer can be loaded with `tiktoken`:

```python
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
```

Encode and decode:

```python
text = "Hello, how are you?"
token_ids = tokenizer.encode(text)
print(token_ids)

decoded_text = tokenizer.decode(token_ids)
print(decoded_text)
```

The exact token IDs depend on the tokenizer. Use the same tokenizer expected by the model.

Some tokenizers treat special tokens differently from ordinary text. For example:

```python
token_ids = tokenizer.encode(
    raw_text,
    allowed_special={"<|endoftext|>"}
)
```

This permits the specified special token to be interpreted as a special token by `tiktoken`.

## 6. GPT training is next-token prediction

GPT-style language models are trained to predict the next token from the tokens that came before it.

Suppose the token sequence is:

```text
[a, b, c, d, e]
```

For a context length of four, one training example is:

```text
Input (X):  [a, b, c, d]
Target (Y): [b, c, d, e]
```

The target sequence is shifted by one token:

- Given `a`, predict `b`.
- Given `a, b`, predict `c`.
- Given `a, b, c`, predict `d`.
- Given `a, b, c, d`, predict `e`.

A causal GPT model prevents each position from looking at future tokens when making its prediction.

### Code for one input-target pair

```python
context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size + 1]
```

If `enc_sample` is `[10, 20, 30, 40, 50]`, then:

```text
x = [10, 20, 30, 40]
y = [20, 30, 40, 50]
```

Both sequences have the same length, but the target is shifted by one token.

### Inspecting each prediction

```python
for i in range(1, context_size + 1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))
```

- `context` contains the tokens used as context.
- `desired` is the next token's integer ID.
- `tokenizer.decode([desired])` uses a list because `decode()` expects a sequence of token IDs. One token may be a word fragment, punctuation, or a space-related token—not necessarily a whole word.
- This example uses a growing context, not a fixed-size sliding window.
- `enc_sample` must contain at least `context_size + 1` token IDs.

## 7. What are `max_length` and `stride`?

These control how examples are extracted from the full token sequence.

- **`max_length`**: number of tokens in each input sequence.
- **`stride`**: how far the next window's starting position moves.

Example token IDs:

```text
[10, 20, 30, 40, 50, 60, 70, 80, 90]
```

With `max_length=4` and `stride=2`, input windows overlap:

```text
Window 1: [10, 20, 30, 40]
Window 2:           [30, 40, 50, 60]
Window 3:                     [50, 60, 70, 80]
```

With `stride=4`, windows do not overlap:

```text
Window 1: [10, 20, 30, 40]
Window 2:                     [50, 60, 70, 80]
```

The target window is shifted by one token:

```text
Input:  [10, 20, 30, 40]
Target: [20, 30, 40, 50]
```

A smaller stride generally produces more overlapping examples. Examples share tokens, but are not necessarily identical.

## 8. The custom `GPTDatasetV1` class

PyTorch provides a base `Dataset` class. `GPTDatasetV1` is a custom class because we must define how GPT input-target pairs are created and retrieved.

```python
import torch
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(
            txt,
            allowed_special={"<|endoftext|>"}
        )

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
```

### What each part does

- `class GPTDatasetV1(Dataset)`: defines a custom PyTorch dataset.
- `__init__(...)`: runs when the dataset object is created; tokenizes text and prepares examples.
- `self.input_ids`: stores input sequences.
- `self.target_ids`: stores next-token target sequences.
- `tokenizer.encode(...)`: converts raw text to token IDs.
- `range(0, len(token_ids) - max_length, stride)`: moves the window through the token sequence.
- `input_chunk`: selects `max_length` input tokens.
- `target_chunk`: selects the next-token sequence, shifted by one.
- `torch.tensor(...)`: converts IDs into PyTorch tensors.
- `__len__()`: returns the number of examples.
- `__getitem__(idx)`: returns the input and target for one example.

The loop bound ensures the target also has `max_length` tokens. If the text is too short to make one complete input-target pair, the dataset contains no examples.

## 9. What is predefined and what is custom?

| Name | Source | Role |
|---|---|---|
| `Dataset` | PyTorch | Base class for datasets |
| `GPTDatasetV1` | Custom code | Defines GPT training examples |
| `__init__`, `__len__`, `__getitem__` | Python special-method names | Initialize, report size, and retrieve examples; you write their logic |
| `DataLoader` | PyTorch | Batches and loads examples |
| `create_dataloader_v1()` | Custom helper function | Connects tokenizer, dataset, and DataLoader |
| `torch.tensor()` | PyTorch | Creates tensors |
| `tiktoken.get_encoding()` | `tiktoken` | Loads a tokenizer encoding |
| `iter()` and `next()` | Python built-ins | Create an iterator and retrieve its next item |

PyTorch supplies general-purpose tools, but it does not automatically know your particular input-target construction rules. That is why the custom dataset class is useful.

## 10. The `create_dataloader_v1()` helper

```python
def create_dataloader_v1(
    txt,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0
):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader
```

This helper:

1. Loads the GPT-2 tokenizer.
2. Creates a `GPTDatasetV1` object.
3. Wraps the dataset in a PyTorch `DataLoader`.
4. Returns the DataLoader.

Important parameters:

- `batch_size`: number of examples per batch.
- `shuffle=True`: shuffles the order of examples.
- `drop_last=True`: drops a final batch if it contains fewer than `batch_size` examples.
- `num_workers=0`: loads data in the main process, a simple starting point for learning and debugging.

Because this helper defaults to `drop_last=True`, if the dataset has fewer than `batch_size` examples, the DataLoader yields no batches. In that case, `next()` raises `StopIteration`. Use `drop_last=False` to keep a final incomplete batch.

## 11. Getting the first batch

```python
dataloader = create_dataloader_v1(
    raw_text,
    batch_size=8,
    max_length=4,
    stride=4,
    shuffle=False
)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)

print("Inputs:\n", inputs)
print("\nTargets:\n", targets)
```

- `batch_size=8`: asks for eight examples per batch.
- `max_length=4`: each input and target has four token IDs.
- `stride=4`: moves the window by four tokens, so input windows do not overlap.
- `shuffle=False`: preserves the dataset's example order.
- `iter(dataloader)`: creates an iterator over batches.
- `next(data_iter)`: retrieves the first available batch.
- `inputs, targets = ...`: unpacks the two tensors returned for that batch.

When a full batch is returned, shapes are normally:

```text
inputs:  (8, 4)
targets: (8, 4)
```

The first dimension is batch size; the second is tokens per example. Actual values depend on the source text and tokenizer.

## 12. Understanding batches visually

A batch of eight examples contains eight rows:

```text
Inputs (8 examples × 4 tokens)
[
  [x1, x2, x3, x4],
  [x1, x2, x3, x4],
  ...
]

Targets (8 examples × 4 next-token IDs)
[
  [x2, x3, x4, x5],
  [x2, x3, x4, x5],
  ...
]
```

These values are illustrative. Each target row is shifted by one token relative to its matching input row.

## 13. Training versus generation

### During training

1. The model receives input token IDs.
2. It predicts a probability distribution for the next token at each position.
3. A loss function—commonly cross-entropy—compares predictions with the target IDs.
4. Backpropagation calculates gradients.
5. The optimizer updates the model's parameters to improve future predictions.

A wrong prediction does not replace a token ID in the dataset. Instead, the model's parameters are adjusted.

### During generation

The model receives a prompt and predicts the next token. A token is selected from its predicted distribution, appended to the context, and the process repeats. Ordinary generation does not supply a known target token from a training dataset.

## 14. Common questions and clarifications

### Does one token always equal one word?
No. A token can be a complete word, part of a word, punctuation, or another text unit.

### Is `max_length` the total length of the original text?
No. It is the length of each input example and, in this dataset, each target example.

### Does `stride` change the size of each example?
No. `max_length` sets example length; `stride` sets how far the next window moves.

### Does `GPTDatasetV1` train the model?
No. It prepares and serves examples. A separate training loop runs the model, calculates loss, backpropagates, and updates parameters.

### Is `create_dataloader_v1()` predefined?
No. It is a helper function written for this workflow. `DataLoader` itself is predefined in PyTorch.

### Why does `tokenizer.decode([desired])` use brackets?
`desired` is one integer token ID. The decoder expects a sequence of IDs, so the ID is wrapped in a list. One ID may decode to only part of a word.

### What does `inputs, targets = next(data_iter)` return?
It returns one batch containing input sequences and their shifted target sequences. With a full batch, each tensor usually has shape `(batch_size, max_length)`.

## 15. Final takeaway

**Raw text → token IDs → shifted input-target examples → Dataset → DataLoader batches → GPT training loop.**

Remember:

- Tokenization converts text into integer IDs.
- GPT pretraining learns next-token prediction.
- Targets are the input sequence shifted by one token.
- `max_length` controls sequence length.
- `stride` controls window movement.
- `Dataset` defines how examples are stored and retrieved.
- `DataLoader` groups examples into batches.
- Data-loading code prepares the data; it does not itself train the model.
