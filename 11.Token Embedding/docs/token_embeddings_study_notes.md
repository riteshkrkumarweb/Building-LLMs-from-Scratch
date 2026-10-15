# Token Embeddings — Study Notes

## 1. What Are Token Embeddings?

**Token embeddings** represent tokens as numerical vectors. A token may be a whole word, part of a word (a subword), or a character.

Other terms used:
- **Vector embeddings:** emphasizes that tokens are represented as vectors.
- **Word embeddings:** a common term, but less precise for modern language models because their tokens are not always complete words.

The purpose is to give a model numerical inputs that can learn useful relationships in language.

### The text-to-model pipeline

1. **Input text:** for example, `This is an example`.
2. **Tokenization:** split the text into tokens.
3. **Token IDs:** map each token to an integer ID from the tokenizer's vocabulary.
4. **Token embeddings:** use each token ID to retrieve a vector.
5. **Language-model processing:** the model processes the vectors and learns to predict text.

Modern GPT-style tokenizers often use subword tokenization, such as byte-pair encoding (BPE), so one word may be split into multiple tokens.

---

## 2. Why Not Use Token IDs Directly?

A tokenizer assigns an ID to each token. These IDs identify tokens, but their numerical values do **not** automatically express meaning.

For example, a vocabulary might assign:
- `cat` → 34
- `kitten` → −33 (illustrative only)
- `book` → 2.9 (illustrative only)

The numbers are identifiers, not meaningful measurements. The model cannot infer from those arbitrary numbers alone that *cat* and *kitten* are semantically related.

**Main limitation:** arbitrary token IDs do not directly encode semantic similarity.

> **Remember:** Token IDs answer “Which token is this?” Embeddings provide a learnable vector representation for that token.

---

## 3. Why Is One-Hot Encoding Not Enough?

One-hot encoding represents each vocabulary item with a vector containing one `1` and zeros everywhere else.

For a vocabulary of four tokens, a possible example is:

| Token | One-hot vector |
|---|---|
| dog | `[1, 0, 0, 0]` |
| puppy | `[0, 1, 0, 0]` |
| cat | `[0, 0, 1, 0]` |
| banana | `[0, 0, 0, 1]` |

These vectors distinguish tokens, but they do not naturally express that *dog* is more similar in meaning to *puppy* than to *banana*. The representations are equally distinct in this simple scheme.

One-hot vectors can be used as a mathematical way to select embedding rows, but using them directly as model representations does not provide learned semantic relationships.

---

## 4. Embeddings Can Represent Semantic Relationships

A vector has multiple numerical dimensions. During training, the model learns values that can help represent relationships between tokens.

The lecture uses a simplified example with five human-designed features:

1. Has a tail
2. Is edible
3. Has four legs
4. Makes a sound
5. Is a pet

Illustrative feature vectors might look like this:

| Item | Tail | Edible | Four legs | Makes sound | Pet |
|---|---:|---:|---:|---:|---:|
| Dog | High | Low | High | High | High |
| Cat | High | Low | High | High | High |
| Apple | Low | High | Low | Low | Low |
| Banana | Low | High | Low | Low | Low |

Dog and cat have similar values, as do apple and banana. Dog and banana have very different values.

This illustration shows the basic intuition: **similar vectors can represent similar things**.

**Important distinction:** the feature values above are hand-designed for teaching. Real token embeddings are not usually created by manually assigning features such as “has a tail.” A neural network learns the embedding values from training.

### Connection to CNNs

The lecture compares embeddings with convolutional neural networks (CNNs):
- Images contain spatial relationships between pixels and features.
- Text contains semantic relationships between tokens.
- A useful representation can help a model exploit information in the data instead of treating every input as an unrelated identifier.

---

## 5. How Can We Check Whether Embeddings Capture Meaning?

The lecture demonstrates a pretrained Word2Vec model trained on Google News data. Its embeddings have **300 dimensions** per word.

### 5.1 Word-vector arithmetic

A famous illustrative example is:

`king + woman − man ≈ queen`

The idea is to add the vector for *king* to the vector for *woman*, subtract the vector for *man*, and search for the vocabulary vector closest to the result.

In the lecture's demonstration, *queen* ranks as the closest result. This suggests that the embedding space has learned some useful relationships.

**Caution:** this is an illustrative behavior of a particular trained embedding model, not a guarantee that every embedding model will produce the same answer or that vector arithmetic always works.

### 5.2 Similarity between words

The lecture compares pairs such as:

- woman and man
- king and queen
- uncle and aunt
- boy and girl
- nephew and niece
- paper and water

Related pairs receive higher similarity scores in the demonstrated model than the unrelated pair *paper* and *water*.

A similarity score is a way to compare vectors. The exact score depends on the similarity measure and the model being used.

### 5.3 Distance and magnitude

The lecture also calculates the **magnitude of the difference** between two vectors. In notation, this is commonly written as:

\[
\|v_1-v_2\|
\]

Here, \(v_1\) and \(v_2\) are two vectors. Subtracting them produces a difference vector; its magnitude (or norm) measures the length of that difference.

In the lecture's example, the difference magnitude for *man* and *woman* is smaller than the magnitude for *semiconductor* and *earth*. This is used as another illustration of how vectors may reflect relationships.

**Interpret carefully:**
- A smaller distance can indicate that two embeddings are closer under the chosen distance metric.
- Distance is not a universal measure of meaning; its interpretation depends on the embedding model, the metric, and how the vectors were trained.
- Similarity and distance are related ideas, but they are not always interchangeable.

---

## 6. How Are Embeddings Created for an LLM?

The lecture explains that an LLM has a vocabulary and assigns an embedding vector to every token ID.

Two dimensions determine the shape of the embedding matrix:

1. **Vocabulary size:** number of token IDs in the vocabulary.
2. **Embedding dimension:** number of values in each token's vector.

### GPT-2 example from the lecture

The transcript gives these values for GPT-2:
- Vocabulary size: **50,257**
- Embedding dimension for the smallest model discussed: **768**

Therefore, the token embedding matrix has the shape:

`(50,257, 768)`

That means:
- 50,257 rows — one row per token ID.
- 768 columns — one value for each dimension of a token's embedding.

Each row is the vector associated with one token ID.

**Extra clarification:** the matrix contains \(50,257 \times 768 = 38,597,376\) embedding values. The values are trainable parameters.

---

## 7. How Are the Embedding Values Learned?

The lecture describes the process as follows:

1. Decide the vocabulary size.
2. Choose the embedding dimension.
3. Create the embedding matrix with the required shape.
4. Initialize the weights, commonly with small random values.
5. Train the language model on text.
6. Use backpropagation and an optimizer to update trainable parameters, including the embedding weights.

At initialization, the vectors do not yet contain the useful learned relationships we want. Training adjusts them as part of the model's overall learning process.

The lecture emphasizes that the embedding matrix is trained **alongside** the rest of the language model. The model learns to predict the next token, and the training updates the parameters that help it do so.

### Key idea

Token embeddings in a GPT-style model are not necessarily imported from a separate pretrained word-vector dataset. The lecture's GPT-2 explanation describes embeddings being learned as part of GPT-2's own training process.

---

## 8. PyTorch Example: Creating an Embedding Layer

The lecture uses a toy vocabulary of six tokens:

`quick`, `fox`, `is`, `in`, `the`, `house`

For simplicity, it chooses an embedding dimension of 3.

```python
import torch

# Six possible token IDs; each token gets a 3-dimensional vector.
embedding_layer = torch.nn.Embedding(
    num_embeddings=6,
    embedding_dim=3
)

# Example token IDs
input_ids = torch.tensor([2, 3, 5, 1])

# Retrieve the embedding vector for each ID
embeddings = embedding_layer(input_ids)

print(embeddings)
print(embeddings.shape)
```

### What each part means

- `torch.nn.Embedding(6, 3)` creates an embedding table with **6 rows and 3 columns**.
- `num_embeddings=6` is the number of token IDs that can be looked up. It is not the number of tokens in the current sentence.
- `embedding_dim=3` means each token is represented by three numbers.
- `input_ids` contains four token IDs.
- `embedding_layer(input_ids)` looks up the row for each ID and returns the corresponding vector.
- `embeddings.shape` will be `torch.Size([4, 3])`: four token IDs, each represented by a three-dimensional vector.

The actual values will normally be different each time the layer is initialized, unless you control the random seed or load trained weights.

**Important:** the IDs must be valid indices. For a table with six rows, valid IDs are `0` through `5`.

---

## 9. The Embedding Matrix Is a Lookup Table

This is one of the lecture's most important practical points.

Suppose an embedding matrix has six rows:

| Token ID | Row selected | Returned vector |
|---:|---:|---|
| 0 | First row | Vector for ID 0 |
| 1 | Second row | Vector for ID 1 |
| 2 | Third row | Vector for ID 2 |
| 3 | Fourth row | Vector for ID 3 |
| 4 | Fifth row | Vector for ID 4 |
| 5 | Sixth row | Vector for ID 5 |

The token ID is used as a row index. The embedding layer returns that row.

For multiple IDs, the layer performs the lookup for each ID and returns multiple vectors in the same order.

### Why “lookup table”?

A dictionary lets you look up a value using a key. Similarly, an embedding layer uses a token ID to select the vector stored at the corresponding row.

**Remember:** the layer does not calculate a new semantic vector from scratch every time it receives an ID. It retrieves the current learned vector from its weight matrix. Training can update those stored values.

---

## 10. Embedding Matrix vs. Linear Layer

The lecture gives an additional explanation for readers familiar with neural networks.

An embedding lookup can produce the same result as:
1. Turning each token ID into a one-hot vector.
2. Multiplying that one-hot vector by a suitable weight matrix.

The one-hot vector has a single `1`, so the multiplication selects the corresponding row/weights.

However, a direct embedding lookup is more computationally efficient because it avoids creating and multiplying large one-hot vectors full of zeros. This matters when the vocabulary is large.

### Practical takeaway

Use an embedding layer such as `torch.nn.Embedding` for token-ID-to-vector lookup in ordinary language-model implementations. It is designed for this operation.

---

## 11. Vocabulary Size vs. Embedding Dimension

These two settings have different meanings.

| Setting | Meaning | Example |
|---|---|---|
| Vocabulary size | How many distinct token IDs the embedding table supports | 6 |
| Embedding dimension | How many numbers represent each token | 3 |
| Embedding matrix shape | Rows × columns | `(6, 3)` |
| Number of stored values | Vocabulary size × embedding dimension | `6 × 3 = 18` |

If the vocabulary has 6 token IDs and the embedding dimension is 3, there are 6 different vectors, and each vector has 3 values.

If the vocabulary size increases, the matrix gets more rows. If the embedding dimension increases, each row gets more columns.

---

## 12. What Token Embeddings Do Not Cover: Word Position

The lecture ends by introducing **positional embeddings** as the next topic.

Token embeddings represent learned information about tokens, but the position of a token in a sentence also matters. For example, changing the order of words can change a sentence's meaning.

The lecture's next step is to explain how positional information is represented and combined with token information.

**Keep these concepts separate:**
- **Token embedding:** vector representation associated with a token ID.
- **Positional embedding:** representation of where a token occurs in a sequence.

---

## 13. Common Misunderstandings

1. **“A token ID is an embedding.”**  
   No. A token ID is an integer identifier; an embedding is a vector of learned values.

2. **“Embedding dimension is vocabulary size.”**  
   No. Vocabulary size determines the number of rows; embedding dimension determines the number of columns.

3. **“The embedding values are meaningful when first created.”**  
   Not necessarily. They are commonly initialized randomly and become useful through training.

4. **“An embedding layer manually checks the meaning of a word.”**  
   No. It looks up a vector by token ID. The relationships are reflected in the learned parameter values.

5. **“Every token is a complete word.”**  
   No. GPT-style tokenizers can use subwords and other token units.

6. **“Nearby vectors always mean two words are synonyms.”**  
   Not necessarily. Embeddings can reflect different kinds of relationships, and results depend on the model and metric.

7. **“The model only trains its embedding matrix.”**  
   No. The embedding matrix is one part of a much larger language model whose other trainable parameters are also updated.

---

## 14. Quick Revision

- Token embeddings convert token IDs into vectors.
- Token IDs identify tokens but do not inherently encode semantic relationships.
- One-hot encoding distinguishes tokens but does not naturally encode learned similarity.
- Trained embeddings can place related tokens closer together in vector space.
- The lecture demonstrates vector arithmetic with `king + woman − man ≈ queen`.
- Embedding matrix shape = `(vocabulary size, embedding dimension)`.
- The transcript's GPT-2 example uses a vocabulary size of 50,257 and an embedding dimension of 768 for the smallest model discussed.
- Embedding weights are initialized and then optimized during model training.
- `torch.nn.Embedding` is a lookup table from token IDs to vectors.
- Embedding lookup is more efficient than multiplying large one-hot vectors by a weight matrix.
- Token embeddings do not by themselves encode the position of each token in a sequence; positional embeddings are a separate topic.

