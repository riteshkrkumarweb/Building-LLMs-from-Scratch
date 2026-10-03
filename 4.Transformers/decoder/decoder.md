
### Decoder

**Decoder uses the available context to generate tokens step by step.**

It does **not** mean that the decoder simply changes numbers back into text.

### Important

```text
Decoder ≠ Numbers → Words
```

The conversion from text to numerical form is mainly handled by:(not encoder and decoder)

```text
Text
 ↓
Tokenizer
 ↓
Token IDs
 ↓
Embedding
 ↓
Vectors
```

So, in simple terms:

**Encoder → Understands the context and relationships between tokens.**
**Decoder → Uses that context to generate the next tokens.**
**Tranformer have both Encoder and Decoder but GPTs has only Decoder with the encoder features**