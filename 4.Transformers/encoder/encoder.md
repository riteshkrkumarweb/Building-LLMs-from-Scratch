### Encoder

**Encoder understands the relationships and context between input tokens.**

It does **not** mean that the encoder simply changes words into numbers.
### Important

```text
Encoder ≠ Words → Numbers
```

The conversion from text to numerical form is mainly handled by:(not Encoder and Decoder)

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
