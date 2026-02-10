# IMDB Sentiment Analysis with Transformers

A deep learning project that uses the Transformer architecture to classify movie reviews from the IMDB dataset as positive or negative.

## Overview

This project demonstrates how to:
- Load and preprocess the IMDB movie reviews dataset
- Fine-tune a pre-trained transformer model (DistilBERT) for sentiment analysis
- Evaluate the model's performance
- Make predictions on new reviews

## Architecture

### Transformer Architecture
The diagram shows the classic Transformer architecture with:

**Encoder (Left Side):**
- **Input Embedding**: Converts text tokens to dense vectors
- **Positional Encoding**: Adds position information to embeddings
- **Multi-Head Attention**: Allows the model to focus on different parts of the input
- **Feed Forward**: Applies non-linear transformations
- **Add & Norm**: Residual connections with layer normalization
- Repeated N times for deep feature extraction

**Decoder (Right Side):**
- **Output Embedding**: Converts output tokens to vectors
- **Positional Encoding**: Adds position information
- **Masked Multi-Head Attention**: Prevents looking at future tokens
- **Multi-Head Attention**: Cross-attends to encoder outputs
- **Feed Forward**: Non-linear transformations
- **Add & Norm**: Residual connections with layer normalization
- Repeated N times

**Output Layer:**
- **Linear**: Projects to vocabulary size
- **Softmax**: Converts to probability distribution

### Our Implementation
We use **DistilBERT**, a distilled version of BERT (Bidirectional Encoder Representations from Transformers):
- Only uses the encoder part of the transformer
- 40% smaller and 60% faster than BERT
- Retains 97% of BERT's language understanding
- Pre-trained on millions of documents
- Fine-tuned on IMDB for sentiment classification

## Dataset

**IMDB Movie Reviews Dataset**
- **Training Set**: 25,000 labeled reviews
- **Test Set**: 25,000 labeled reviews
- **Classes**: Binary (Positive/Negative)
- **Source**: Large Movie Review Dataset v1.0

## How It Works

### 1. **Tokenization**
```python
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
```
- Converts text to token IDs
- Adds special tokens ([CLS], [SEP])
- Truncates to max length (512 tokens)
- Applies padding for batch processing

### 2. **Model Architecture**
```python
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2  # Binary classification
)
```

**DistilBERT Layers:**
- 6 transformer encoder layers
- 12 attention heads per layer
- 768 hidden dimensions
- ~66 million parameters
- Classification head on top ([CLS] token -> 2 classes)

### 3. **Training Process**

**Key Components:**
- **Optimizer**: AdamW (Adam with weight decay)
- **Learning Rate**: 2e-5 with warmup
- **Batch Size**: 8 samples per batch
- **Epochs**: 3 full passes through data
- **Loss Function**: Cross-entropy loss

**Training Flow:**
```
Input Review -> Tokenization -> DistilBERT Encoder -> [CLS] Token ->
Linear Layer -> Softmax -> Probability Distribution -> Loss Calculation ->
Backpropagation -> Weight Update
```

### 4. **Attention Mechanism**

The multi-head attention allows the model to:
- Focus on relevant words (e.g., "amazing", "terrible")
- Understand context ("not good" vs "very good")
- Capture long-range dependencies
- Learn different aspects simultaneously

**Attention Formula:**
```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
```
- Q (Query): What am I looking for?
- K (Key): What do I contain?
- V (Value): What do I actually represent?

### 5. **Prediction**

For a new review:
1. Tokenize the input text
2. Pass through DistilBERT encoder
3. Extract [CLS] token representation
4. Apply classification head
5. Softmax to get probabilities
6. Argmax to get final prediction

## Project Structure

```
transformers/
   imdb_sentiment_analysis.py    # Main training script
   requirements.txt              # Python dependencies
   README.md                     # This file
   imdb_sentiment_model/         # Training checkpoints (created during training)
   imdb_sentiment_model_final/   # Final saved model (created after training)
```

## Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

Required packages:
- `torch`: PyTorch deep learning framework
- `transformers`: Hugging Face transformers library
- `datasets`: Hugging Face datasets library
- `scikit-learn`: For evaluation metrics
- `numpy`: Numerical computing
- `accelerate`: For distributed training

## Usage

### Train the Model

**Full training (1-3 hours):**
```bash
python imdb_sentiment_analysis.py
```

**Quick test with subset (5-10 minutes):**
Edit `imdb_sentiment_analysis.py` and uncomment lines 40-41:
```python
dataset['train'] = dataset['train'].select(range(1000))
dataset['test'] = dataset['test'].select(range(500))
```

### Training Output

The script will:
1. Download the IMDB dataset
2. Load DistilBERT model and tokenizer
3. Tokenize all reviews
4. Train for 3 epochs
5. Evaluate after each epoch
6. Save the best model
7. Test with sample predictions

### Expected Results

With full training, you should achieve:
- **Accuracy**: ~92-94%
- **Precision**: ~92-94%
- **Recall**: ~92-94%
- **F1 Score**: ~92-94%

## Making Predictions

After training, the script automatically tests on sample reviews:

```python
test_reviews = [
    "This movie was absolutely fantastic! Best film I've seen this year.",
    "Terrible movie, waste of time. I want my money back.",
    "The acting was decent but the plot was confusing.",
]
```

**Example Output:**
```
Review: "This movie was absolutely fantastic! Best film I've seen this year."
Prediction: Positive
Confidence: Negative=2.34%, Positive=97.66%

Review: "Terrible movie, waste of time. I want my money back."
Prediction: Negative
Confidence: Negative=98.45%, Positive=1.55%
```

## Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning Rate | 2e-5 | Step size for weight updates |
| Batch Size | 8 | Samples processed together |
| Epochs | 3 | Full passes through dataset |
| Max Length | 512 | Maximum token sequence length |
| Warmup Steps | 500 | Gradual learning rate increase |
| Weight Decay | 0.01 | L2 regularization |

## Model Performance

### What the Model Learns

The transformer learns to:
- Identify sentiment-bearing words ("excellent", "awful")
- Understand negation ("not bad" -> positive)
- Recognize intensifiers ("very good", "extremely poor")
- Capture context and nuance
- Handle sarcasm to some degree

### Attention Visualization Example

For the review: **"This movie was not good"**

The attention mechanism learns to:
- Connect "not" with "good" (negation)
- Weight "not good" together as negative sentiment
- Downweight neutral words like "This", "movie", "was"

## Technical Details

### Why DistilBERT?

1. **Efficiency**: 40% smaller, 60% faster than BERT
2. **Performance**: Retains 97% of BERT's capabilities
3. **Distillation**: Trained to mimic BERT's behavior
4. **Bidirectional**: Understands context from both directions
5. **Pre-trained**: Already knows English language patterns

### Transfer Learning

The model uses transfer learning:
1. **Pre-training**: DistilBERT learned on massive text corpora
2. **Fine-tuning**: We adapt it to IMDB sentiment analysis
3. **Benefit**: Requires less data and training time

### Computational Requirements

- **GPU**: Recommended (CUDA-enabled NVIDIA GPU)
- **CPU**: Works but much slower
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: ~1GB for model and dataset

## Customization

### Use Different Models

Replace `"distilbert-base-uncased"` with:
- `"bert-base-uncased"`: Original BERT (larger, more accurate)
- `"roberta-base"`: RoBERTa (optimized BERT variant)
- `"albert-base-v2"`: ALBERT (parameter efficient)
- `"xlnet-base-cased"`: XLNet (autoregressive)

### Adjust Training

Modify `TrainingArguments`:
```python
training_args = TrainingArguments(
    num_train_epochs=5,           # More epochs
    learning_rate=3e-5,            # Different learning rate
    per_device_train_batch_size=16 # Larger batch size
)
```

## Evaluation Metrics

- **Accuracy**: Overall correctness
- **Precision**: Of predicted positives, how many are correct?
- **Recall**: Of actual positives, how many did we find?
- **F1 Score**: Harmonic mean of precision and recall

## Troubleshooting

### Out of Memory
- Reduce batch size: `per_device_train_batch_size=4`
- Reduce max length: `max_length=256`
- Use smaller model: Try `"distilbert-base-uncased"`

### Slow Training
- Enable GPU: Check CUDA installation
- Use smaller dataset: Uncomment subset selection
- Reduce epochs: `num_train_epochs=1`

### Poor Performance
- Train longer: Increase epochs
- Adjust learning rate: Try 1e-5 or 5e-5
- Use full dataset: Don't use subset

## References

1. **Attention Is All You Need** (Vaswani et al., 2017)
   - Original Transformer paper

2. **BERT: Pre-training of Deep Bidirectional Transformers** (Devlin et al., 2018)
   - BERT architecture and pre-training

3. **DistilBERT** (Sanh et al., 2019)
   - Distilled version of BERT

4. **Hugging Face Transformers**
   - https://huggingface.co/docs/transformers

## License

MIT

## Contributing

Contributions are welcome. Please open an issue to discuss changes before submitting a pull request.
