# 🧠 RNN Educational Package

A comprehensive educational package for learning **Recurrent Neural Networks (RNN)** from basics to advanced applications. This package contains 13 carefully designed Python programs that progressively teach RNN concepts with practical implementations.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Program Descriptions](#program-descriptions)
  - [1. Basic RNN Cell](#1-basic-rnn-cell)
  - [2. Text Generation](#2-text-generation)
  - [3. Sentiment Analysis](#3-sentiment-analysis)
  - [4. Time Series Prediction](#4-time-series-prediction)
  - [5. LSTM Implementation](#5-lstm-implementation)
  - [6. GRU Implementation](#6-gru-implementation)
  - [7. Bidirectional RNN](#7-bidirectional-rnn)
  - [8. Sequence-to-Sequence](#8-sequence-to-sequence)
  - [9. Attention Mechanism](#9-attention-mechanism)
  - [10. Multi-layer RNN](#10-multi-layer-rnn)
  - [11. Character-level Language Model](#11-character-level-language-model)
  - [12. Named Entity Recognition](#12-named-entity-recognition)
  - [13. Stock Price Prediction](#13-stock-price-prediction)
- [Output Examples](#output-examples)
- [Learning Path](#learning-path)
- [Contributing](#contributing)

---

## 🎯 Overview

This educational package is designed for students, researchers, and practitioners who want to understand Recurrent Neural Networks deeply. Each program builds upon previous concepts, providing a structured learning experience.

**Key Features:**
- ✅ Progressive difficulty levels
- ✅ Well-commented code with explanations
- ✅ Visualization of results
- ✅ Real-world applications
- ✅ Both theoretical and practical insights

---

## 🛠️ Installation

### Prerequisites

```bash
Python 3.8+
pip (Python package manager)
```

### Required Libraries

```bash
pip install numpy pandas matplotlib tensorflow keras scikit-learn
```

### Optional Libraries

```bash
pip install seaborn plotly
```

---

## 📚 Program Descriptions

### 1. Basic RNN Cell
**File:** `01_basic_rnn_cell.py`

**Difficulty:** 🟢 Beginner

**Description:**
Implements a basic RNN cell from scratch using NumPy. This program demonstrates:
- Forward propagation in RNN
- Hidden state computation
- Weight initialization
- Single-step prediction

**Key Concepts:**
- RNN architecture fundamentals
- Hidden state mechanism
- Activation functions (tanh)

**Output:**
- Console output showing hidden state evolution
- Basic computation flow

![Basic RNN Cell Output](outputs%20-%20graphs/01_basic_rnn_output.png)

---

### 2. Text Generation
**File:** `02_text_generation.py`

**Difficulty:** 🟡 Intermediate

**Description:**
Builds a character-level text generation model using RNN. The model learns patterns from input text and generates new text sequences.

**Key Concepts:**
- Character encoding/decoding
- Sequence prediction
- Temperature sampling
- Text preprocessing

**Features:**
- Train on custom text corpus
- Generate creative text outputs
- Adjustable generation length
- Temperature-based creativity control

**Output:**
- Generated text samples
- Training loss curves

![Text Generation Example](outputs%20-%20graphs/02_text_generation.png)

---

### 3. Sentiment Analysis
**File:** `03_sentiment_analysis.py`

**Difficulty:** 🟡 Intermediate

**Description:**
Implements sentiment classification using RNN on movie reviews or social media data. Classifies text as positive, negative, or neutral.

**Key Concepts:**
- Word embeddings
- Sequence classification
- Binary/Multi-class classification
- Evaluation metrics (accuracy, F1-score)

**Features:**
- Pre-trained word embeddings support
- Confusion matrix visualization
- ROC curve analysis
- Real-time prediction

**Output:**
- Accuracy metrics
- Confusion matrix
- Sample predictions

![Sentiment Analysis Results](outputs%20-%20graphs/03_sentiment_analysis.png)

---

### 4. Time Series Prediction
**File:** `04_time_series_prediction.py`

**Difficulty:** 🟡 Intermediate

**Description:**
Demonstrates RNN's capability to predict future values in time series data. Uses sliding window approach for sequence prediction.

**Key Concepts:**
- Temporal dependencies
- Sliding window technique
- Multi-step forecasting
- Data normalization

**Features:**
- Multiple time series support
- Configurable lookback window
- Visualization of predictions vs actual
- Error metrics (MSE, MAE)

**Output:**
- Prediction plots
- Error metrics
- Forecast visualization

![Time Series Prediction](outputs%20-%20graphs/04_time_series.png)

---

### 5. LSTM Implementation
**File:** `05_lstm_implementation.py`

**Difficulty:** 🟠 Advanced

**Description:**
Implements Long Short-Term Memory (LSTM) networks to solve the vanishing gradient problem. Compares performance with basic RNN.

**Key Concepts:**
- LSTM architecture (forget, input, output gates)
- Cell state mechanism
- Gradient flow improvement
- Memory retention

**Features:**
- Custom LSTM cell implementation
- Gate activation visualization
- Comparison with vanilla RNN
- Long sequence handling

**Output:**
- Gate activations over time
- Performance comparison charts
- Memory state visualization

![LSTM Architecture](outputs%20-%20graphs/05_lstm.png)

---

### 6. GRU Implementation
**File:** `06_gru_implementation.py`

**Difficulty:** 🟠 Advanced

**Description:**
Implements Gated Recurrent Unit (GRU), a simplified version of LSTM with fewer parameters but comparable performance.

**Key Concepts:**
- GRU architecture (update and reset gates)
- Simplified gating mechanism
- Parameter efficiency
- Performance vs complexity trade-off

**Features:**
- GRU vs LSTM comparison
- Training time analysis
- Parameter count comparison
- Accuracy benchmarking

**Output:**
- Performance metrics
- Training curves
- Gate behavior visualization

![GRU vs LSTM](outputs%20-%20graphs/06_gru.png)

---

### 7. Bidirectional RNN
**File:** `07_bidirectional_rnn.py`

**Difficulty:** 🟠 Advanced

**Description:**
Implements bidirectional RNN that processes sequences in both forward and backward directions, capturing context from both sides.

**Key Concepts:**
- Forward and backward passes
- Context aggregation
- Improved sequence understanding
- Applications in NLP

**Features:**
- Bidirectional architecture
- Context-aware predictions
- Comparison with unidirectional RNN
- POS tagging example

**Output:**
- Bidirectional flow visualization
- Accuracy improvements
- Context utilization analysis

![Bidirectional RNN](outputs%20-%20graphs/07_bidirectional.png)

---

### 8. Sequence-to-Sequence
**File:** `08_seq2seq.py`

**Difficulty:** 🔴 Expert

**Description:**
Implements encoder-decoder architecture for sequence-to-sequence tasks like machine translation or text summarization.

**Key Concepts:**
- Encoder-decoder paradigm
- Context vector
- Variable length input/output
- Teacher forcing

**Features:**
- Translation example (English to French)
- Attention-free baseline
- BLEU score evaluation
- Beam search decoding

**Output:**
- Translation examples
- Training progress
- BLEU scores

![Seq2Seq Architecture](outputs%20-%20graphs/08_seq2seq.png)

---

### 9. Attention Mechanism
**File:** `09_attention_mechanism.py`

**Difficulty:** 🔴 Expert

**Description:**
Adds attention mechanism to seq2seq model, allowing the decoder to focus on relevant parts of input sequence.

**Key Concepts:**
- Attention weights computation
- Query-key-value paradigm
- Alignment visualization
- Improved translation quality

**Features:**
- Attention weight visualization
- Interpretability through attention maps
- Performance improvement over vanilla seq2seq
- Multiple attention types (additive, multiplicative)

**Output:**
- Attention heatmaps
- Translation quality metrics
- Alignment visualization

![Attention Mechanism](outputs%20-%20graphs/09_attention.png)

---

### 10. Multi-layer RNN
**File:** `10_multilayer_rnn.py`

**Difficulty:** 🟠 Advanced

**Description:**
Stacks multiple RNN layers to create deep recurrent networks with increased representational capacity.

**Key Concepts:**
- Layer stacking
- Hierarchical feature learning
- Depth vs width trade-off
- Dropout between layers

**Features:**
- Configurable number of layers
- Layer-wise output analysis
- Optimal depth finding
- Overfitting prevention

**Output:**
- Performance vs depth curves
- Layer activation visualization
- Training dynamics

![Multi-layer RNN](outputs%20-%20graphs/10_multilayer.png)

---

### 11. Character-level Language Model
**File:** `11_char_language_model.py`

**Difficulty:** 🟡 Intermediate

**Description:**
Builds a character-level language model that learns the probability distribution of characters in text and generates realistic text.

**Key Concepts:**
- Character embeddings
- Language modeling
- Perplexity metric
- Sampling strategies

**Features:**
- Shakespeare text generation
- Code generation examples
- Adjustable creativity (temperature)
- Beam search sampling

**Output:**
- Generated text samples
- Perplexity curves
- Character distribution analysis

![Character Language Model](outputs%20-%20graphs/11_char_lm.png)

---

### 12. Named Entity Recognition
**File:** `12_ner.py`

**Difficulty:** 🟠 Advanced

**Description:**
Implements Named Entity Recognition (NER) system to identify and classify named entities (persons, organizations, locations) in text.

**Key Concepts:**
- Sequence labeling
- BIO tagging scheme
- Entity extraction
- Conditional Random Fields integration

**Features:**
- Multi-entity type support
- Entity-level F1 score
- Nested entity handling
- Real-world text processing

**Output:**
- Entity extraction examples
- Precision/Recall/F1 metrics
- Confusion matrix for entity types

![NER Results](outputs%20-%20graphs/12_ner.png)

---

### 13. Stock Price Prediction
**File:** `13_stock_prediction.py`

**Difficulty:** 🟡 Intermediate

**Description:**
Applies RNN to predict stock prices using historical data. Demonstrates practical application in financial forecasting.

**Key Concepts:**
- Financial time series
- Technical indicators integration
- Multi-feature prediction
- Risk assessment

**Features:**
- Multiple stock symbols support
- Technical indicator incorporation (RSI, MACD)
- Confidence intervals
- Backtesting framework

**Output:**
- Price prediction charts
- Error metrics (RMSE, MAPE)
- Profit/Loss simulation

![Stock Prediction](outputs%20-%20graphs/13_stock_prediction.png)

---

## 📊 Output Examples

All programs generate visual outputs and graphs saved in the `outputs - graphs/` directory:

```
outputs - graphs/
├── 01_basic_rnn_output.png
├── 02_text_generation.png
├── 03_sentiment_analysis.png
├── 04_time_series.png
├── 05_lstm.png
├── 06_gru.png
├── 07_bidirectional.png
├── 08_seq2seq.png
├── 09_attention.png
├── 10_multilayer.png
├── 11_char_lm.png
├── 12_ner.png
└── 13_stock_prediction.png
```

---

## 🎓 Learning Path

### Recommended Order for Beginners:

1. **Start Here:** Basic RNN Cell (01) → Understanding fundamentals
2. **Text Basics:** Text Generation (02) → Character-level processing
3. **Classification:** Sentiment Analysis (03) → Sequence classification
4. **Time Series:** Time Series Prediction (04) → Temporal patterns
5. **Advanced Cells:** LSTM (05) → Solving vanishing gradients
6. **Efficiency:** GRU (06) → Simplified gating
7. **Context:** Bidirectional RNN (07) → Both-way context
8. **Deep Networks:** Multi-layer RNN (10) → Stacking layers
9. **Language Modeling:** Char Language Model (11) → Text generation
10. **Applications:** NER (12), Stock Prediction (13)
11. **Advanced Topics:** Seq2Seq (08), Attention (09)

### For Experienced Practitioners:

Jump to specific topics based on your interest:
- **NLP Focus:** 02, 03, 08, 09, 11, 12
- **Time Series Focus:** 04, 13
- **Architecture Deep Dive:** 05, 06, 07, 10

---

## 💡 Tips for Learning

1. **Read the Code:** Each file is heavily commented. Read comments carefully.
2. **Modify Parameters:** Experiment with hyperparameters to see their effects.
3. **Visualize:** Always check the generated graphs to understand model behavior.
4. **Compare:** Run multiple programs to compare different approaches.
5. **Start Simple:** Begin with basic programs even if you have experience.

---

## 🤝 Contributing

Contributions are welcome! If you'd like to add new programs or improve existing ones:

1. Fork the repository
2. Create a feature branch
3. Add your program following the existing naming convention
4. Update this README with your program description
5. Submit a pull request

---

## 📝 License

This educational package is provided for learning purposes. Feel free to use, modify, and distribute for educational purposes.

---

## 📧 Contact

For questions, suggestions, or feedback:
- Open an issue in the repository
- Email: [afyonluoglu@gmail.com]

---

## 🙏 Acknowledgments

This package was created to make RNN concepts accessible to everyone. Special thanks to:
- The deep learning community
- TensorFlow and Keras teams
- All contributors and users

---

**Happy Learning! 🚀**

*"The best way to learn is by doing. Start coding, experiment, and enjoy the journey through the fascinating world of Recurrent Neural Networks!"*
