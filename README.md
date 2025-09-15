# 📝 TextSummarizer - Advanced Text Summarization Toolkit

## 📋 Project Overview
TextSummarizer is a comprehensive and production-ready text summarization toolkit that implements both extractive and abstractive summarization techniques. Built with clean architecture and modular design, it includes multiple extractive algorithms (TextRank, TF-IDF, Frequency-based), state-of-the-art abstractive models (BART, T5, Pegasus), and hybrid approaches for optimal summarization quality. The system supports various document types with extensive customization options and requires no external data downloads.

## 🎯 Objectives
- Generate concise and informative summaries from long-form text using multiple algorithms
- Implement both extractive and abstractive summarization following best practices
- Support various text formats with flexible preprocessing and tokenization
- Provide modular architecture with independent extractive and abstractive components
- Enable model evaluation with comprehensive metrics and performance analysis
- Offer configurable parameters for different summarization scenarios and requirements

## 📊 Text Processing Information
| Attribute | Details |
|-----------|---------|
| **Supported Formats** | Plain text, articles, documents, research papers |
| **Input Processing** | Automatic sentence/word tokenization, text cleaning |
| **Output Options** | Configurable summary length, sentence count control |
| **Language Support** | English (with built-in stopwords and language processing) |
| **Tokenization** | Regex-based tokenization with no external dependencies |

## 🔧 Technical Implementation

### 🔌 Summarization Architectures
- **Extractive Methods**: TextRank (PageRank-based), TF-IDF scoring, Frequency-based ranking
- **Abstractive Models**: BART, T5, Pegasus transformer models via Hugging Face
- **Hybrid Approach**: Combines extractive sentence selection with abstractive refinement
- **Text Processing**: Custom TextProcessor class with robust sentence/word tokenization
- **Similarity Metrics**: Cosine similarity for sentence ranking and selection

### 🧹 Text Preprocessing
**Preprocessing Pipeline:**
- Automatic text cleaning and normalization
- Citation removal and whitespace normalization
- Sentence boundary detection with abbreviation handling
- Word tokenization with contraction support
- Stopword filtering with comprehensive English stopwords list
- No external data dependencies or downloads required

### ⚙️ Summarization Architecture
**Extractive Summarization Process:**
1. **TextRank Algorithm**: Graph-based sentence ranking using PageRank
   - Build sentence similarity matrix using cosine similarity
   - Apply PageRank algorithm for sentence importance scoring
   - Select top-ranked sentences maintaining original order

2. **TF-IDF Method**: Statistical importance-based sentence selection
   - Calculate Term Frequency-Inverse Document Frequency scores
   - Rank sentences by cumulative TF-IDF values
   - Extract highest-scoring sentences for summary

3. **Frequency-based**: Simple word frequency scoring approach
   - Calculate normalized word frequencies excluding stopwords
   - Score sentences based on average word importance
   - Select sentences with highest average scores

**Abstractive Summarization:**
- Pre-trained transformer models (BART, T5, Pegasus)
- Sequence-to-sequence generation with attention mechanisms
- Configurable output length and generation parameters
- Graceful fallback to extractive methods if models unavailable

### 📈 Training Features
**Evaluation and Metrics:**
- Compression ratio calculation for summary efficiency
- Word overlap analysis between original and summary
- Automatic performance benchmarking across methods
- Comparative analysis of extractive vs abstractive approaches

**Model Management:**
- Optional transformer model loading and caching
- Memory-efficient processing for large documents
- Configurable parameters for different use cases
- Error handling and fallback mechanisms

## 📊 Visualizations
- **Performance Metrics**: Compression ratios and word overlap statistics
- **Method Comparison**: Side-by-side comparison of different summarization approaches
- **Summary Quality**: Analysis of summary coherence and information retention
- **Processing Statistics**: Token counts, sentence distributions, and processing time

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Required libraries: scikit-learn, networkx, numpy
- Optional: transformers, torch (for abstractive summarization)

### Installation
1. Clone the repository:
```bash
git clone https://github.com/zubair-csc/TextSummarizer.git
cd TextSummarizer
```

2. Install required libraries:
```bash
pip install scikit-learn networkx numpy
# Optional for abstractive summarization:
pip install transformers torch
```

### Basic Usage
No external data downloads required - everything works out of the box:
```python
python text_summarization.py
```

This will:
- Load the sample text and demonstrate all summarization methods
- Show extractive summaries using TextRank, TF-IDF, and frequency-based approaches
- Display abstractive summaries (if transformers available)
- Provide evaluation metrics and performance comparison

## 📈 Usage Examples

### Extractive Summarization
```python
from text_summarization import ExtractiveSummarizer

# Initialize summarizer
summarizer = ExtractiveSummarizer()

# TextRank summarization
summary = summarizer.textrank_summarize(text, num_sentences=3)

# TF-IDF summarization
summary = summarizer.tfidf_summarize(text, num_sentences=3)

# Frequency-based summarization
summary = summarizer.frequency_summarize(text, num_sentences=3)
```

### Abstractive Summarization
```python
from text_summarization import AbstractiveSummarizer

# Initialize abstractive summarizer
summarizer = AbstractiveSummarizer()

# BART summarization
summary = summarizer.bart_summarize(text, max_length=150)

# T5 summarization
summary = summarizer.t5_summarize(text, max_length=150)

# Pegasus summarization
summary = summarizer.pegasus_summarize(text, max_length=150)
```

### Hybrid Approach
```python
from text_summarization import HybridSummarizer

# Combine extractive and abstractive methods
summarizer = HybridSummarizer()
summary = summarizer.hybrid_summarize(text, extract_ratio=0.4, final_length=100)
```

### Custom Text Processing
```python
from text_summarization import TextProcessor

# Custom text processing
processor = TextProcessor()
sentences = processor.sent_tokenize(text)
words = processor.word_tokenize(text)
clean_text = processor.clean_text(text)
```

### Evaluation and Metrics
```python
from text_summarization import evaluate_summary

# Evaluate summary quality
metrics = evaluate_summary(original_text, summary)
print(f"Compression ratio: {metrics['compression_ratio']:.2f}")
print(f"Word overlap: {metrics['word_overlap']:.2f}")
```

## 🔮 Future Enhancements
- Multi-language support with language detection and processing
- Advanced evaluation metrics (ROUGE, BLEU, BERTScore)
- Web interface for interactive text summarization
- API endpoint development for integration with other applications
- Support for document formats (PDF, DOCX, HTML)
- Batch processing capabilities for multiple documents
- Custom model fine-tuning for domain-specific summarization
- Integration with popular NLP frameworks and pipelines

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 🙌 Acknowledgments
- **Scikit-learn** for TF-IDF vectorization and machine learning utilities
- **NetworkX** for graph-based algorithms and PageRank implementation
- **Hugging Face Transformers** for state-of-the-art abstractive models
- **PyTorch** for deep learning framework support
- Open source community for continuous support and inspiration

## 📞 Contact
Zubair - [GitHub Profile](https://github.com/zubair-csc)

Project Link: [https://github.com/zubair-csc/018-TextSummarizer-Extractive-Abstractive-Hybrid](https://github.com/zubair-csc/018-TextSummarizer-Extractive-Abstractive-Hybrid)

⭐ Star this repository if you found it helpful!
