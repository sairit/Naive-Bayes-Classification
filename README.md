# Bayesian Email Spam Classification System

A comprehensive implementation of a Naive Bayes classifier for email spam detection, featuring text preprocessing, probability calculations with Laplace smoothing, and performance evaluation across multiple cutoff thresholds.

## Project Overview

This project implements a probabilistic approach to email spam classification using Bayesian inference. The system processes training data to learn word distributions in spam versus legitimate emails, then applies these learned probabilities to classify new emails with configurable decision thresholds.

## Mathematical Foundation

### Naive Bayes Classification

The classifier applies Bayes' theorem with the "naive" assumption of conditional independence between words:

```
P(Spam|Words) ∝ P(Words|Spam) × P(Spam)
P(Ham|Words) ∝ P(Words|Ham) × P(Ham)
```

For a set of words W = {w₁, w₂, ..., wₙ}, the naive independence assumption gives:

```
P(Words|Class) = ∏ᵢ P(wᵢ|Class)
```

### Laplace Smoothing

To handle unseen words and prevent zero probabilities, the system implements Laplace smoothing:

```
P(word|class) = (count(word, class) + k) / (total_words_in_class + 2k)
```

Where k = 1 is the smoothing parameter, ensuring all probabilities remain non-zero.

### Classification Decision

The system classifies an email as spam when:

```
P(Spam|Words) > cutoff × P(Ham|Words)
```

This allows for adjustable sensitivity through the cutoff parameter.

## Technical Implementation

### Core Components

**TextProcessor Class** (`Train.py`)
- Handles training data preprocessing and probability calculation
- Implements text cleaning with punctuation removal and case normalization
- Applies stop word filtering to focus on meaningful terms
- Generates word probability distributions with Laplace smoothing

**BayesClassification Class** (`Test.py`)
- Loads pre-trained model data (word probabilities and class counts)
- Performs classification on test data
- Evaluates performance across multiple cutoff values
- Generates comprehensive metrics and visualizations

### Text Preprocessing Pipeline

1. **Case Normalization**: Convert all text to lowercase for consistency
2. **Punctuation Removal**: Strip special characters while preserving word boundaries
3. **Stop Word Filtering**: Remove common words that provide minimal classification value
4. **Whitespace Normalization**: Clean and tokenize the processed text

### Probability Calculation

The system maintains word counts for each class during training:

```python
word_counts[word] = [ham_count, spam_count]
```

Probabilities are calculated using Laplace smoothing to ensure robust performance:

```python
ham_prob = (ham_count + k) / (2 * k + total_ham)
spam_prob = (spam_count + k) / (2 * k + total_spam)
```

## Performance Evaluation

### Metrics Calculated

- **Accuracy**: (TP + TN) / (TP + TN + FP + FN)
- **Precision**: TP / (TP + FP) - Spam detection accuracy
- **Recall**: TP / (TP + FN) - Spam detection completeness
- **F1 Score**: 2 × (Precision × Recall) / (Precision + Recall)

### Classification Rates

- **True Positive Rate**: TP / Total_Spam
- **False Positive Rate**: FP / Total_Ham
- **True Negative Rate**: TN / Total_Ham
- **False Negative Rate**: FN / Total_Spam

### Cutoff Analysis

The system evaluates performance across cutoff values from 0.0 to 0.95 in 0.05 increments, allowing for:
- ROC curve analysis through FPR/TPR relationships
- Optimal threshold selection based on specific requirements
- Trade-off analysis between precision and recall

## Usage

### Training Phase
```bash
python Train.py
```
Processes training data and generates:
- `Dictionary.txt`: JSON file containing word probabilities
- `HSCount.txt`: Ham and spam email counts

### Testing Phase
```bash
python Test.py
```
Loads the trained model and provides:
- Batch evaluation across multiple cutoffs
- Interactive testing with custom cutoff values
- Performance visualizations and metrics

### File Structure
```
BayesClassification/
├── train/
│   ├── Dictionary.txt    # Word probability distributions
│   └── HSCount.txt       # Class count statistics
├── Train.py              # Training data processor
├── Test.py               # Classification and evaluation
└── StopWords.txt         # Common words to filter
```

## Key Features

### Robust Text Processing
- Handles various text encodings and special characters
- Configurable stop word filtering
- Consistent text normalization across training and testing

### Probabilistic Classification
- Implements proper Bayesian inference with independence assumptions
- Laplace smoothing prevents numerical instabilities
- Configurable decision thresholds for different use cases

### Comprehensive Evaluation
- Multiple performance metrics for thorough assessment
- Visual analysis through matplotlib plots
- Interactive testing capabilities for threshold optimization

### Scalable Design
- Modular architecture separating training and testing phases
- JSON-based model persistence for easy deployment
- Memory-efficient probability calculations

## Mathematical Considerations

The implementation addresses several important aspects of probabilistic classification:

1. **Numerical Stability**: Laplace smoothing prevents zero probabilities that would break the multiplicative probability calculation

2. **Feature Independence**: While the "naive" assumption is rarely true in practice, it significantly simplifies computation while often providing good results

3. **Threshold Optimization**: The configurable cutoff allows balancing between false positives (legitimate emails marked as spam) and false negatives (spam emails not detected)

4. **Class Imbalance**: The system handles datasets with unequal numbers of spam and ham emails through proper probability normalization

This implementation demonstrates my understanding of probabilistic machine learning principles, text processing techniques, and performance evaluation methodologies in the context of practical spam detection systems.
