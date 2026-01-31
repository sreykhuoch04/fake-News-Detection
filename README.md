# 📰 Fake News Detection Using Pretrained Transformers (RoBERTa)
Transformer-Based Fake News Detection with RoBERTa


*Flowchart of the Fake News Detection System*

## Overview
This project implements an automated fake news detection system using a fine-tuned RoBERTa transformer model. The system classifies news articles as fake or real based on textual content, providing confidence scores and highlighted keywords for user interpretation. Leveraging deep contextual language representations and self-attention mechanisms, the system achieves high accuracy and robust performance in text classification tasks.

Key Features:
- Binary Classification: Detects fake or real news articles.
- Confidence Scores: Provides the probability of prediction.
- Keyword Highlighting: Shows influential words contributing to prediction.
- User History Tracking: Stores inputs and results in CSV format.
- Transformer-Based Approach: Uses RoBERTa for understanding complex language patterns.
- Fine-Tuned Model: Pretrained RoBERTa is adapted to domain-specific fake news data.

---

## Dataset
- English news articles labeled as fake or real.
- Each instance includes title and content, combined to improve prediction accuracy.
- Dataset is balanced to prevent bias toward any class.
- Preprocessing includes:
  - Removal of URLs, HTML tags, and special characters
  - Normalization of whitespace and line breaks
  - Tokenization using RoBERTa tokenizer
  - Padding or truncation to fixed length sequences

---

## Methodology

The workflow of the fake news detection system includes the following steps:

### 1. Data Preprocessing
- Combine title & content (optional for stronger signal)
- Remove URLs, HTML tags, special characters
- Normalize whitespace
- Tokenization using RoBERTa tokenizer
- Padding / truncation to fixed sequence length

### 2. Model Fine-Tuning
- Load pretrained RoBERTa model (`roberta-base`)
- Add a sequence classification head for binary classification
- Fine-tune on labeled dataset with:
  - Weighted cross-entropy loss (for class imbalance)
  - AdamW optimizer (stabilizes training)

### 3. Model Evaluation
- Metrics used:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC-AUC
- Evaluated on a held-out test set using stratified sampling

### 4. Results Display & Storage
- Show prediction result to the user
- Display confidence score
- Highlight influential keywords
- Save user input and results to CSV history

---

## Model Performance

| Metric        | Score |
|---------------|-------|
| Accuracy      | 100%  |
| Precision     | 1.00  |
| Recall        | 1.00  |
| F1-score      | 1.00  |
| ROC-AUC Score | 0.9988|

The model demonstrates excellent class separation and is highly reliable for real-world fake news detection.

---

## Challenges
During development, several challenges were encountered:
- Fine-tuning RoBERTa required careful hyperparameter selection.
- Training instability in early epochs.
- Long training times due to transformer complexity.
- High CPU/GPU requirements for large datasets.
- Risk of overfitting when using many epochs.
- Difficulty explaining predictions, even with high accuracy.
- Some misclassifications occurred in articles from different countries, e.g., Cambodia or Thailand news, where local contexts differ.
- Most news articles have titles, which are critical for model input but require special preprocessing.

---

## Future Work & Recommendations
- Incorporate multimodal data such as images and videos.
- Apply explainable AI techniques to improve model transparency.
- Explore lightweight transformer models for real-time deployment.
- Extend system to multilingual fake news detection.
- Evaluate cross-domain generalization on unseen datasets.
- Implement title-focused preprocessing to improve predictions on short or headline-heavy articles.
- Optimize the system for deployment on low-resource devices with model compression or distillation.
🛠 Technologies Used

Python
PyTorch
Hugging Face Transformers
RoBERTa
Scikit-learn
Pandas, NumPy
👩‍🎓 Authors

SROY Liza
YORY Sreykhuoch
Department of Applied Mathematics and Statistics
Institute of Technology of Cambodia

👨‍🏫 Supervisor
Dr. Phauk Sokkhey
