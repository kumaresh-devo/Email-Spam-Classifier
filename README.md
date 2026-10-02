# Email Spam Classifier

A machine learning-based command-line project that automatically classifies email and SMS messages into **SPAM** or **HAM** (legitimate/non-spam) using natural language processing (NLP) and a Multinomial Naive Bayes classifier.

---

## Project Objective

To build an efficient, terminal-based Email Spam Classifier that distinguishes between spam and legitimate messages by analyzing raw email content using text preprocessing, TF-IDF feature extraction, and supervised machine learning.

The entire interaction happens directly inside your **terminal/command prompt** with zero web framework dependencies (no HTML, CSS, JavaScript, or Flask).

---

## Features

- **Spam vs. Ham Detection**: Accurately classifies emails as either malicious/unwanted spam or legitimate correspondence.
- **NLP Text Preprocessing**: Strips URLs, HTML artifacts, special characters, extra whitespace, and performs case-folding.
- **TF-IDF Feature Extraction**: Utilizes unigrams and bigrams (`ngram_range=(1, 2)`) with sublinear term frequency to identify spam patterns like *"free prize"*, *"urgent call"*, or *"claim now"*.
- **Multinomial Naive Bayes Model**: Fast, lightweight, and effective probabilistic classifier for text categorization.
- **Dynamic Metrics Evaluation**: Computes real-time **Accuracy** and **Precision** on a held-out 20% test dataset (never hard-coded).
- **Dynamic Prediction Confidence**: Outputs real-time probability confidence calculated directly via `predict_proba()`.
- **Interactive Multi-Email Testing**: Interactive terminal loop allows testing multiple messages sequentially with clean exit on `n`.
- **Self-Healing / Auto-Training**: If model artifacts are missing, `predict.py` automatically triggers training so it runs immediately out of the box.

---

## Technologies Used

- **Language**: Python 3.8+
- **Data Manipulation**: `pandas`, `numpy`
- **Machine Learning & NLP**: `scikit-learn` (`MultinomialNB`, `TfidfVectorizer`, `train_test_split`, `metrics`)
- **Model Serialization**: `joblib`

---

## Dataset

This project utilizes the authentic **UCI SMS Spam Collection Dataset**:
- **Total Records**: 5,572 labeled messages
- **Legitimate (Ham)**: 4,825 messages (~86.6%)
- **Spam**: 747 messages (~13.4%)
- **Location**: `dataset/spam.csv`

### Dataset Setup Instructions:
The repository already includes the verified dataset in `dataset/spam.csv`. If you ever need to restore it or place your own custom dataset, ensure the file is named `spam.csv` inside the `dataset/` directory and contains two primary columns:
1. `label` - with values `ham` or `spam`.
2. `message` - the raw text of the email/message.

*Note: If `dataset/spam.csv` is ever deleted or missing, running `python train.py` will automatically download the authentic dataset from the UCI archive mirror.*

---

## How the ML Model Works

1. **Text Cleaning**:
   - Converts all incoming text to lowercase.
   - Cleans web links (`http://...`, `www....`), HTML tags, and non-alphanumeric noise while preserving key tokens.
   - Trims redundant whitespaces.

2. **Feature Extraction (TF-IDF)**:
   - Evaluates Term Frequency-Inverse Document Frequency (TF-IDF).
   - Generates n-grams (single words and two-word combinations) to capture contextual phrasing.
   - Applies sublinear scaling to balance the weight of frequent spam keywords.

3. **Classification (Multinomial Naive Bayes)**:
   - Uses Bayes' Theorem with Laplace smoothing ($\alpha = 0.2$):
     $$P(\text{Spam} \mid \text{Words}) \propto P(\text{Spam}) \prod_{i} P(\text{word}_i \mid \text{Spam})$$
   - Computes probability distribution over both classes:
     $$\text{Confidence} = \max\bigl(P(\text{Spam}), P(\text{Ham})\bigr) \times 100\%$$

4. **Evaluation**:
   - The dataset is split into an 80% training set and a 20% stratified test set.
   - Model Accuracy and Spam Precision are evaluated on unseen test data.

---

## Project Structure

```
Email-Spam-Classifier/
│
├── dataset/
│   └── spam.csv              # Authentic labeled spam/ham dataset (5,572 records)
│
├── models/                   # Generated on model training
│   ├── spam_model.pkl        # Serialized MultinomialNB model
│   ├── tfidf_vectorizer.pkl  # Fitted TfidfVectorizer
│   └── metrics.json          # Actual test accuracy and precision
│
├── train.py                  # Preprocessing, training, evaluation, and serialization
├── predict.py                # Real-time interactive terminal classification CLI
├── requirements.txt          # Python package dependencies
└── README.md                 # Project documentation
```

---

## Installation

1. Open your terminal / command prompt.
2. Navigate to the project directory:
   ```bash
   cd Email-Spam-Classifier
   ```
3. Install dependencies using pip:
   ```bash
   pip install -r requirements.txt
   ```

---

## How to Run the Project

### Step 1: Train the Model (Optional / Initial Setup)
To inspect the training pipeline, split data, and calculate evaluation metrics:
```bash
python train.py
```

### Step 2: Run Real-Time Terminal Classifier
To classify messages interactively:
```bash
python predict.py
```
*(If you run `python predict.py` directly without training first, it will automatically train and save the model before prompting you).*

---

## Example Terminal Output

```text
========================================
       EMAIL SPAM CLASSIFIER
========================================

Model Accuracy : 97.94%
Precision      : 97.01%

Enter email:
> Congratulations! You won a free prize. Click now to claim your reward!

Prediction : SPAM
Confidence : 99.68%

========================================

Enter another email? (y/n): y

Enter email:
> Hi, please send me the project report tomorrow.

Prediction : HAM
Confidence : 99.26%

========================================

Enter another email? (y/n): n

Thank you for using Email Spam Classifier. Exiting cleanly.
```
