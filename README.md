# Fake-News-Detection-System
📌 Overview

This project implements an AI-powered Fake News Detection System using Natural Language Processing (NLP) and Machine Learning.
The system classifies news articles as Fake or Real by learning linguistic patterns from real-world datasets.

The project follows a research-oriented, end-to-end pipeline including data preprocessing, feature extraction, multi-model training, and detailed error analysis.

🎯 Objectives

Detect fake news using textual content

Apply NLP preprocessing techniques

Compare multiple machine learning models

Analyze errors using confusion matrices

Build an end-to-end prediction pipeline

📊 Dataset

Source: Kaggle – Fake and Real News Dataset

Files:

Fake.csv

True.csv

⚠️ Note:
Due to large file size, the dataset is not included in this repository.

🔗 Download the dataset from Kaggle:
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

After downloading, place the files inside the following directory:

data/
├── Fake.csv
└── True.csv

🧠 Methodology
1️⃣ Text Preprocessing

Lowercasing

Removal of punctuation and numbers

Stopword removal

Lemmatization

Removal of very short tokens

2️⃣ Feature Extraction

Technique: TF-IDF (Term Frequency–Inverse Document Frequency)

Max Features: 5000

Converts text data into numerical vectors for ML models

3️⃣ Machine Learning Models

The following models were trained and evaluated:

Model	Description
Logistic Regression	Strong baseline classifier
Naive Bayes	Fast and efficient for text data
Support Vector Machine (SVM)	High accuracy for sparse text features
4️⃣ Evaluation Metrics

Accuracy

Precision

Recall

F1-score

Confusion Matrix (for error analysis)

Special attention is given to false negatives, as misclassifying fake news as real is more harmful.

🔍 Error Analysis

Confusion matrices were generated for all models to analyze misclassification patterns.
Among the tested models, SVM showed the best performance with fewer critical errors.

🚀 Features

End-to-end NLP + ML pipeline

Multi-model comparison

Confusion matrix visualization

Custom prediction function for user input

Research-style evaluation

🧪 Custom Prediction

You can test the model with your own news text:

predict_news("Paste any news article text here")


Output:

FAKE NEWS ❌

REAL NEWS ✅

🛠️ Tech Stack

Language: Python

Libraries:

pandas

numpy

scikit-learn

nltk

matplotlib

Tools: VS Code, Jupyter Notebook

📁 Project Structure
Fake_News_Project/
├── FakeNewsDetection.ipynb
├── data/
│   ├── Fake.csv   (not uploaded)
│   └── True.csv   (not uploaded)
├── README.md
├── .gitignore

⚖️ Ethical Considerations

Dataset bias may affect predictions

Automated systems should assist, not replace, human judgment

Risk of false positives and false negatives

Importance of responsible AI usage

📈 Future Enhancements

Transformer-based models (BERT)

Source credibility analysis

Multilingual fake news detection

Web-based deployment using APIs

🧾 Resume / CV Description

Developed an AI-powered fake news detection system using NLP and machine learning. Implemented text preprocessing, TF-IDF feature extraction, and evaluated multiple classifiers including Logistic Regression, Naive Bayes, and SVM. Performed error analysis using confusion matrices and built a custom prediction pipeline for unseen news articles.

👨‍🎓 Author

Aditya Bhardwaj
Computer Science & Engineering
Cybersecurity & AI/ML Enthusiast
