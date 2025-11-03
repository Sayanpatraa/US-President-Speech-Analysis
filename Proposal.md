# Analyzing Presidential Rhetoric and Political Polarization

## Overview

Over the past several decades, the United States has experienced increasing polarization in political discourse. The words of presidents play a critical role in shaping public opinion, mobilizing movements, and influencing ideological divides.  
This project applies Natural Language Processing (NLP) techniques to analyze presidential speeches from the founding era to the present day. By examining tone, sentiment, and dominant topics, the project aims to uncover how rhetoric, emotion, and divisiveness evolve across presidents, parties, and historical contexts.

---

## Problem Statement

**Problem Chosen:**  
How has U.S. presidential rhetoric changed over time, and does the language used by different presidents or parties reflect growing political polarization?

**Why This Problem?**  
Language is a powerful mirror of social and political sentiment. By quantitatively studying presidential speeches, we can understand:
- How emotional tone (positive/negative) and divisiveness have evolved over time.
- Whether certain presidents or parties tend to use more unifying versus polarizing rhetoric.
- How topics and emotional emphasis shift during national crises, wars, and elections.

---

## Dataset Description

**Primary Dataset:**  
- **Source:** The Miller Center of Public Affairs Presidential Speech Archive (https://data.millercenter.org/)  
- **Content:** 1,000+ official presidential speeches ranging from George Washington to Joe Biden.  
- **Metadata:** Title, Date, Transcript, Context (Event), President Name.  

**Additional Dataset (for comparison or validation):**  
- **Source:** The American Presidency Project (https://www.presidency.ucsb.edu/)  
- **Purpose:** Supplement with additional speeches or metadata such as political party, approval ratings, or time in office.  

**Data Enhancements:**  
We will add columns for:
- President  
- Political Party  
- Term Start and Term End  
- Computed NLP features such as Sentiment Score, Emotion, and Topic Label.

---

## NLP Methods and Models

This project integrates classical, rule-based, and deep learning NLP methods.

| NLP Concept | Model / Technique | Purpose |
|--------------|------------------|----------|
| Rule-Based Models | Logistic Regression, Naive Bayes | Classify speeches by sentiment or party |
| Pretrained Models | VADER, FinBERT | Extract sentiment polarity and emotion |
| Topic Modeling | Latent Dirichlet Allocation (LDA) | Identify key recurring topics across eras |
| Neural Models | LSTM or BiLSTM (Recurrent Networks) | Capture contextual tone in long-form speeches |
| Interpretability | SHAP, LIME | Explain model predictions and interpret divisive language |
| Autoencoder (optional) | Denoising Autoencoder | Compress speech embeddings for clustering or visualization |

---

## Packages and Tools

| Category | Libraries / Tools | Purpose |
|-----------|------------------|----------|
| Data Processing | pandas, numpy, regex | Data cleaning, structuring, and feature creation |
| Text Preprocessing | nltk, spacy, re | Tokenization, stopword removal, lemmatization |
| Modeling and Training | scikit-learn, gensim, tensorflow/keras | Training classical and neural models |
| Sentiment Analysis | nltk.sentiment.vader, transformers, finbert-embedding | Emotion and tone measurement |
| Topic Modeling | gensim.models.ldamodel, pyLDAvis | Topic discovery and visualization |
| Visualization | matplotlib, seaborn, wordcloud, plotly | Trend analysis and result presentation |
| Explainability | lime, shap | Model interpretability |

---

## NLP Tasks

1. **Data Preparation**
   - Parse speech transcripts and metadata.
   - Clean and normalize text (remove punctuation, HTML tags, and noise).
   - Add political metadata (party, term period).
   - Convert to suitable feature formats (TF-IDF, embeddings).

2. **Exploratory Data Analysis (EDA)**
   - Word frequency and bigram analysis.
   - Sentiment trends over time.
   - Comparison of tone and topics by president and political party.

3. **Modeling**
   - Sentiment Classification: Predict speech tone using VADER or Logistic Regression.
   - Topic Modeling: Identify dominant themes via LDA.
   - Sequence Modeling: Use LSTM to capture contextual sentiment patterns.

4. **Evaluation**
   - Classification Metrics: Accuracy, Precision, Recall, F1-Score.
   - Topic Modeling Metric: Topic Coherence (C_v).
   - Explainability: SHAP or LIME to interpret model outputs.

5. **Visualization**
   - Sentiment over time (line plots).
   - Topic evolution by decade.
   - Word clouds of positive versus negative speeches.
   - Party-wise tone comparison.

---

## Evaluation Metrics

| Task | Metric | Description |
|------|---------|-------------|
| Sentiment Analysis | Accuracy, F1-Score | Compare predicted sentiment versus human label |
| Topic Modeling | Coherence Score | Evaluate quality of discovered topics |
| Classification | Precision, Recall | Judge classifier performance for tone or party prediction |
| Model Explainability | SHAP / LIME Values | Identify influential words and phrases |

---

## Expected Outcomes

By the end of this project, we expect to:
- Quantitatively measure how presidential tone evolved from the 18th century to the present.  
- Identify whether parties differ in emotional polarity or divisiveness.  
- Visualize topic evolution during major historical events (wars, crises, elections).  
- Demonstrate model interpretability using SHAP/LIME to explain what drives divisive rhetoric.

---

## Deployment Plan (PyCharm)

1. Clone the GitHub repository or download this project folder.  
2. Open it in PyCharm → File → Open → project_folder.  
3. Create a virtual environment (`python -m venv venv`) and activate it.  
4. Install dependencies:  
   ```bash
   
   pip install -r requirements.txt
   ```

## Author 
**Yonathan Shimelis**  
M.S. Data Science  
The George Washington University 

**Sayan Patra**  
M.S. Data Science  
The George Washington University  

---

## License

This project is licensed under the **MIT License** — see the `LICENSE` file for details.

---

## References

 The Miller Center of Public Affairs. (n.d.). *Presidential Speech Archive.* Retrieved from [https://data.millercenter.org/](https://data.millercenter.org/)  
 The American Presidency Project. (n.d.). *Public Papers of the Presidents.* Retrieved from [https://www.presidency.ucsb.edu/](https://www.presidency.ucsb.edu/)  
 Araci, D. (2019). *FinBERT: Financial Sentiment Analysis with Pretrained Language Models.* arXiv:1908.10063.  
 Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). *Latent Dirichlet Allocation.* *Journal of Machine Learning Research*, 3, 993–1022.  
 Lundberg, S. M., & Lee, S. I. (2017). *A Unified Approach to Interpreting Model Predictions (SHAP).* *Advances in Neural Information Processing Systems.*  
 Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). *"Why Should I Trust You?": Explaining the Predictions of Any Classifier.* *Proceedings of the 22nd ACM SIGKDD International      Conference on Knowledge Discovery and Data Mining.*  
 Hutto, C. J., & Gilbert, E. (2014). *VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text.* *Proceedings of the International AAAI Conference on Web and Social Media.*  

---
