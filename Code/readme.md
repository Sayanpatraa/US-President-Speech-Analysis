# Presidential Rhetoric NLP Pipeline — README

This repository contains an end-to-end workflow for analyzing U.S. presidential rhetoric using:

- Transformer-based models  
- Classical machine learning models  
- A fully interactive Streamlit application for real-time inference  

This README explains **exactly how to run all project files in the correct order**.

## Project Files Overview

```
APP_analysis_code.ipynb             # Step 1: Data exploration + preprocessing
Transformers_.py                    # Step 2: Transformer-based training / feature extraction
Classifier_models.py                # Step 3: Classical ML model training
app.py                               # Step 4: Streamlit application (deployment)
presidential_statements_scraped.csv # Dataset file
requirements.txt                    # Python package dependencies
```

---

# 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs the scientific stack including numpy, pandas, scikit-learn, nltk, tqdm, matplotlib, transformers, torch, and streamlit.

---

# 2. Step 1 — Data Exploration & Preprocessing

### File: APP_analysis_code.ipynb

```bash
jupyter notebook APP_analysis_code.ipynb
```

This notebook performs:

- Loading the dataset (`presidential_statements_scraped.csv`)  
- Cleaning and text normalization  
- Token length analysis  
- Exploratory plots  
- Preparing preprocessed text for model training  

---

# 3. Step 2 — Train Transformer-Based Models

### File: Transformers_.py

```bash
python Transformers_.py
```

This script:

- Loads cleaned data  
- Tokenizes text using HuggingFace Transformers  
- Trains or fine-tunes a Transformer model  
- Generates features or embeddings for downstream classifiers  
- Saves trained model weights and tokenizers  

---

# 4. Step 3 — Train  Models

### File: Classifier_models.py

```bash
python Classifier_models.py
```

This script:

- Loads the dataset  
- Loads features generated from the Transformer model  
- Trains classical ML models  
- Saves resulting trained models for later use  
- Outputs evaluation metrics  

---

# 5. Step 4 — Launch the Streamlit Web Application

### File: app.py

```bash
streamlit run app.py
```

The application includes:

### Presidential Authorship Classification
### Sentiment Analysis (Integrated in UI)
### Reproducible Preprocessing
### Visualization Tools

---

# Execution Order Summary

| Step | File | Description |
|------|------|-------------|
| 1️ | APP_analysis_code.ipynb | Data cleaning + exploratory analysis |
| 2️ | Transformers_.py | Train Transformer model and generate features |
| 3️ | Classifier_models.py | Train classical ML models |
| 4️ | app.py | Launch Streamlit web application |

---

# Full Workflow Example

```bash
pip install -r requirements.txt
jupyter notebook APP_analysis_code.ipynb
python Transformers_.py
python Classifier_models.py
streamlit run app.py
```

---

# Troubleshooting Guide

- Ensure correct model paths  
- Ensure dataset is located properly  
- Clear Streamlit cache if errors persist  

---

# Citation

```
Patra, S. (2025). Analyzing U.S. Presidential Rhetoric Using Multi-Stage NLP Models.
George Washington University.
Shimelis, Y. (2025). Analyzing U.S. Presidential Rhetoric Using Multi-Stage NLP Models.
George Washington University.
```
