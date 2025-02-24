# nlp-text-classification
Here's a description you can use for your GitHub repository:  

---

# Text Classification with SVM and TF-IDF  

This project implements a text classification pipeline using **Support Vector Machines (SVM)** with **TF-IDF vectorization**. The model is trained and evaluated using **K-Fold Cross-Validation**, addressing class imbalance with class weighting.  

## Features  

- **Preprocessing:**  
  - Tokenization  
  - Lowercasing  
  - Stopword removal  
  - Lemmatization  
  - Typo correction  

- **Feature Engineering:**  
  - TF-IDF vectorization with n-grams (unigram, bigram, trigram)  
  - Max features = 10,000  

- **Model Training & Evaluation:**  
  - Support Vector Classifier (SVC) with a linear kernel  
  - Class balancing using `compute_class_weight`  
  - **5-fold Cross-Validation**  
  - F1-score as the evaluation metric  

- **Predictions:**  
  - The model predicts labels for the test dataset and saves them to a CSV file.  

## Dependencies  

- `pandas`  
- `numpy`  
- `re`  
- `nltk`  
- `scikit-learn`  
- `tqdm`  
- `matplotlib` and `seaborn` (for data visualization)  

## Usage  

1. Install dependencies:  
   ```bash
   pip install pandas numpy nltk scikit-learn tqdm matplotlib seaborn
   ```
2. Run the script:  
   ```bash
   python text_classification.py
   ```  
   Ensure `train.csv` and `test.csv` exist in the same directory.  

## Output  

- The script prints F1-scores for each fold during training.  
- The final predictions are saved as `0316_test.csv`.  

## Notes  

- Make sure to download NLTK stopwords and lemmatizer datasets before running the script:  
  ```python
  import nltk
  nltk.download('punkt')
  nltk.download('wordnet')
  nltk.download('stopwords')
  ```  
- The typo correction step may cause errors if words lack WordNet synonyms. Consider refining this step for production use.  

---

Would you like to add any specific details about dataset assumptions or model improvements? 🚀
