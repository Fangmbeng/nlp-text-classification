import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from tqdm import tqdm

# Set seed value
np.random.seed(42)

# Load data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
# Load data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Print the number of classes and their counts
class_counts = train_df['label'].value_counts()
print('Class counts:')
print(class_counts)

# Visualize the class distribution
sns.countplot(x='label', data=train_df)
plt.title('Class distribution in training data')
plt.show()

# Solve imbalance problem
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_df['label']), y=train_df['label'])

# Tokenization, normalization, stopword removal, lemmatization, and typo correction
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

def preprocess_text(text):
    # Tokenize
    words = word_tokenize(text.lower())

    # Remove punctuation
    words = [re.sub(r'[^\w\s]', '', w) for w in words if w]

    # Remove stopwords
    words = [w for w in words if w not in stopwords.words('english')]

    # Typo correction
    words = [nltk.corpus.reader.wordnet.synsets(w, pos='v')[0].lemmas()[0].name_ for w in words if len(nltk.corpus.reader.wordnet.synsets(w, pos='v')) > 0]

    # Lemmatization
    words = [WordNetLemmatizer().lemmatize(w) for w in words]

    return ' '.join(words)

train_df['text'] = train_df['text'].apply(preprocess_text)
test_df['text'] = test_df['text'].apply(preprocess_text)

# Vectorize the data using TF-IDF and SVM
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 3), stop_words='english')
X = vectorizer.fit_transform(train_df['text'])
y = train_df['label']

# K-fold cross-validation
kf = KFold(n_splits=5, shuffle=True)
fold = 1

# Initialize results dataframe
results = pd.DataFrame(columns=['fold', 'train_samples', 'test_samples', 'f1_score'])

# Loop through each fold
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Fit the model
    clf = SVC(kernel='linear', class_weight=class_weights)
    clf.fit(X_train, y_train)

    # Predict on the test set
    y_pred = clf.predict(X_test)

    # Calculate F1 score
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Save results
    results = results.append({'fold': fold, 'train_samples': len(X_train), 'test_samples': len(X_test), 'f1_score': f1}, ignore_index=True)

    # Print progress bar and F1 score
    print(f'Fold {fold}: Training samples: {len(X_train)}, Test samples: {len(X_test)}, F1 score: {f1:.4f}')
    fold += 1

# Print overall results
print(results)

# Predict on the test set
y_pred = clf.predict(vectorizer.transform(test_df['text']))

# Save predictions to file
test_df[['id', 'label']] = pd.DataFrame(y_pred, columns=['label'])
test_df.to_csv('./0316_test.csv', index=False)
