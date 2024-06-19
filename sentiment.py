import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import re
import pickle

df = pd.read_excel('IMDB-Movie-Reviews-Large-Dataset-50k/train.xlsx')

def get_clean(x):
    x = str(x).lower()
    x = re.sub(r'[^a-zA-Z\s]', '', x)
    x = re.sub(r'\s+', ' ', x)
    return x.strip()

df['Reviews'] = df['Reviews'].apply(lambda x: get_clean(x))

tfidf = TfidfVectorizer(max_features=5000)
X = df['Reviews']
y = df['Sentiment']
X = tfidf.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

clf = LinearSVC()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))

with open('sentiment_analysis.pkl', 'wb') as file:
    pickle.dump((clf, tfidf), file)
