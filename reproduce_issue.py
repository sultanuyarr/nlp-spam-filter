
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Mock data from what we saw (roughly) or load file
try:
    df = pd.read_csv("data/spam.csv")
except:
    print("Could not read file")
    exit()

df['label'] = df['label'].map({'ham': 0, 'spam': 1})
stop_words = set(stopwords.words('turkish'))

def temizle(text):
    text = text.lower()
    text = re.sub(r'[^a-zçğıöşü\s]', '', text)
    kelimeler = text.split()
    kelimeler = [k for k in kelimeler if k not in stop_words]
    return " ".join(kelimeler)

df['clean_message'] = df['message'].apply(temizle)

# Print first few cleaned messages to see if they are empty
print("Cleaned data sample:")
print(df[['message', 'clean_message', 'label']].head())

X_train, X_test, y_train, y_test = train_test_split(
    df['clean_message'], df['label'], test_size=0.2, random_state=42
)

print(f"Train counts:\n{y_train.value_counts()}")

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

feature_names = vectorizer.get_feature_names_out()
print(f"Vocab size: {len(feature_names)}")
print(f"Vocab: {feature_names}")

test_inputs = ["Merhaba", "Nasılsın", "Normal mesaj", "Test mesajı", "Kazan", "Para"]

for msg in test_inputs:
    clean = temizle(msg)
    vec = vectorizer.transform([clean])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    print(f"Input: '{msg}' -> Clean: '{clean}' -> Pred: {pred} ({'SPAM' if pred==1 else 'NORMAL'}), Proba: {proba}")
