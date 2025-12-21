import pandas as pd
import nltk
import re
import tkinter as tk
from tkinter import messagebox

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

nltk.download('stopwords')

# ======================
# VERİ ve MODEL
# ======================

df = pd.read_csv("data/spam.csv")
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

stop_words = set(stopwords.words('turkish'))

def temizle(text):
    text = text.lower()
    text = re.sub(r'[^a-zçğıöşü\s]', '', text)
    kelimeler = text.split()
    kelimeler = [k for k in kelimeler if k not in stop_words]
    return " ".join(kelimeler)

df['clean_message'] = df['message'].apply(temizle)

X_train, X_test, y_train, y_test = train_test_split(
    df['clean_message'], df['label'], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

# ======================
# TAHMİN FONKSİYONU
# ======================

def tahmin_et():
    mesaj = text_entry.get("1.0", tk.END).strip()
    if not mesaj:
        messagebox.showwarning("Uyarı", "Lütfen bir mesaj girin")
        return

    temiz = temizle(mesaj)
    vec = vectorizer.transform([temiz])
    sonuc = model.predict(vec)

    if sonuc[0] == 1:
        sonuc_label.config(text="SONUÇ: SPAM ❌", fg="red")
    else:
        sonuc_label.config(text="SONUÇ: NORMAL ✅", fg="green")

# ======================
# ARAYÜZ
# ======================

root = tk.Tk()
root.title("Spam Filtresi")
root.geometry("400x300")

tk.Label(root, text="Mesajınızı Girin:", font=("Arial", 12)).pack(pady=5)

text_entry = tk.Text(root, height=5, width=40)
text_entry.pack(pady=5)

tk.Button(root, text="Tahmin Et", command=tahmin_et, font=("Arial", 11)).pack(pady=10)

sonuc_label = tk.Label(root, text="", font=("Arial", 14))
sonuc_label.pack(pady=10)

root.mainloop()
