import pandas as pd
import nltk
import re
import tkinter as tk
import customtkinter as ctk
from tkinter import messagebox
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    df = pd.read_csv("data/spam.csv")
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
except Exception as e:
    messagebox.showerror("Hata", f"Veri dosyası okunamadı: {e}")
    exit()

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

class SpamApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("🛡️ Profesyonel Spam Filtresi")
        self.geometry("600x650")
        self.resizable(False, False)

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        
        self.header_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.header_frame.grid(row=0, column=0, sticky="ew", pady=(30, 10))
        
        self.header_label = ctk.CTkLabel(self.header_frame, text="AI Spam Detector", 
                                       font=("Roboto Medium", 24))
        self.header_label.pack()
        
        self.sub_header = ctk.CTkLabel(self.header_frame, text="Yapay zeka ile mesaj güvenliğini analiz edin",
                                     font=("Roboto", 12), text_color="gray")
        self.sub_header.pack()

        self.main_card = ctk.CTkFrame(self, corner_radius=15)
        self.main_card.grid(row=1, column=0, sticky="nsew", padx=30, pady=20)
        self.main_card.grid_columnconfigure(0, weight=1)

        self.input_label = ctk.CTkLabel(self.main_card, text="Analiz Edilecek Mesaj", 
                                      font=("Roboto Medium", 14), anchor="w")
        self.input_label.grid(row=0, column=0, sticky="w", padx=20, pady=(20, 5))

        self.text_entry = ctk.CTkTextbox(self.main_card, height=120, font=("Roboto", 13), 
                                       corner_radius=10, border_color="#E0E0E0", border_width=1)
        self.text_entry.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 20))
        
        self.analyze_btn = ctk.CTkButton(self.main_card, text="ANALİZ ET", 
                                       font=("Roboto Medium", 14), height=45, corner_radius=8,
                                       fg_color="#0066CC", hover_color="#0052A3",
                                       command=self.tahmin_et)
        self.analyze_btn.grid(row=2, column=0, sticky="ew", padx=20, pady=10)

        self.result_frame = ctk.CTkFrame(self.main_card, corner_radius=10, fg_color="transparent")

        self.result_icon_label = ctk.CTkLabel(self.result_frame, text="", font=("Segoe UI Emoji", 40))
        self.result_icon_label.pack(pady=(10, 0))

        self.result_text_label = ctk.CTkLabel(self.result_frame, text="", font=("Roboto Medium", 18))
        self.result_text_label.pack(pady=5)

        self.result_desc_label = ctk.CTkLabel(self.result_frame, text="", font=("Roboto", 12))
        self.result_desc_label.pack(pady=(0, 15))

        self.action_frame = ctk.CTkFrame(self.main_card, fg_color="transparent")
        
        self.btn_keep = ctk.CTkButton(self.action_frame, text="Kalsın", width=120, height=35,
                                    fg_color="transparent", border_width=1, border_color="#A0A0A0", 
                                    text_color=("gray10", "gray90"), hover_color=("gray90", "gray20"),
                                    command=self.kalsin)
        self.btn_keep.pack(side="left", padx=10)

        self.btn_clear = ctk.CTkButton(self.action_frame, text="Sil & Sıfırla", width=120, height=35,
                                     fg_color="#DC3545", hover_color="#C82333",
                                     command=self.sil)
        self.btn_clear.pack(side="left", padx=10)

    def tahmin_et(self):
        mesaj = self.text_entry.get("1.0", "end-1c").strip()
        if not mesaj:
            return

        temiz = temizle(mesaj)
        vec = vectorizer.transform([temiz])
        sonuc = model.predict(vec)
        prob = model.predict_proba(vec)[0]

        self.result_frame.grid(row=3, column=0, sticky="ew", padx=20, pady=(20, 10))
        self.action_frame.grid(row=4, column=0, pady=(0, 20))
        
        if sonuc[0] == 1:
            confidence = prob[1] * 100
            self.result_frame.configure(fg_color=("#FDECEC", "#3E1B1B"))
            self.result_icon_label.configure(text="🚫")
            self.result_text_label.configure(text="Spam Tespit Edildi!", text_color="#DC3545")
            self.result_desc_label.configure(text=f"Bu mesaj %{confidence:.1f} oranında spam görünüyor.", text_color="gray")
        else:
            confidence = prob[0] * 100
            self.result_frame.configure(fg_color=("#E8F5E9", "#1B3E20"))
            self.result_icon_label.configure(text="✅")
            self.result_text_label.configure(text="Güvenli Mesaj", text_color="#28A745")
            self.result_desc_label.configure(text=f"Bu mesaj %{confidence:.1f} oranında güvenli.", text_color="gray")

    def sil(self):
        self.text_entry.delete("1.0", "end")
        self.hide_results()
        self.text_entry.focus()

    def kalsin(self):
        self.hide_results()
        self.text_entry.focus()

    def hide_results(self):
        self.result_frame.grid_forget()
        self.action_frame.grid_forget()

if __name__ == "__main__":
    app = SpamApp()
    app.mainloop()
