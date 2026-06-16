import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('Panduan Tubes/tourist_reviews.csv')

print("=" * 60)
print("TAHAP 1: DATA EXPLORATION")
print("=" * 60)

print("\n--- Informasi Dataset ---")
print(f"Jumlah baris: {df.shape[0]}")
print(f"Jumlah kolom: {df.shape[1]}")
print(f"Kolom: {list(df.columns)}")

print("\n--- Info Tipe Data ---")
print(df.dtypes)

print("\n--- 5 Data Pertama ---")
print(df.head())

print("\n--- Statistik Deskriptif ---")
print(df.describe(include='all'))

print("\n--- Jumlah Missing Values ---")
print(df.isnull().sum())

print("\n--- Jumlah Duplikat ---")
print(f"Baris duplikat: {df.duplicated().sum()}")

target_cols = ['accessibility', 'facility', 'activity']

print("\n--- Distribusi Kelas ---")
for col in target_cols:
    print(f"\nDistribusi '{col}':")
    dist = df[col].value_counts()
    print(dist)
    total = dist.sum()
    for label, count in dist.items():
        print(f"  {label}: {count / total * 100:.2f}%")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
colors = ['#2ecc71', '#e74c3c', '#3498db']

for i, col in enumerate(target_cols):
    dist = df[col].value_counts()
    axes[i].bar(dist.index, dist.values, color=colors)
    axes[i].set_title(f'Distribusi Kelas: {col}', fontsize=14, fontweight='bold')
    axes[i].set_xlabel('Sentimen')
    axes[i].set_ylabel('Jumlah')
    for j, v in enumerate(dist.values):
        axes[i].text(j, v + 10, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('distribusi_kelas.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nGrafik distribusi kelas disimpan sebagai 'distribusi_kelas.png'")

print("\n--- Identifikasi Imbalanced Data ---")
for col in target_cols:
    dist = df[col].value_counts()
    ratio = dist.min() / dist.max()
    print(f"\n{col}:")
    print(f"  Kelas mayoritas: {dist.idxmax()} ({dist.max()})")
    print(f"  Kelas minoritas: {dist.idxmin()} ({dist.min()})")
    print(f"  Rasio min/max: {ratio:.4f}")
    if ratio < 0.5:
        print(f"  Status: IMBALANCED (rasio < 0.5)")
    else:
        print(f"  Status: BALANCED (rasio >= 0.5)")

print("\n" + "=" * 60)
print("TAHAP 2: DATA CLEANSING")
print("=" * 60)

print(f"\nJumlah data sebelum cleansing: {df.shape[0]}")

df_clean = df.copy()

print("\n--- Hapus Baris Duplikat ---")
before = df_clean.shape[0]
df_clean = df_clean.drop_duplicates()
after = df_clean.shape[0]
print(f"Baris duplikat dihapus: {before - after}")

print("\n--- Hapus Baris Kosong (pada kolom text & target) ---")
before = df_clean.shape[0]
df_clean = df_clean.dropna(subset=['text'] + target_cols)
after = df_clean.shape[0]
print(f"Baris kosong dihapus: {before - after}")

print("\n--- Case Folding ---")
df_clean['text_clean'] = df_clean['text'].astype(str).str.lower()
print("Case folding selesai.")
print(f"Contoh: '{df_clean['text'].iloc[0][:80]}...'")
print(f"     -> '{df_clean['text_clean'].iloc[0][:80]}...'")

print("\n--- Pembersihan dengan Regex ---")

def clean_text(text):
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df_clean['text_clean'] = df_clean['text_clean'].apply(clean_text)
print("Pembersihan URL, mention, hashtag, tanda baca, dan angka selesai.")
print(f"Contoh hasil: '{df_clean['text_clean'].iloc[0][:100]}...'")

print("\n--- Hapus Stopword Bahasa Indonesia ---")
stop_factory = StopWordRemoverFactory()
stopword_remover = stop_factory.create_stop_word_remover()
stopwords_list = stop_factory.get_stop_words()
print(f"Jumlah stopword Sastrawi: {len(stopwords_list)}")

def remove_stopwords(text):
    words = text.split()
    filtered = [w for w in words if w not in stopwords_list]
    return ' '.join(filtered)

df_clean['text_clean'] = df_clean['text_clean'].apply(remove_stopwords)
print("Stopword removal selesai.")
print(f"Contoh hasil: '{df_clean['text_clean'].iloc[0][:100]}...'")

print("\n--- Stemming dengan Sastrawi ---")
stem_factory = StemmerFactory()
stemmer = stem_factory.create_stemmer()

def stem_text(text):
    return stemmer.stem(text)

print("Proses stemming sedang berjalan (ini membutuhkan waktu)...")
df_clean['text_clean'] = df_clean['text_clean'].apply(stem_text)
print("Stemming selesai.")
print(f"Contoh hasil: '{df_clean['text_clean'].iloc[0][:100]}...'")

print("\n--- Hapus Baris dengan Teks Kosong Setelah Cleansing ---")
before = df_clean.shape[0]
df_clean = df_clean[df_clean['text_clean'].str.strip().astype(bool)]
after = df_clean.shape[0]
print(f"Baris dengan teks kosong dihapus: {before - after}")

print(f"\nJumlah data setelah cleansing: {df_clean.shape[0]}")

print("\n--- Perbandingan Sebelum dan Sesudah Cleansing ---")
sample_indices = df_clean.index[:3]
for idx in sample_indices:
    print(f"\nBaris {idx}:")
    print(f"  Sebelum: {df_clean.loc[idx, 'text'][:100]}...")
    print(f"  Sesudah: {df_clean.loc[idx, 'text_clean'][:100]}...")

df_clean.to_csv('tourist_reviews_cleaned.csv', index=False)
print("\nData bersih disimpan sebagai 'tourist_reviews_cleaned.csv'")

print("\n" + "=" * 60)
print("TAHAP 3: DATA TRANSFORMATION")
print("=" * 60)

print("\n--- Label Encoding ---")
label_encoders = {}
df_transformed = df_clean.copy()

for col in target_cols:
    le = LabelEncoder()
    df_transformed[f'{col}_encoded'] = le.fit_transform(df_transformed[col])
    label_encoders[col] = le
    print(f"\n{col}:")
    for i, label in enumerate(le.classes_):
        print(f"  {label} -> {i}")

print("\n--- Ekstraksi Fitur dengan TF-IDF ---")
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf_vectorizer.fit_transform(df_transformed['text_clean'])
print(f"Dimensi matriks TF-IDF: {X_tfidf.shape}")
print(f"Jumlah fitur (kata unik): {len(tfidf_vectorizer.get_feature_names_out())}")

top_n = 20
feature_names = tfidf_vectorizer.get_feature_names_out()
tfidf_mean = X_tfidf.mean(axis=0).A1
top_indices = tfidf_mean.argsort()[-top_n:][::-1]
print(f"\nTop {top_n} kata berdasarkan rata-rata TF-IDF:")
for rank, idx in enumerate(top_indices, 1):
    print(f"  {rank}. {feature_names[idx]}: {tfidf_mean[idx]:.6f}")

print("\n--- SMOTE untuk Menangani Imbalanced Data ---")
smote = SMOTE(random_state=42)

X_resampled = {}
y_resampled = {}

for col in target_cols:
    y = df_transformed[f'{col}_encoded'].values
    print(f"\n{'=' * 40}")
    print(f"SMOTE untuk target: {col}")
    print(f"{'=' * 40}")
    
    print(f"\nDistribusi SEBELUM SMOTE:")
    unique, counts = np.unique(y, return_counts=True)
    for u, c in zip(unique, counts):
        original_label = label_encoders[col].inverse_transform([u])[0]
        print(f"  {original_label} ({u}): {c}")
    
    X_res, y_res = smote.fit_resample(X_tfidf, y)
    
    X_resampled[col] = X_res
    y_resampled[col] = y_res
    
    print(f"\nDistribusi SESUDAH SMOTE:")
    unique, counts = np.unique(y_res, return_counts=True)
    for u, c in zip(unique, counts):
        original_label = label_encoders[col].inverse_transform([u])[0]
        print(f"  {original_label} ({u}): {c}")
    
    print(f"\nUkuran data: {X_tfidf.shape[0]} -> {X_res.shape[0]}")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for i, col in enumerate(target_cols):
    y_before = df_transformed[f'{col}_encoded'].values
    y_after = y_resampled[col]
    
    labels_before = label_encoders[col].inverse_transform(np.unique(y_before))
    counts_before = [np.sum(y_before == v) for v in np.unique(y_before)]
    
    labels_after = label_encoders[col].inverse_transform(np.unique(y_after))
    counts_after = [np.sum(y_after == v) for v in np.unique(y_after)]
    
    axes[0][i].bar(labels_before, counts_before, color=colors)
    axes[0][i].set_title(f'{col} - Sebelum SMOTE', fontsize=12, fontweight='bold')
    axes[0][i].set_xlabel('Sentimen')
    axes[0][i].set_ylabel('Jumlah')
    for j, v in enumerate(counts_before):
        axes[0][i].text(j, v + 10, str(v), ha='center', fontweight='bold')
    
    axes[1][i].bar(labels_after, counts_after, color=colors)
    axes[1][i].set_title(f'{col} - Sesudah SMOTE', fontsize=12, fontweight='bold')
    axes[1][i].set_xlabel('Sentimen')
    axes[1][i].set_ylabel('Jumlah')
    for j, v in enumerate(counts_after):
        axes[1][i].text(j, v + 10, str(v), ha='center', fontweight='bold')

plt.suptitle('Perbandingan Distribusi Kelas Sebelum dan Sesudah SMOTE', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('distribusi_smote.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nGrafik perbandingan SMOTE disimpan sebagai 'distribusi_smote.png'")

print("\n" + "=" * 60)
print("RINGKASAN DATA PREPARATION")
print("=" * 60)
print(f"Jumlah data awal: {df.shape[0]}")
print(f"Jumlah data setelah cleansing: {df_clean.shape[0]}")
print(f"Jumlah fitur TF-IDF: {X_tfidf.shape[1]}")
for col in target_cols:
    print(f"Jumlah data setelah SMOTE ({col}): {X_resampled[col].shape[0]}")
print(f"\nFile output:")
print(f"  - tourist_reviews_cleaned.csv")
print(f"  - distribusi_kelas.png")
print(f"  - distribusi_smote.png")
