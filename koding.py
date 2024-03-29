# -*- coding: utf-8 -*-
"""Koding.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1w1SQ1yVcHdVC3n1MZk7bVoGlSLB46pzc
"""

import pandas as pd
import math
import re
import nltk
import Sastrawi
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter

sw = stopwords.words('indonesian')

data = pd.read_csv('sample dataset.csv')
data

# def clean_data(data, stopwords=False, words_to_exclude=None):
#     cleaned_data = re.sub(r"[^\w\s]", "", data)
#     cleaned_data = re.sub(r"[\d]", "", cleaned_data)
#     cleaned_data = stemmer.stem(cleaned_data)

#     if stopwords:
#         if words_to_exclude is None:
#             words_to_exclude = []
#         cleaned_data = " ".join([word if word.lower() not in map(str.lower, sw) or word.lower() in map(str.lower, words_to_exclude) else "" for word in cleaned_data.split()])

#     return cleaned_data

def clean_data(sentence, stopwords=True, words_to_exclude=[]):
    cleaned_sentence = []
    words = sentence.lower().split()

    for word in words:
        if word in words_to_exclude:
            cleaned_sentence.append(word)
        elif stopwords and word not in sw:
            cleaned_sentence.append(word)

    cleaned_data = ' '.join(cleaned_sentence)
    cleaned_data = re.sub(r"[^\w\s]", "", cleaned_data)
    cleaned_data = re.sub(r"[\d]", "", cleaned_data)


    return cleaned_data

# Membersihkan seluruh daftar kalimat dalam kolom 'Question'
excluded_words = ['apa', 'mengapa', 'kapan', 'dimana', 'siapa', 'bagaimana','berapa']
cleaned_sentences = data['Question'].apply(lambda x: clean_data(x, stopwords=True, words_to_exclude=excluded_words)).tolist()

# print(cleaned_sentences)

# excluded_words = ['apa', 'mengapa', 'kapan', 'dimana', 'siapa', 'bagaimana','berapa']
# cleaned_sentences = data['Question'].apply(lambda x: clean_datas(x, stopwords=True, words_to_exclude=excluded_words)).tolist()
# cleaned_sentences
# questions_with_keywords = [question for question in cleaned_sentences if any(keyword in question.lower() for keyword in excluded_words)]
# questions_with_keywords

print("Masukkan Pertanyaan Anda!")
user_input = input()
cleaned_user_input = clean_data(user_input, stopwords=True, words_to_exclude=excluded_words)
# cleaned_user_input

cleaned_sentences.insert(0, cleaned_user_input)

# for i in cleaned_sentences :
#   print (i)

def tokenisasi(list_of_strings):
    tokenized_list = []
    for string in list_of_strings:
        tokens = string.split() # Memisahkan string menjadi token dengan spasi sebagai pemisah
        tokenized_list.append(tokens)
    return tokenized_list

hasil_tokenisasi = tokenisasi(cleaned_sentences)

# for i in hasil_tokenisasi :
#   print(i)

tf_scores = []
for sentence in hasil_tokenisasi:
    word_count = Counter(sentence)  # Menghitung jumlah kata dalam setiap dokumen
    total_words = len(sentence)  # Total jumlah kata dalam dokumen
    tf = {word: count / total_words for word, count in word_count.items()}  # Menghitung TF untuk setiap kata
    tf_scores.append(tf)

# Membuat DataFrame pandas dari hasil TF
tf_data = {}
for idx, tf in enumerate(tf_scores):
    tf_data[f"TF Dokumen {idx+1}"] = tf

# Mengonversi ke DataFrame
tf_df = pd.DataFrame(tf_data).fillna(0)

# Menampilkan DataFrame hasil TF
# print(tf_df)

# Menghitung jumlah dokumen yang mengandung setiap kata
doc_frequency = {}
for sentence in hasil_tokenisasi:
    unique_words = set(sentence)
    for word in unique_words:
        doc_frequency[word] = doc_frequency.get(word, 0) + 1

# Menghitung IDF untuk setiap kata
num_documents = len(hasil_tokenisasi)
idf_scores = {}
for word, freq in doc_frequency.items():
    idf = math.log(num_documents / (freq + 1))  # Formula IDF
    idf_scores[word] = idf

# # Menampilkan hasil IDF untuk setiap kata
# print("\nIDF Scores:")
# for word, idf in idf_scores.items():
#     print(f"{word}: {idf}")

# Menghitung TF-IDF untuk setiap kata dalam setiap dokumen
tfidf_scores = []
for tf in tf_scores:
    tfidf = {word: tf[word] * idf_scores[word] for word in tf}  # Menghitung TF-IDF untuk setiap kata
    tfidf_scores.append(tfidf)

# Mengonversi ke DataFrame pandas
tfidf_data = {}
for idx, tfidf in enumerate(tfidf_scores):
    tfidf_data[f"TF-IDF Dokumen {idx+1}"] = tfidf

# Membuat DataFrame dari hasil TF-IDF
tfidf_df = pd.DataFrame(tfidf_data).fillna(0)

# # Menampilkan DataFrame hasil TF-IDF
# print(tfidf_df)

# Transpose DataFrame untuk mendapatkan kolom yang merepresentasikan TF-IDF dari dokumen pertama (indeks ke-0)
base_document = tfidf_df.iloc[:, 0:1]  # Dokumen pertama sebagai base document
other_documents = tfidf_df.iloc[:, 1:]  # Dokumen lainnya
base_document

# Menghitung cosine similarity menggunakan fungsi cosine_similarity
similarities = []
for idx, column in other_documents.items():
    dot_product = (base_document.values.flatten() * column.values).sum()  # Perkalian titik antara dua vektor
    base_norm = math.sqrt((base_document ** 2).sum().values[0])  # Norma vektor dokumen pertama
    other_norm = math.sqrt((column ** 2).sum())  # Norma vektor dokumen lainnya
    similarity = dot_product / (base_norm * other_norm) if (base_norm * other_norm) != 0 else 0  # Cosine similarity
    similarities.append(similarity)

# Membuat pasangan indeks dokumen dengan nilai cosine similarity
similarity_pairs = list(zip(range(1, len(similarities) + 1), similarities))  # Indeks dimulai dari 1

# Mengurutkan pasangan berdasarkan nilai cosine similarity dari yang terbesar ke terkecil
sorted_pairs = sorted(similarity_pairs, key=lambda x: x[1], reverse=True)

# Mengambil hanya 5 dokumen dengan nilai cosine similarity tertinggi
top_5_similar = sorted_pairs[:5]

# Menampilkan dokumen beserta nilai cosine similarity yang telah diurutkan
print("Dokumen dengan nilai cosine similarity terurut:")
for idx, sim in top_5_similar:
    if sim > 0:
        print(f"Isi Dokumen {idx}: {hasil_tokenisasi[idx]}")
        print(f"Nilai Cosine Similarity: {sim}")

# pattern_based = [
#     ['dimana', '<keyword>'],
#     ['siapa', '<keyword>'],
#     ['kapan', '<keyword>'],
#     ['apa', '<keyword>'],
#     ['kenapa', '<keyword>'],
#     ['bagaimana', '<keyword>']
# ]

# # Mengambil hanya indeks dokumen teratas dari nilai cosine similarity tertinggi
# top_document_indexes = [idx for idx, sim in top_5_similar]

# # Menggunakan kata-kata dari dokumen teratas untuk membentuk pola pertanyaan
# for doc_idx in top_document_indexes:
#     kata_kunci = clean_token[doc_idx]  # clean_token adalah variabel dokumen yang telah dibersihkan
#     for pattern in pattern_based:
#         # Mencetak pola pertanyaan dengan mengganti '<keyword>' dengan kata-kata dari dokumen teratas
#         pertanyaan = ' '.join(pattern).replace('<keyword>', ' '.join(kata_kunci))
#         print(pertanyaan)

# print("Isi dari 5 dokumen dengan nilai cosine similarity tertinggi:")
# for idx, sim in top_5_similar:
#     if sim > 0:
#         print(f"Isi Dokumen {idx}: {hasil_tokenisasi[idx]}")

# def eliminate_words(token_lists):
#     words_to_eliminate = ['apa', 'mengapa', 'dimana', 'kapan', 'siapa', 'bagaimana']

#     # Eliminate specified words from each list of tokens
#     cleaned_lists = []
#     for token_list in token_lists:
#         cleaned = [word for word in token_list if word.lower() not in words_to_eliminate]
#         cleaned_lists.append(cleaned)

#     return cleaned_lists

question_categories = {
    'apa': 'objek',
    'mengapa': 'alasan',
    'kapan': 'waktu',
    'dimana': 'lokasi',
    'siapa': 'orang',
    'bagaimana': 'cara',
    'berapa' : 'nominal'
}

# Membuat kamus untuk menyimpan dokumen-dokumen berdasarkan kategori pertanyaan
categorized_docs = {category: [] for category in question_categories.values()}


# Memeriksa setiap dokumen dalam top 5 hasil cosine similarity
for idx, sim in top_5_similar:
    if sim > 0:
        doc = hasil_tokenisasi[idx]
        for word in doc:
            if word in question_categories:
                category = question_categories[word]
                categorized_docs[category].append((idx, doc))  # Menyimpan index dan dokumen dalam kategori
                break  # Hanya satu kata kunci yang dibutuhkan untuk masuk ke kategori

# Memasukkan input ke dalam kategori yang sesuai
input_category = None
for word in hasil_tokenisasi[0]:
    if word in question_categories:
        input_category = question_categories[word]
        categorized_docs[input_category]
        break

# Mengambil dokumen-dokumen yang sesuai dengan kategori dari inputan
relevant_docs = categorized_docs[input_category] if input_category else []


# Menampilkan dokumen-dokumen yang sesuai dengan kategori dari inputan
# if relevant_docs:
#     print(f"Dokumen dalam kategori '{input_category}':")
#     for idx, doc in relevant_docs:
#         print(f"Index ke-{idx}: {doc}")
#         # answer_text = data.loc[idx, 'Answer']  # Mendapatkan jawaban berdasarkan indeks
#         # print( "JAWABAN")
#         # print(answer_text)

# else:
#     print("Tidak ada dokumen yang sesuai dengan kategori dari input.")

import re

# Function to extract pattern from a sentence
def extract_pattern(sentence):
    patterns = {
        "Siapa": r"siapa \w+",
        "Dimana": r"dimana \w+",
        "Kapan": r"kapan \w+",
        "Apa": r"apa \w+",
        "Kenapa": r"kenapa \w+",
        "Bagaimana": r"bagaimana \w+",
        "Berapa" : r"berapa \w+"
    }
    for pattern_name, pattern in patterns.items():
          match = re.search(pattern, sentence)
          if match:
              return pattern_name, match.group(0)

    return None, None

# Ambil hanya indeks dari top 5 similar
candidate_indices = [idx for idx, doc in relevant_docs]

# Ambil pertanyaan-pertanyaan yang sesuai dari clean_token
candidate_questions = [hasil_tokenisasi[idx] for idx in candidate_indices]

input_user = ' '.join(hasil_tokenisasi[0])

# Extracting pattern from user_input
user_pattern, user_match = extract_pattern(input_user)

# Tampilkan indeks sebelumnya dari top 5 similar
previous_indices = [idx for idx, _ in top_5_similar if idx in candidate_indices]

result_texts = []

if user_pattern:
    print(f"Pola dari user_input: {user_match} dengan hasil kandidat yang benar yaitu:")
    for index, question in enumerate(candidate_questions):
        candidate_pattern, candidate_match = extract_pattern(' '.join(question))
        if candidate_pattern == user_pattern:
            question_text = ' '.join(question)
            keyword_length = len(candidate_pattern.split()) - 1
            words_after_keyword = question_text.split()[keyword_length:]
            # Menghilangkan kata kunci dari hasil
            result_text = ' '.join(words_after_keyword[1:]) if len(words_after_keyword) > 1 else ''
            result_texts.append((previous_indices[index], result_text))  # Simpan result_text ke dalam daftar

            print(f"Index ke-{previous_indices[index]}: {result_text}")

            # answer_text = data.loc[previous_indices[index], 'Answer']  # Mendapatkan jawaban berdasarkan indeks
            # print( "JAWABAN")
            # print(answer_text)

else:
    print("Tidak ada pola yang cocok dengan user_input.")


def count_similar_words(target_text, text):
    target_words = set(target_text.split())
    text_words = set(text.split())
    similar_words = target_words.intersection(text_words)
    return len(similar_words)

def find_most_similar_text_with_index(target_text, result_texts):
    max_similarity = 0
    most_similar_text = ""
    most_similar_index = -1

    for index, text in result_texts:
        similarity_count = count_similar_words(target_text, text)
        if similarity_count > max_similarity:
            max_similarity = similarity_count
            most_similar_text = text
            most_similar_index = index

    return most_similar_text, most_similar_index


# Menggabungkan clean_token[0] menjadi satu teks tanpa tokenisasi
target_text = ' '.join(hasil_tokenisasi[0])

most_similar_text, most_similar_index = find_most_similar_text_with_index(target_text, result_texts)
if most_similar_text != "":
    print("Teks yang paling mirip:", most_similar_text)
    print("Index dari teks yang paling mirip:", most_similar_index-1)
    print("Pertanyaan : ", user_input)
    answer_text = data.loc[most_similar_index-1, 'Answer']  # Mendapatkan jawaban berdasarkan indeks
    print( "JAWABAN")
    print(answer_text)
else:
    print("Tidak ada teks yang cocok atau hasil kosong.")



