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
from flask import Flask, render_template, request

app = Flask(__name__)

sw = stopwords.words('indonesian')
excluded_words = ['apa', 'mengapa', 'kapan', 'dimana', 'siapa', 'bagaimana', 'berapa', 'kenapa']
data = pd.read_csv('Dataset Ori.csv')



def preprocess_and_tokenize(sentence, stopwords=True, words_to_exclude=[]):
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

    tokenized_list = []
    tokenized_list = cleaned_data.split() 
 

    return tokenized_list

# def calculate_cosine_similarity(hasil_tokenisasi):
#     # Menghitung TF
#     tf_scores = []
#     for sentence in hasil_tokenisasi:
#         word_count = Counter(sentence)  # Menghitung jumlah kata dalam setiap dokumen
#         total_words = len(sentence)  # Total jumlah kata dalam dokumen
#         tf = {word: count / total_words for word, count in word_count.items()}  # Menghitung TF untuk setiap kata
#         tf_scores.append(tf)

#     # Membuat DataFrame pandas dari hasil TF
#     tf_data = {}
#     for idx, tf in enumerate(tf_scores):
#         tf_data[f"TF Dokumen {idx+1}"] = tf

#     # Mengonversi ke DataFrame
#     tf_df = pd.DataFrame(tf_data).fillna(0)

#     # Menghitung IDF
#     doc_frequency = {}
#     for sentence in hasil_tokenisasi:
#         unique_words = set(sentence)
#         for word in unique_words:
#             doc_frequency[word] = doc_frequency.get(word, 0) + 1

#     num_documents = len(hasil_tokenisasi)
#     idf_scores = {}
#     for word, freq in doc_frequency.items():
#         idf = math.log(num_documents / (freq + 1))  # Formula IDF
#         idf_scores[word] = idf

#     # Menghitung TF-IDF
#     tfidf_scores = []
#     for tf in tf_scores:
#         tfidf = {word: tf[word] * idf_scores[word] for word in tf}  # Menghitung TF-IDF untuk setiap kata
#         tfidf_scores.append(tfidf)

#     # Mengonversi ke DataFrame pandas
#     tfidf_data = {}
#     for idx, tfidf in enumerate(tfidf_scores):
#         tfidf_data[f"TF-IDF Dokumen {idx+1}"] = tfidf

#     # Membuat DataFrame dari hasil TF-IDF
#     tfidf_df = pd.DataFrame(tfidf_data).fillna(0)

#     # Transpose DataFrame untuk mendapatkan kolom yang merepresentasikan TF-IDF dari dokumen pertama (indeks ke-0)
#     base_document = tfidf_df.iloc[:, 0:1]  # Dokumen pertama sebagai base document
#     other_documents = tfidf_df.iloc[:, 1:]  # Dokumen lainnya
#     base_document

#     # Menghitung cosine similarity menggunakan fungsi cosine_similarity
#     similarities = []
#     for idx, column in other_documents.items():
#         dot_product = (base_document.values.flatten() * column.values).sum()  # Perkalian titik antara dua vektor
#         base_norm = math.sqrt((base_document ** 2).sum().values[0])  # Norma vektor dokumen pertama
#         other_norm = math.sqrt((column ** 2).sum())  # Norma vektor dokumen lainnya
#         similarity = dot_product / (base_norm * other_norm) if (base_norm * other_norm) != 0 else 0  # Cosine similarity
#         similarities.append(similarity)

#     # Membuat pasangan indeks dokumen dengan nilai cosine similarity
#     similarity_pairs = list(zip(range(1, len(similarities) + 1), similarities))  # Indeks dimulai dari 1

#     # Mengurutkan pasangan berdasarkan nilai cosine similarity dari yang terbesar ke terkecil
#     sorted_pairs = sorted(similarity_pairs, key=lambda x: x[1], reverse=True)

#     # Mengambil hanya 5 dokumen dengan nilai cosine similarity tertinggi
#     top_5_similar = sorted_pairs[:5]

#     #     # Menyimpan hasil dalam list of dictionaries
#     # results = []
#     # for idx, sim in top_5_similar:
#     #     if sim > 0:
#     #         result = {
#     #             'idx': idx,
#     #             'sim': sim,
#     #             'text': hasil_tokenisasi[idx]
#     #         }
#     #         results.append(result)

#     return top_5_similar

def rule_based(cleaned_sentences, cleaned_user_input):
    question_categories = {
    'apa': 'objek',
    'mengapa': 'alasan',
    'kapan': 'waktu',
    'dimana': 'lokasi',
    'siapa': 'orang',
    'bagaimana': 'cara',
    'berapa' : 'nominal',
    'kenapa' : 'alasan'
    }
    categorized_docs = {category: [] for category in question_categories.values()}

    for index, sentence in enumerate(cleaned_sentences):
        for word in sentence :
                if word in question_categories:
                    category = question_categories[word]
                    categorized_docs[category].append((index, sentence))  # Menyimpan index dan dokumen dalam kategori
                    break 

    input_category = None
    for word in cleaned_user_input:
        if word in question_categories:
            input_category = question_categories[word]
            categorized_docs[input_category]
            break

    relevant_docs = categorized_docs[input_category] if input_category else []

    return relevant_docs

# def extract_pattern(sentence):
#     patterns = {
#         "Siapa": r"siapa \w+",
#         "Dimana": r"dimana \w+",
#         "Kapan": r"kapan \w+",
#         "Apa": r"apa \w+",
#         "Kenapa": r"kenapa \w+",
#         "Bagaimana": r"bagaimana \w+",
#         "Berapa" : r"berapa \w+",
#         "kenapa" : r"kenapa \w+"
#     }
#     for pattern_name, pattern in patterns.items():
#           match = re.search(pattern, sentence)
#           if match:
#               return pattern_name, match.group(0)

#     return None, None

def count_similar_words(target_text, text):
    target_words = set(target_text.split())
    text_words = set(text)
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



# for idx, doc in relevant_docs:
#         print(f"Index ke-{idx}: {doc}")
#         # answer_text = data.loc[idx, 'Answer']  # Mendapatkan jawaban 


def get_answer(relevant_docs, cleaned_user_input):
    # Menggabungkan clean_token[0] menjadi satu teks tanpa tokenisasi
    target_text = ' '.join(cleaned_user_input)

    # # Menampilkan indeks dari relevant_docs
    # for idx, (index, doc) in enumerate(relevant_docs):
    #     print(f"Index ke-{index}: {' '.join(doc)}")

    # Menggunakan relevant_docs sebagai result_texts
    most_similar_text, most_similar_index = find_most_similar_text_with_index(target_text, relevant_docs)
    if most_similar_text != "":
        # print("Teks yang paling mirip:", most_similar_text)
        # print("Index dari teks yang paling mirip:", most_similar_index)
        answer_text = data.loc[most_similar_index, 'Answer']  # Mendapatkan jawaban berdasarkan indeks
        # print("JAWABAN")
        return print(answer_text)
    else:
        return print("Tidak ada teks yang cocok atau hasil kosong.")

    # if user_pattern:
    #     for index, question in enumerate(candidate_questions):
    #         candidate_pattern, candidate_match = extract_pattern(' '.join(question))
    #         if candidate_pattern == user_pattern:
    #             question_text = ' '.join(question)
    #             keyword_length = len(candidate_pattern.split()) - 1
    #             words_after_keyword = question_text.split()[keyword_length:]
    #             result_text = ' '.join(words_after_keyword[1:]) if len(words_after_keyword) > 1 else ''
    #             result_texts.append((previous_indices[index], result_text))

    # else:
    #     return "Tidak ada pola yang cocok dengan user_input."

    # target_text = ' '.join(cleaned_sentences[0])
    # most_similar_text, most_similar_index = find_most_similar_text_with_index(target_text, result_texts)
    
    # if most_similar_text != "":
    #     answer_text = data.loc[most_similar_index-1, 'Answer']
    #     return answer_text

    # return "Tidak ada teks yang cocok atau hasil kosong."

    # hasil_preproses = tokenisasi(cleaned_sentences)

        # for i in cleaned_sentences:
        #     print(i)

 # Ganti 'user_input' sesuai dengan nama input dalam formulir
# response = get_answer(cleaned_sentences, cosine_similarity_results, extract_pattern)
# print(response)


@app.route('/')
def index():
    return render_template('index.html')  # Ganti 'index.html' dengan nama file template HTML kamu

@app.route('/hasil', methods=['POST'])
def hasil():
    if request.method == 'POST':
        user_input = request.form['user_input']
        cleaned_user_input = preprocess_and_tokenize(user_input, stopwords=True, words_to_exclude=excluded_words)
        cleaned_sentences = data['Question'].apply(lambda x: preprocess_and_tokenize(x, stopwords=True, words_to_exclude=excluded_words)).tolist()
        # cleaned_sentences.insert(0, cleaned_user_input)
        # cosine_similarity_results = calculate_cosine_similarity(cleaned_sentences)
        # for result in cosine_similarity_results:
        #     print(f"Isi Dokumen {result['idx']}: {result['text']}")
        #     print(f"Nilai Cosine Similarity: {result['sim']}")
 
        relevant_docs = rule_based(cleaned_sentences, cleaned_user_input) 
        # Membersihkan seluruh daftar kalimat dalam kolom 'Question'
        response = get_answer(relevant_docs, cleaned_user_input)
        return render_template('index.html', user_input=user_input, response=response)  # Ganti 'hasil.html' dengan nama file template HTML untuk hasil

if __name__ == '__main__':
    app.run(debug=True)