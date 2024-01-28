import math
from collections import Counter
import pandas as pd
import re

data = pd.read_csv('Dataset Original.csv')
class TextAnalyzer:
    # def calculate_cosine_similarity(self, hasil_tokenisasi):
    #     tf_scores = []
    #     for sentence in hasil_tokenisasi:
    #         word_count = Counter(sentence)
    #         total_words = len(sentence)
    #         tf = {word: count / total_words for word, count in word_count.items()}
    #         tf_scores.append(tf)

    #     tf_data = {}
    #     for idx, tf in enumerate(tf_scores):
    #         tf_data[f"TF Dokumen {idx+1}"] = tf

    #     tf_df = pd.DataFrame(tf_data).fillna(0)

    #     doc_frequency = {}
    #     for sentence in hasil_tokenisasi:
    #         unique_words = set(sentence)
    #         for word in unique_words:
    #             doc_frequency[word] = doc_frequency.get(word, 0) + 1

    #     num_documents = len(hasil_tokenisasi)
    #     idf_scores = {}
    #     for word, freq in doc_frequency.items():
    #         idf = math.log(num_documents / (freq + 1))
    #         idf_scores[word] = idf

    #     tfidf_scores = []
    #     for tf in tf_scores:
    #         tfidf = {word: tf[word] * idf_scores[word] for word in tf}
    #         tfidf_scores.append(tfidf)

    #     tfidf_data = {}
    #     for idx, tfidf in enumerate(tfidf_scores):
    #         tfidf_data[f"TF-IDF Dokumen {idx+1}"] = tfidf

    #     tfidf_df = pd.DataFrame(tfidf_data).fillna(0)

    #     base_document = tfidf_df.iloc[:, 0:1]
    #     other_documents = tfidf_df.iloc[:, 1:]

    #     similarities = []
    #     for idx, column in other_documents.items():
    #         dot_product = (base_document.values.flatten() * column.values).sum()
    #         base_norm = math.sqrt((base_document ** 2).sum().values[0])
    #         other_norm = math.sqrt((column ** 2).sum())
    #         similarity = dot_product / (base_norm * other_norm) if (base_norm * other_norm) != 0 else 0
    #         similarities.append(similarity)

    #     similarity_pairs = list(zip(range(1, len(similarities) + 1), similarities))
    #     sorted_pairs = sorted(similarity_pairs, key=lambda x: x[1], reverse=True)
    #     top_5_similar = sorted_pairs[:5]

    #     return top_5_similar

    # Fungsi lain yang ada pada kode bisa juga dimasukkan ke dalam kelas ini
    # ...
        # def extract_pattern(self, sentence):
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
    #         match = re.search(pattern, sentence)
    #         if match:
    #             return pattern_name, match.group(0)

    #     return None, None

    def rule_based(self, cleaned_sentences, cleaned_user_input):
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



    def count_similar_words(self, target_text, text):
        target_words = set(target_text.split())
        text_words = set(text)
        similar_words = target_words.intersection(text_words)
        return len(similar_words)

    def find_most_similar_text_with_index(self, target_text, result_texts):
        max_similarity = 0
        most_similar_text = ""
        most_similar_index = -1

        for index, text in result_texts:
            similarity_count = self.count_similar_words(target_text, text)
            if similarity_count > max_similarity:
                max_similarity = similarity_count
                most_similar_text = text
                most_similar_index = index
        return most_similar_text, most_similar_index


    def get_answer(self,relevant_docs, cleaned_user_input):
        # Menggabungkan clean_token[0] menjadi satu teks tanpa tokenisasi
        target_text = ' '.join(cleaned_user_input)

        # # Menampilkan indeks dari relevant_docs
        # for idx, (index, doc) in enumerate(relevant_docs):
        #     print(f"Index ke-{index}: {' '.join(doc)}")

        # Menggunakan relevant_docs sebagai result_texts
        most_similar_text, most_similar_index = self.find_most_similar_text_with_index(target_text, relevant_docs)
        if most_similar_text != "":
            # print("Teks yang paling mirip:", most_similar_text)
            # print("Index dari teks yang paling mirip:", most_similar_index)
            answer_text = data.loc[most_similar_index, 'Answer']  # Mendapatkan jawaban berdasarkan indeks
            # print("JAWABAN")
            return answer_text
        else:
            return "Mohon masukan kalimat pertanyaan."

