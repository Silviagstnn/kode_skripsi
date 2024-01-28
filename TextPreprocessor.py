import re
import nltk
from nltk.corpus import stopwords

class TextPreprocessor:
    
    def __init__(self):
        self.sw = stopwords.words('indonesian')
        self.excluded_words = ['apa', 'mengapa', 'kapan', 'dimana', 'siapa', 'bagaimana', 'berapa', 'kenapa']

    def preprocess_and_tokenize(self, sentence, stopwords=True):
        cleaned_sentence = []
        words = sentence.lower().split()

        for word in words:
            if word in self.excluded_words:
                cleaned_sentence.append(word)
            elif stopwords and word not in self.sw:
                cleaned_sentence.append(word)

        cleaned_data = ' '.join(cleaned_sentence)
        cleaned_data = re.sub(r"[^\w\s]", "", cleaned_data)
        cleaned_data = re.sub(r"[\d]", "", cleaned_data)

        tokenized_list = cleaned_data.split()

        return tokenized_list
