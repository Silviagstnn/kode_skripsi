import pandas as pd
from flask import Flask, render_template, request
from TextPreprocessor import TextPreprocessor
from TextAnalyzer import TextAnalyzer

app = Flask(__name__)
data = pd.read_csv('Dataset Original.csv')

class FlaskHandler:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.analyzer = TextAnalyzer()

    @app.route('/')
    def index():
        return render_template('index.html')  # Ganti 'index.html' dengan nama file template HTML kamu

    @app.route('/hasil', methods=['POST'])
    def hasil():
        handler = FlaskHandler()  # Buat instance FlaskHandler di sini
        if request.method == 'POST':
            user_input = request.form['user_input']
            cleaned_user_input = handler.preprocessor.preprocess_and_tokenize(user_input)
            cleaned_sentences = data['Question'].apply(lambda x: handler.preprocessor.preprocess_and_tokenize(x)).tolist()
            # cleaned_sentences.insert(0, cleaned_user_input)
            # cosine_similarity_results = handler.analyzer.calculate_cosine_similarity(cleaned_sentences)
            # Other necessary logic for response handling
            relevant_docs = handler.analyzer.rule_based(cleaned_sentences, cleaned_user_input) 
            # Membersihkan seluruh daftar kalimat dalam kolom 'Question'
            response = handler.analyzer.get_answer(relevant_docs, cleaned_user_input)
            return render_template('index.html', user_input=user_input, response=response) # Ganti 'hasil.html' dengan nama file template HTML untuk hasil

if __name__ == '__main__':
    app.run(debug=True)


