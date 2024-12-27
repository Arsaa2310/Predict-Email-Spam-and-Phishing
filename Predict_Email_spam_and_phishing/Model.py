import pandas as pd
import re
import numpy as np
from collections import defaultdict

class PhishingModel:
    def __init__(self, data_file=r'Phishing_Email.csv'):
        self.data_file = data_file
        self.phishing_word_probs = None
        self.safe_word_probs = None
        self.p_phishing_train = None
        self.p_safe_train = None
        self.vocabulary = None
        self.total_phishing_words = None
        self.total_safe_words = None
        self.vocab_size = None

    def clean_text(self, text):
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = text.lower()
        return text

    def remove_special_characters(self, text):
        return re.sub(r'[^a-zA-Z\s]', '', text)

    def remove_numbers(self, text):
        return re.sub(r'\d+', '', text)

    def remove_extra_spaces(self, text):
        return re.sub(r'\s+', ' ', text).strip()

    def remove_short_text(self, text, min_length=10):
        return text if len(text) >= min_length else ''

    def remove_emojis(self, text):
        emoji_pattern = re.compile("[\U00010000-\U0010ffff]", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)

    def preprocessing(self, msg):
        if not isinstance(msg, str):
            msg = str(msg)
        msg = self.clean_text(msg)
        msg = self.remove_special_characters(msg)
        msg = self.remove_numbers(msg)
        msg = self.remove_extra_spaces(msg)
        msg = self.remove_short_text(msg)
        msg = self.remove_emojis(msg)
        return msg

    def load_data(self):
        data2 = pd.read_csv(self.data_file, encoding='utf-8', on_bad_lines='skip', engine='python')
        data2['Message'] = data2['Email Text'].apply(self.preprocessing)
        data2['Phishing'] = data2['Email Type'].map({'Phishing Email': 1, 'Safe Email': 0}).fillna(-1)
        data2 = data2.drop(['Email Text', 'Email Type', 'Unnamed: 0'], axis=1)
        return data2

    def calculate_word_probabilities(self, word_counts, total_words, vocab_size):
        probabilities = {}
        for word in self.vocabulary:
            probabilities[word] = (word_counts[word] + 1) / (total_words + vocab_size)
        return probabilities

    def train_model(self, data2):
        phishing_messages = data2[data2['Phishing'] == 1]['Message']
        safe_messages = data2[data2['Phishing'] == 0]['Message']

        phishing_word_counts = defaultdict(int)
        safe_word_counts = defaultdict(int)

        total_phishing_words = 0
        total_safe_words = 0

        for message in phishing_messages:
            for word in message.split():
                phishing_word_counts[word] += 1
                total_phishing_words += 1

        for message in safe_messages:
            for word in message.split():
                safe_word_counts[word] += 1
                total_safe_words += 1

        self.vocabulary = set(" ".join(data2['Message']).split())
        self.vocab_size = len(self.vocabulary)

        self.phishing_word_probs = self.calculate_word_probabilities(phishing_word_counts, total_phishing_words, self.vocab_size)
        self.safe_word_probs = self.calculate_word_probabilities(safe_word_counts, total_safe_words, self.vocab_size)

        self.p_phishing_train = len(phishing_messages) / len(data2)
        self.p_safe_train = len(safe_messages) / len(data2)

        return (self.phishing_word_probs, self.safe_word_probs, self.p_phishing_train, self.p_safe_train, self.vocabulary, total_phishing_words, total_safe_words, self.vocab_size)

    def predict_message(self, message):
        message = self.preprocessing(message)
        words = message.split()

        phishing_prob = np.log(self.p_phishing_train)
        safe_prob = np.log(self.p_safe_train)

        for word in words:
            phishing_prob += np.log(self.phishing_word_probs.get(word, 1 / (self.total_phishing_words + self.vocab_size)))
            safe_prob += np.log(self.safe_word_probs.get(word, 1 / (self.total_safe_words + self.vocab_size)))

        return 1 if phishing_prob > safe_prob else 0


class SpamModel:
    def __init__(self, data_file=r'mail_data.csv'):
        self.data_file = data_file
        self.spam_word_probs = None
        self.ham_word_probs = None
        self.p_spam_train = None
        self.p_ham_train = None
        self.vocabulary = None
        self.total_spam_words = None
        self.total_ham_words = None
        self.vocab_size = None

    def preprocess_message(self, message):
        message = message.lower()
        message = re.sub(r'[^a-zA-Z\s]', '', message)
        message = re.sub(r'\s+', ' ', message).strip()
        return message

    def calculate_word_probabilities(self, word_counts, total_words, vocab_size):
        probabilities = {}
        for word in self.vocabulary:
            probabilities[word] = (word_counts[word] + 1) / (total_words + vocab_size)
        return probabilities

    def load_data(self):
        d1 = pd.read_csv(self.data_file)
        return d1

    def train_model(self, d1):
        d1['Message'] = d1['Message'].fillna("").astype(str)

        spam_messages = d1[d1['Category'] == 1]['Message']
        ham_messages = d1[d1['Category'] == 0]['Message']

        spam_word_counts = defaultdict(int)
        ham_word_counts = defaultdict(int)

        total_spam_words = 0
        total_ham_words = 0

        for message in spam_messages:
            for word in message.split():
                spam_word_counts[word] += 1
                total_spam_words += 1

        for message in ham_messages:
            for word in message.split():
                ham_word_counts[word] += 1
                total_ham_words += 1

        self.vocabulary = set(" ".join(d1['Message']).split())
        self.vocab_size = len(self.vocabulary)

        self.spam_word_probs = self.calculate_word_probabilities(spam_word_counts, total_spam_words, self.vocab_size)
        self.ham_word_probs = self.calculate_word_probabilities(ham_word_counts, total_ham_words, self.vocab_size)

        self.p_spam_train = len(spam_messages) / len(d1)
        self.p_ham_train = len(ham_messages) / len(d1)

        return (self.spam_word_probs, self.ham_word_probs, self.p_spam_train, self.p_ham_train, self.vocabulary, total_spam_words, total_ham_words, self.vocab_size)

    def predict_message(self, message):
        message = self.preprocess_message(message)
        words = message.split()

        spam_prob = np.log(self.p_spam_train)
        ham_prob = np.log(self.p_ham_train)

        for word in words:
            spam_prob += np.log(self.spam_word_probs.get(word, 1 / (self.total_spam_words + self.vocab_size)))
            ham_prob += np.log(self.ham_word_probs.get(word, 1 / (self.total_ham_words + self.vocab_size)))

        return 1 if spam_prob > ham_prob else 0
