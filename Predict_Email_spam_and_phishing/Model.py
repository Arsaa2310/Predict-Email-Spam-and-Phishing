import pandas as pd
import numpy as np
import re
from collections import defaultdict

class BaseModel:
    def preprocess_message(self, message):
        if not isinstance(message, str):
            message = str(message)
        message = message.lower()
        message = re.sub(r'[^a-zA-Z\s]', '', message)
        message = re.sub(r'\s+', ' ', message).strip()
        return message

    def calculate_word_probabilities(self, word_counts, total_words, vocab_size):
        probabilities = {}
        for word in self.vocabulary:
            probabilities[word] = (word_counts[word] + 1) / (total_words + vocab_size)
        return probabilities

class PhishingModel(BaseModel):
    def __init__(self):
        self.phishing_word_probs = None
        self.safe_word_probs = None
        self.p_phishing_train = None
        self.p_safe_train = None
        self.vocabulary = set()
        self.total_phishing_words = 0
        self.total_safe_words = 0
        self.vocab_size = 0

    def load_data(self):
        # Sample data for demonstration - replace with your actual data path
        data = {
            'Email Text': ['Free money click here', 'Meeting tomorrow at 2pm', 'Verify your account now'],
            'Email Type': ['Phishing Email', 'Safe Email', 'Phishing Email']
        }
        data = pd.DataFrame(data)
        data['Message'] = data['Email Text'].apply(self.preprocess_message)
        data['Phishing'] = data['Email Type'].map({'Phishing Email': 1, 'Safe Email': 0})
        return data

    def train_model(self, data):
        phishing_messages = data[data['Phishing'] == 1]['Message']
        safe_messages = data[data['Phishing'] == 0]['Message']

        phishing_word_counts = defaultdict(int)
        safe_word_counts = defaultdict(int)

        self.total_phishing_words = 0
        self.total_safe_words = 0

        for message in phishing_messages:
            for word in message.split():
                phishing_word_counts[word] += 1
                self.total_phishing_words += 1

        for message in safe_messages:
            for word in message.split():
                safe_word_counts[word] += 1
                self.total_safe_words += 1

        self.vocabulary = set(" ".join(data['Message']).split())
        self.vocab_size = len(self.vocabulary)

        self.phishing_word_probs = self.calculate_word_probabilities(phishing_word_counts, self.total_phishing_words, self.vocab_size)
        self.safe_word_probs = self.calculate_word_probabilities(safe_word_counts, self.total_safe_words, self.vocab_size)

        self.p_phishing_train = len(phishing_messages) / len(data)
        self.p_safe_train = len(safe_messages) / len(data)

    def predict_message(self, message):
        message = self.preprocess_message(message)
        words = message.split()

        phishing_prob = np.log(self.p_phishing_train)
        safe_prob = np.log(self.p_safe_train)

        for word in words:
            if word in self.vocabulary:
                phishing_prob += np.log(self.phishing_word_probs.get(word, 1 / (self.total_phishing_words + self.vocab_size)))
                safe_prob += np.log(self.safe_word_probs.get(word, 1 / (self.total_safe_words + self.vocab_size)))

        return 1 if phishing_prob > safe_prob else 0

class SpamModel(BaseModel):
    def __init__(self):
        self.spam_word_probs = None
        self.ham_word_probs = None
        self.p_spam_train = None
        self.p_ham_train = None
        self.vocabulary = set()
        self.total_spam_words = 0
        self.total_ham_words = 0
        self.vocab_size = 0

    def load_data(self):
        # Sample data for demonstration - replace with your actual data path
        data = {
            'Message': ['Buy now limited offer', 'Team meeting notes', 'Win free iPhone'],
            'Category': [1, 0, 1]  # 1 for spam, 0 for ham
        }
        return pd.DataFrame(data)

    def train_model(self, data):
        data['Message'] = data['Message'].fillna("").astype(str)

        spam_messages = data[data['Category'] == 1]['Message']
        ham_messages = data[data['Category'] == 0]['Message']

        spam_word_counts = defaultdict(int)
        ham_word_counts = defaultdict(int)

        self.total_spam_words = 0
        self.total_ham_words = 0

        for message in spam_messages:
            for word in message.split():
                spam_word_counts[word] += 1
                self.total_spam_words += 1

        for message in ham_messages:
            for word in message.split():
                ham_word_counts[word] += 1
                self.total_ham_words += 1

        self.vocabulary = set(" ".join(data['Message']).split())
        self.vocab_size = len(self.vocabulary)

        self.spam_word_probs = self.calculate_word_probabilities(spam_word_counts, self.total_spam_words, self.vocab_size)
        self.ham_word_probs = self.calculate_word_probabilities(ham_word_counts, self.total_ham_words, self.vocab_size)

        self.p_spam_train = len(spam_messages) / len(data)
        self.p_ham_train = len(ham_messages) / len(data)

    def predict_message(self, message):
        message = self.preprocess_message(message)
        words = message.split()

        spam_prob = np.log(self.p_spam_train)
        ham_prob = np.log(self.p_ham_train)

        for word in words:
            if word in self.vocabulary:
                spam_prob += np.log(self.spam_word_probs.get(word, 1 / (self.total_spam_words + self.vocab_size)))
                ham_prob += np.log(self.ham_word_probs.get(word, 1 / (self.total_ham_words + self.vocab_size)))

        return 1 if spam_prob > ham_prob else 0