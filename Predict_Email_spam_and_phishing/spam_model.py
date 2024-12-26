import numpy as np
import pandas as pd
import re
import os
from collections import defaultdict
import streamlit as st

st.write("Current Working Directory:", os.getcwd())

def load_data():
    d1 = pd.read_csv(r"/mount/src/predict-email-spam-and-phishing/mail_data.csv")
    st.write(d1)
    return d1

def calculate_word_probabilities(word_counts, total_words, vocab_size):
    vocabulary = set(word_counts.keys())
    probabilities = {}
    for word in vocabulary:
        probabilities[word] = (word_counts[word] + 1) / (total_words + vocab_size) 
    return probabilities

def train_model(d1):
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

    vocabulary = set(" ".join(d1['Message']).split())
    vocab_size = len(vocabulary)

    spam_word_probs = calculate_word_probabilities(spam_word_counts, total_spam_words, vocab_size)
    ham_word_probs = calculate_word_probabilities(ham_word_counts, total_ham_words, vocab_size)

    p_spam_train = len(spam_messages) / len(d1)
    p_ham_train = len(ham_messages) / len(d1)

    return spam_word_probs, ham_word_probs, p_spam_train, p_ham_train, vocabulary, total_spam_words, total_ham_words, vocab_size

def preprocess_message(message):
    message = message.lower()
    message = re.sub(r'[^a-zA-Z\s]', '', message)
    message = re.sub(r'\s+', ' ', message).strip()
    return message

def predict_message(message, spam_word_probs, ham_word_probs, p_spam_train, p_ham_train, vocabulary, total_spam_words, total_ham_words, vocab_size):
    message = preprocess_message(message)
    words = message.split()

    spam_prob = np.log(p_spam_train)
    ham_prob = np.log(p_ham_train)

    for word in words:
        spam_prob += np.log(spam_word_probs.get(word, 1 / (total_spam_words + vocab_size)))
        ham_prob += np.log(ham_word_probs.get(word, 1 / (total_ham_words + vocab_size)))

    return 1 if spam_prob > ham_prob else 0 
