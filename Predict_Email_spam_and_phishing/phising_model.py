from collections import defaultdict
import pandas as pd
import re
import numpy as np

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text

def remove_special_characters(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def remove_numbers(text):
    text = re.sub(r'\d+', '', text)
    return text

def remove_extra_spaces(text):
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_short_text(text, min_length=10):
    return text if len(text) >= min_length else ''

def remove_emojis(text):
    emoji_pattern = re.compile("[\U00010000-\U0010ffff]", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def preprocessing(msg):
    if not isinstance(msg, str):
        msg = str(msg)
    msg = clean_text(msg)
    msg = remove_special_characters(msg)
    msg = remove_numbers(msg)
    msg = remove_extra_spaces(msg)
    msg = remove_short_text(msg)
    msg = remove_emojis(msg)
    return msg

def load_data():
    data2 = pd.read_csv('Phishing_Email.csv', encoding='utf-8', on_bad_lines='skip', engine='python')

    data2['Message'] = data2['Email Text'].apply(preprocessing)

    data2['Phishing'] = data2['Email Type'].map({'Phishing Email': 1, 'Safe Email': 0}).fillna(-1)

    data2 = data2.drop(['Email Text', 'Email Type', 'Unnamed: 0'], axis=1)
    return data2

def train_model(data2):
    print(data2)
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

    vocabulary = set(" ".join(data2['Message']).split())
    vocab_size = len(vocabulary)

    def calculate_word_probabilities(word_counts, total_words, vocab_size):
        probabilities = {}
        for word in vocabulary:
            probabilities[word] = (word_counts[word] + 1) / (total_words + vocab_size) 
        return probabilities

    phishing_word_probs = calculate_word_probabilities(phishing_word_counts, total_phishing_words, vocab_size)
    safe_word_probs = calculate_word_probabilities(safe_word_counts, total_safe_words, vocab_size)

    p_phishing_train = len(phishing_messages) / len(data2)
    p_safe_train = len(safe_messages) / len(data2)

    return phishing_word_probs, safe_word_probs, p_phishing_train, p_safe_train, vocabulary, total_phishing_words, total_safe_words, vocab_size

def predict_message(message, phishing_word_probs, safe_word_probs, p_phishing_train, p_safe_train, vocabulary, total_phishing_words, total_safe_words, vocab_size):
    message = preprocessing(message) 
    words = message.split() 

    phishing_prob = np.log(p_phishing_train)
    safe_prob = np.log(p_safe_train)

    for word in words:
        phishing_prob += np.log(phishing_word_probs.get(word, 1 / (total_phishing_words + vocab_size)))
        safe_prob += np.log(safe_word_probs.get(word, 1 / (total_safe_words + vocab_size)))

    return 1 if phishing_prob > safe_prob else 0  
