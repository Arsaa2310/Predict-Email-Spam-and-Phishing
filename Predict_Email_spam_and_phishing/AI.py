import streamlit as st
import imaplib
import email
from email.header import decode_header
from langdetect import detect, LangDetectException
from phising_model import load_data as load_phishing_data, train_model as train_phishing_model, predict_message as predict_phishing_message
from spam_model import load_data as load_spam_data, train_model as train_spam_model, predict_message as predict_spam_message

# LangDetect Configuration
from langdetect import DetectorFactory
DetectorFactory.seed = 0

# Load Models
st.sidebar.header("Model Loading")
phishing_data = load_phishing_data()
phishing_word_probs, phishing_safe_word_probs, p_phishing_train, p_safe_train, phishing_vocabulary, total_phishing_words, total_safe_words, phishing_vocab_size = train_phishing_model(phishing_data)

spam_data = load_spam_data()
spam_word_probs, spam_ham_word_probs, p_spam_train, p_ham_train, spam_vocabulary, total_spam_words, total_ham_words, spam_vocab_size = train_spam_model(spam_data)

# Streamlit UI
st.title("Email Classification: Phishing and Spam Detection")
st.sidebar.title("Navigation")

# Login Section
st.header("Login to Your Email")
email_address = st.text_input("Email Address", placeholder="Enter your email")
password = st.text_input("Password", placeholder="Enter your password", type="password")

if st.button("Login"):
    if email_address and password:
        try:
            mail = imaplib.IMAP4_SSL("imap.gmail.com")
            mail.login(email_address, password)
            st.success("Login successful!")
            mail.select("inbox")

            # Fetch Emails
            status, messages = mail.search(None, "ALL")
            email_ids = messages[0].split()
            emails = []

            for email_id in email_ids[-25:]:  # Get the latest 25 emails
                res, msg = mail.fetch(email_id, "(RFC822)")
                for response in msg:
                    if isinstance(response, tuple):
                        msg = email.message_from_bytes(response[1])
                        subject, encoding = decode_header(msg["Subject"])[0]

                        if isinstance(subject, bytes):
                            try:
                                subject = subject.decode(encoding if encoding else "utf-8")
                            except (UnicodeDecodeError, LookupError) as e:
                                subject = subject.decode("utf-8", errors="replace")
                                continue

                        try:
                            language = detect(subject)
                            if language == "en":
                                phishing_label = predict_phishing_message(
                                    subject,
                                    phishing_word_probs,
                                    phishing_safe_word_probs,
                                    p_phishing_train,
                                    p_safe_train,
                                    phishing_vocabulary,
                                    total_phishing_words,
                                    total_safe_words,
                                    phishing_vocab_size,
                                )
                                spam_label = predict_spam_message(
                                    subject,
                                    spam_word_probs,
                                    spam_ham_word_probs,
                                    p_spam_train,
                                    p_ham_train,
                                    spam_vocabulary,
                                    total_spam_words,
                                    total_ham_words,
                                    spam_vocab_size,
                                )

                                emails.append({
                                    "subject": subject,
                                    "language": language,
                                    "phishing_label": "Phishing" if phishing_label == 1 else "Safe",
                                    "spam_label": "Spam" if spam_label == 1 else "Ham",
                                })
                        except LangDetectException:
                            continue

            mail.close()
            mail.logout()

            # Display Emails
            st.subheader("Inbox")
            for email_data in emails:
                st.write(f"**Subject:** {email_data['subject']}")
                st.write(f"**Language:** {email_data['language']}")
                st.write(f"**Phishing Detection:** {email_data['phishing_label']}")
                st.write(f"**Spam Detection:** {email_data['spam_label']}")
                st.markdown("---")
        except imaplib.IMAP4.error:
            st.error("Invalid email or password. Please try again.")
    else:
        st.warning("Please enter both email and password.")
