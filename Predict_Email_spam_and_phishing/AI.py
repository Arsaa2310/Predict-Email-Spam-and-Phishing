from flask import Flask, render_template, request, redirect, url_for, session
import imaplib
import email
from email.header import decode_header
from langdetect import detect, LangDetectException
from phising_model import load_data as load_phishing_data, train_model as train_phishing_model, predict_message as predict_phishing_message
from spam_model import load_data as load_spam_data, train_model as train_spam_model, predict_message as predict_spam_message

app = Flask(__name__)
app.secret_key = "secret-key"

from langdetect import DetectorFactory
DetectorFactory.seed = 0

phishing_data = load_phishing_data()
phishing_word_probs, phishing_safe_word_probs, p_phishing_train, p_safe_train, phishing_vocabulary, total_phishing_words, total_safe_words, phishing_vocab_size = train_phishing_model(phishing_data)

spam_data = load_spam_data()
spam_word_probs, spam_ham_word_probs, p_spam_train, p_ham_train, spam_vocabulary, total_spam_words, total_ham_words, spam_vocab_size = train_spam_model(spam_data)

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email_address = request.form["email"]
        password = request.form["password"]

        # Save credentials to session
        session["email"] = email_address
        session["password"] = password

        return redirect(url_for("inbox"))
    return render_template("login.html")

@app.route("/inbox")
def inbox():
    email_address = session.get("email")
    password = session.get("password")

    if not email_address or not password:
        return redirect(url_for("login"))

    try:
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(email_address, password)

        mail.select("inbox")
        status, messages = mail.search(None, "ALL")

        email_ids = messages[0].split()
        emails = []

        for email_id in email_ids[-25:]:
            res, msg = mail.fetch(email_id, "(RFC822)")

            for response in msg:
                if isinstance(response, tuple):
                    msg = email.message_from_bytes(response[1])
                    subject, encoding = decode_header(msg["Subject"])[0]
                    if isinstance(subject, bytes):
                        try:
                            # Decode subject with encoding or fallback to 'utf-8'
                            subject = subject.decode(encoding if encoding else "utf-8")
                        except (UnicodeDecodeError, LookupError) as e:
                            # Handle decoding error, fallback to 'utf-8' or replace invalid characters
                            subject = subject.decode("utf-8", errors="replace")
                            print(f"Error decoding subject: {e}")
                            continue

                    try:
                        language = detect(subject)
                        if language == "en":
                            phishing_label = predict_phishing_message(subject, phishing_word_probs, phishing_safe_word_probs, p_phishing_train, p_safe_train, phishing_vocabulary, total_phishing_words, total_safe_words, phishing_vocab_size)
                            spam_label = predict_spam_message(subject, spam_word_probs, spam_ham_word_probs, p_spam_train, p_ham_train, spam_vocabulary, total_spam_words, total_ham_words, spam_vocab_size)

                            emails.append({
                                "subject": subject,
                                "language": language,
                                "phishing_label": "Phishing" if phishing_label == 1 else "Safe",
                                "spam_label": "Spam" if spam_label == 1 else "Ham"
                            })
                    except LangDetectException:
                        continue

        mail.close()
        mail.logout()

        return render_template("inbox.html", emails=emails)
    except imaplib.IMAP4.error:
        return "Invalid email or password. Please try again."

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

if __name__ == "__main__":
    app.run(debug=True)
