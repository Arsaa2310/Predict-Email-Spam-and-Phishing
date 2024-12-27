import streamlit as st
import imaplib
import email
from email.header import decode_header
from langdetect import detect, LangDetectException
from Model import PhishingModel, SpamModel

# LangDetect Configuration
from langdetect import DetectorFactory
DetectorFactory.seed = 0

# Load Models
st.sidebar.header("Model Loading")
phishing_model = PhishingModel()
phishing_data = phishing_model.load_data()
phishing_model.train_model(phishing_data)  # Training model and saving parameters inside class

spam_model = SpamModel()
spam_data = spam_model.load_data()
spam_model.train_model(spam_data)  # Training model and saving parameters inside class

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

            for email_id in email_ids[-10:]:  # Get the latest 10 emails
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

                        # Try to extract the email content (body)
                        body = ""
                        if msg.is_multipart():
                            for part in msg.walk():
                                content_type = part.get_content_type()
                                content_disposition = str(part.get("Content-Disposition"))

                                if "attachment" not in content_disposition:
                                    if content_type == "text/plain":
                                        body = part.get_payload(decode=True).decode("utf-8", errors="ignore")
                                    elif content_type == "text/html":
                                        body = part.get_payload(decode=True).decode("utf-8", errors="ignore")
                        else:
                            body = msg.get_payload(decode=True).decode("utf-8", errors="ignore")

                        try:
                            language = detect(subject)
                            if language == "en":
                                phishing_label = phishing_model.predict_message(body)
                                spam_label = spam_model.predict_message(body)

                                emails.append({
                                    "subject": subject,
                                    "body": body,
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
                st.write(f"**Body:** {email_data['body'][:500]}...")  # Displaying the first 500 chars of email body
                st.markdown("---")
        except imaplib.IMAP4.error:
            st.error("Invalid email or password. Please try again.")
    else:
        st.warning("Please enter both email and password.")
