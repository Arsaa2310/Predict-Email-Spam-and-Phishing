import streamlit as st
import imaplib
import email
from email.header import decode_header
import pandas as pd
from Model import PhishingModel, SpamModel

# Initialize models
phishing_model = PhishingModel()
phishing_data = phishing_model.load_data()
phishing_model.train_model(phishing_data)

spam_model = SpamModel()
spam_data = spam_model.load_data()
spam_model.train_model(spam_data)

# Streamlit UI
st.title("Email Classification: Phishing and Spam Detection")

# Login Section
st.header("Login to Your Email")
email_address = st.text_input("Email Address", placeholder="Enter your email")
password = st.text_input("Password", placeholder="Enter your password", type="password")

if st.button("Login"):
    if email_address and password:
        try:
            # Connect to Gmail
            mail = imaplib.IMAP4_SSL("imap.gmail.com")
            mail.login(email_address, password)
            st.success("Login successful!")
            
            # Select inbox
            mail.select("inbox")
            
            # Search for emails
            _, messages = mail.search(None, "ALL")
            email_ids = messages[0].split()
            
            # Create a list to store email data
            email_data = []
            
            # Get the last 10 emails
            for email_id in email_ids[-10:]:
                _, msg = mail.fetch(email_id, "(RFC822)")
                email_message = email.message_from_bytes(msg[0][1])
                
                # Get subject
                subject, encoding = decode_header(email_message["Subject"])[0]
                if isinstance(subject, bytes):
                    subject = subject.decode(encoding if encoding else 'utf-8', errors='replace')
                
                # Get body
                if email_message.is_multipart():
                    body = ''
                    for part in email_message.walk():
                        if part.get_content_type() == "text/plain":
                            try:
                                body = part.get_payload(decode=True).decode()
                                break
                            except:
                                continue
                else:
                    try:
                        body = email_message.get_payload(decode=True).decode()
                    except:
                        body = "Could not decode email body"
                
                # Make predictions
                try:
                    phishing_prediction = phishing_model.predict_message(body)
                    spam_prediction = spam_model.predict_message(body)
                except Exception as e:
                    st.error(f"Error in prediction: {str(e)}")
                    phishing_prediction = None
                    spam_prediction = None
                
                # Add to email data
                email_data.append({
                    'Subject': subject,
                    'Body': body[:200] + "...",  # Show first 200 characters
                    'Phishing': "Phishing" if phishing_prediction == 1 else "Safe",
                    'Spam': "Spam" if spam_prediction == 1 else "Ham"
                })
            
            # Create and display DataFrame
            df = pd.DataFrame(email_data)
            st.dataframe(df)
            
            # Logout
            mail.close()
            mail.logout()
            
        except imaplib.IMAP4.error as e:
            st.error(f"Login failed: {str(e)}")
    else:
        st.warning("Please enter both email and password.")