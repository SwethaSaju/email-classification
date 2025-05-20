import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Download NLTK resources (only the first time)
nltk.download('punkt')
nltk.download('stopwords')

# Sample training data (replace this with saved model/vectorizer for production)
sample_emails = [
    "You won a free ticket to Bahamas!",
    "Meeting at 3pm regarding Q3 report.",
    "Claim your prize now!",
    "Let's catch up over coffee tomorrow."
]
labels = ['spam', 'primary', 'spam', 'primary']

df = pd.DataFrame({'text': sample_emails, 'label': labels})
stop_words = set(stopwords.words('english'))

def clean_text(text):
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalnum() and word not in stop_words]
    return ' '.join(words)

df['cleaned'] = df['text'].apply(clean_text)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['cleaned'])
y = df['label']

model = MultinomialNB()
model.fit(X, y)

# Streamlit UI
st.title("📧 Email Spam Classifier")

email_input = st.text_area("Paste your email content here:")

if st.button("Classify"):
    if not email_input.strip():
        st.warning("Please enter email content.")
    else:
        cleaned_input = clean_text(email_input)
        vector = vectorizer.transform([cleaned_input])
        prediction = model.predict(vector)[0]

        if prediction == 'spam':
            st.error("🚫 This email is SPAM.")
        else:
            st.success("✅ This email is NOT SPAM.")
