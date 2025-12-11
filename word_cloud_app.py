import streamlit as st
import pypdf
import nltk
import re
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from wordcloud import WordCloud

# Download NLTK resources (run once)
nltk.download('punkt')
nltk.download('stopwords')

# App UI
st.title("📘 PDF Word Analyzer")
st.write("Upload a PDF file to extract text, list frequent words, and view a word cloud.")

# File uploader
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

# Only run processing if PDF is uploaded
if uploaded_file is not None:
    
    # Read PDF
    reader = pypdf.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    
    # Check if text exists
    if not text.strip():
        st.error("No extractable text found in the PDF.")
        st.stop()
    
    # Preprocessing
    def preprocess_text(t):
        t = t.lower()
        t = re.sub(r"[^a-zA-Z\s]", " ", t)
        t = re.sub(r"\s+", " ", t)
        return t
    
    clean_text = preprocess_text(text)

    # Tokenize
    words = word_tokenize(clean_text, preserve_line=True)   # <--- FIX


    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    words = [w for w in words if w not in stop_words and len(w) > 2 and w.isalnum()]

    # Check if ANY words exist
    if len(words) == 0:
        st.error("No valid words found to generate a word cloud.")
        st.stop()

    # Frequency distribution
    freq = FreqDist(words)
    st.subheader("📊 Top 20 Frequent Words")
    st.write(freq.most_common(20))

    # Generate word cloud
    wordcloud = WordCloud(
        width=1000,
        height=500,
        stopwords=stop_words,
        colormap="plasma",
        max_words=200,
        background_color="white"
    ).generate(" ".join(words))

    # Display word cloud
    st.subheader("☁ Word Cloud")
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)

else:
    st.info("Please upload a PDF file to proceed.")
