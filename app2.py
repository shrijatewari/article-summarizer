# app.py
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import yake
import fitz  # PyMuPDF

# --------------------
# Model Config
# --------------------
DEFAULT_MODEL = "sshleifer/distilbart-cnn-12-6"  # Fast, no sentencepiece

# --------------------
# Summarizer Loader
# --------------------
@st.cache_resource(show_spinner=False)
def load_summarizer(model_name: str = DEFAULT_MODEL):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=-1)
        return summarizer
    except Exception as e:
        st.warning(f"Could not load summarizer model {model_name}: {e}")
        return None

# --------------------
# Simple Extractive Fallback
# --------------------
def simple_extractive_fallback(text, max_sentences=3):
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if not sentences:
        return ""
    sentences_sorted = sorted(sentences, key=lambda s: len(s), reverse=True)
    return " ".join(sentences_sorted[:max_sentences])

# --------------------
# Summarization
# --------------------
def summarize_text(text: str,
                   summarizer=None,
                   min_length: int = 30,
                   max_length: int = 130):
    if not text.strip():
        return ""
    if summarizer is None:
        summarizer = load_summarizer()
    if summarizer:
        try:
            result = summarizer(
                text,
                truncation=True,
                min_length=min_length,
                max_length=max_length,
                do_sample=False
            )
            return result[0]["summary_text"].strip()
        except Exception as e:
            st.warning(f"Transformer summarizer failed: {e}")
    return simple_extractive_fallback(text)

# --------------------
# Chunking for Long Texts
# --------------------
def chunk_and_summarize(text, summarizer, chunk_size=800, overlap=50):
    summaries = []
    start = 0
    while start < len(text):
        chunk = text[start:start+chunk_size]
        s = summarize_text(chunk, summarizer=summarizer)
        summaries.append(s)
        start += chunk_size - overlap
    combined = " ".join(summaries)
    return summarize_text(combined, summarizer=summarizer, max_length=150)

# --------------------
# Keyword-based Question Generator
# --------------------
def generate_questions(text, num_questions=5):
    kw_extractor = yake.KeywordExtractor(lan="en", n=2, top=num_questions)
    keywords = [kw for kw, score in kw_extractor.extract_keywords(text)]
    questions = [f"What is {kw}?" for kw in keywords]
    return questions

# --------------------
# PDF Text Extractor
# --------------------
def extract_text_from_pdf(uploaded_file):
    pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in pdf_document:
        text += page.get_text()
    return text.strip()

# --------------------
# Streamlit UI
# --------------------
st.title("Article Summarizer & Q&A")

uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"])
user_input = st.text_area("...or paste your article here:")

if uploaded_pdf:
    with st.spinner("Extracting text from PDF..."):
        user_input = extract_text_from_pdf(uploaded_pdf)
        st.success(f"Extracted {len(user_input)} characters from PDF.")

summarizer = load_summarizer()

if st.button("Summarize"):
    if not user_input.strip():
        st.warning("Please enter or upload text first.")
    else:
        with st.spinner("Generating summary..."):
            if len(user_input) > 1000:
                summary = chunk_and_summarize(user_input, summarizer)
            else:
                summary = summarize_text(user_input, summarizer=summarizer)
        st.subheader("Summary")
        st.write(summary)

if st.button("Generate Questions"):
    if not user_input.strip():
        st.warning("Please enter or upload text first.")
    else:
        with st.spinner("Generating questions..."):
            qs = generate_questions(user_input)
        st.subheader("Questions")
        for q in qs:
            st.write("• " + q)
