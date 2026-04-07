import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Page config
st.set_page_config(page_title="AI Summarizer", layout="centered")

# Title
st.markdown("<h1 style='text-align: center;'>🧠 AI Text Summarizer</h1>", unsafe_allow_html=True)

model_name = "google/flan-t5-small"

# Load model (cached)
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        device_map="cpu"
    )
    return tokenizer, model

tokenizer, model = load_model()

# Text input
text = st.text_area("Enter your text here:")

# File upload
uploaded_file = st.file_uploader("Or upload a .txt file", type=["txt"])

file_text = ""
if uploaded_file is not None:
    file_text = uploaded_file.read().decode("utf-8")

# Summary length selection
length = st.selectbox("Select summary length", ["Short", "Medium", "Long"])

if length == "Short":
    max_len = 50
    min_len = 20
elif length == "Medium":
    max_len = 100
    min_len = 40
else:
    max_len = 150
    min_len = 60

# Summarize button
if st.button("Summarize"):

    # Decide input source
    final_text = text if text else file_text

    if final_text:
        input_text = "Summarize this text clearly:\n" + final_text

        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )

        outputs = model.generate(
            inputs["input_ids"],
            max_length=max_len,
            min_length=min_len,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True
        )

        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Display summary
        st.subheader("Summary:")
        st.write(summary)

        # Download button
        st.download_button(
            label="Download Summary",
            data=summary,
            file_name="summary.txt",
            mime="text/plain"
        )

    else:
        st.warning("Please enter text or upload a file")