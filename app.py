# First: install necessary packages if not already installed
import subprocess
import sys

required_packages = ["streamlit", "transformers", "torch"]

for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Now import and run the app
import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load model and tokenizer
@st.cache_resource
def load_model_and_tokenizer(model_path="./fine_tuned_poetry_model_v3"):
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    special_tokens = {
        'pad_token': '<|pad|>',
        'bos_token': '<|startofpoem|>',
        'eos_token': '<|endofpoem|>'
    }
    tokenizer.add_special_tokens(special_tokens)

    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# App UI
st.title("Poem's Generator")
st.markdown("Interact with your fine-tuned GPT-2 poetry model.")

title = st.text_input("Poem Title", value="Nature and Humanity")
poet = st.text_input("Poet Name", value="myself")
tags = st.text_input("Tags (comma-separated)", value="Peace, Hope, Joy, Struggle, Victory, Unity, Diversity")
num_poems = st.slider("Number of Poems", min_value=1, max_value=1000, value=1)

st.sidebar.header("Generation Settings")
temperature = st.sidebar.slider("Temperature", 0.1, 1.5, 0.7)
top_p = st.sidebar.slider("Top-p", 0.1, 1.0, 0.8)
top_k = st.sidebar.slider("Top-k", 0, 1000, 100)
max_length = st.sidebar.slider("Max Length", 50, 300, 150)
repetition_penalty = st.sidebar.slider("Repetition Penalty", 1.0, 2.0, 1.5)

if st.button("💬 Generate Poem"):
    prompt = (
        f"<|startofpoem|>\n"
        f"Title: {title}\n"
        f"Poet: {poet}\n"
        f"Tags: {tags}\n\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        num_return_sequences=num_poems,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.convert_tokens_to_ids("<|endofpoem|>")
    )

    st.subheader("poem generator")
    for i, output in enumerate(outputs):
        decoded_poem = tokenizer.decode(output, skip_special_tokens=True).strip()
        # Format like chat
        with st.chat_message("assistant"):
            st.markdown(f"**Poem {i+1}:**")
            st.markdown(decoded_poem)
