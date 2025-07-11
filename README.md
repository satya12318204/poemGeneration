# GPT-2 Fine-Tuned Poetry Generator

This project fine-tunes a pre-trained GPT-2 model using a custom poetry dataset to generate high-quality poems. The model is trained to understand structure, theme, and poetic flow through domain-specific tokens and carefully prepared data.

## 📖 Overview

This repository includes:

* A fine-tuning pipeline for GPT-2 using the Hugging Face Transformers library.
* A Streamlit-based app for generating poems interactively.
* Detailed documentation for reproducibility and understanding.

## 🔧 Environment Setup

Make sure you have Python 3.7+ installed. All dependencies are listed in `requirements.txt`. To install them:

```bash
pip install -r requirements.txt
```

## 🚀 Features

* Custom dataset preprocessing and tokenization
* GPT-2 model fine-tuning using Hugging Face Trainer
* Inference pipeline with top-k, top-p sampling
* Web app interface for poem generation via Streamlit
* Support for saving and loading trained models

## 🔹 Fine-Tuning Pipeline

1. **Prepare Dataset**: Place your poetry dataset in `.txt` or `.csv` format.
2. **Tokenization**: Tokenization is handled within the training pipeline or pre-processing notebook.
3. **Training**: Use a notebook (`generator.ipynb`) or scripts to fine-tune the model.
4. **Saving**: Trained model and tokenizer are saved under `fine_tuned_poetry_model_v3/`.

## 🎨 Streamlit App

Launch the poem generator web interface using:

```bash
streamlit run app.py
```

Select generation parameters like temperature, top-k, and max length to control the creativity of the output.

## 📁 Project Structure

```plaintext
.
├── .gitattributes
├── LICENSE
├── GPT2_Poetry_Documentation.docx       # Full documentation of model, training, and usage
├── PoetryFoundationData.csv             # Source poetry dataset
├── README.md                            # Project overview
├── app.py                               # Streamlit web UI for poem generation
├── generator.ipynb                      # Notebook for model training and generation
├── requirements.txt                     # Python dependencies
└── fine_tuned_poetry_model_v3/          # Directory storing the fine-tuned GPT-2 model
```

## ⚙️ Example Usage

Example command to fine-tune GPT-2 (if applicable):

```bash
python train.py --model_name gpt2 --train_file data/poems.txt --output_dir models/ --epochs 4 --batch_size 2
```

## 💡 Tips for Better Results

* Start with a small dataset and increase size gradually.
* Experiment with training epochs and learning rate.
* Use temperature and top-p settings to adjust randomness in output.

## 📄 License

MIT License.

---

Happy fine-tuning and enjoy generating beautiful poetry! 🌟
