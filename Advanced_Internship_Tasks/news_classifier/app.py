"""
Streamlit web app for the fine-tuned BERT AG News topic classifier.

Run with:
    streamlit run app.py

Expects a fine-tuned model saved at ./bert_agnews_model (produced by
task1_bert_classifier.ipynb), containing the standard Hugging Face
files (config.json, model weights, tokenizer files) plus an optional
label_mapping.json.
"""

import os
import json

import numpy as np
import streamlit as st

# Heavy ML imports are wrapped so the app can show a friendly error
# instead of crashing if torch/transformers aren't installed.
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    IMPORT_ERROR_MSG = str(e)

# ----------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="News Topic Classifier (BERT)",
    page_icon="📰",
    layout="centered",
)

MODEL_DIR = "./bert_agnews_model"
MAX_LENGTH = 64

# Fallback label names used if label_mapping.json is missing
DEFAULT_LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]

# Category emojis for a nicer display (purely cosmetic)
CATEGORY_EMOJI = {
    "World": "🌍",
    "Sports": "⚽",
    "Business": "💼",
    "Sci/Tech": "🔬",
}


@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer(model_dir: str):
    """
    Load the fine-tuned model, tokenizer, and label mapping from disk.
    Cached so the (relatively expensive) model load only happens once
    per Streamlit session, not on every user interaction.
    """
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            f"Model directory '{model_dir}' was not found. "
            f"Run task1_bert_classifier.ipynb first to fine-tune and save the model, "
            f"or update MODEL_DIR in app.py to point to your saved model."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()  # inference mode (disables dropout etc.)

    # Try to load the custom label mapping saved during training;
    # fall back to the model's own config, then to AG News defaults.
    label_path = os.path.join(model_dir, "label_mapping.json")
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            mapping = json.load(f)
        id2label = {int(k): v for k, v in mapping["id2label"].items()}
    elif getattr(model.config, "id2label", None):
        id2label = {int(k): v for k, v in model.config.id2label.items()}
    else:
        id2label = {i: name for i, name in enumerate(DEFAULT_LABEL_NAMES)}

    return model, tokenizer, id2label


def predict(text: str, model, tokenizer, id2label: dict):
    """
    Run inference on a single piece of text and return a dict of
    {label_name: probability} for every class, plus the top prediction.
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze().numpy()

    # Guard against a single-example batch collapsing to a 0-d array
    probs = np.atleast_1d(probs)

    num_labels = len(id2label)
    label_probs = {id2label[i]: float(probs[i]) for i in range(num_labels)}
    top_idx = int(np.argmax(probs))
    top_label = id2label[top_idx]
    top_confidence = float(probs[top_idx])

    return top_label, top_confidence, label_probs


# ----------------------------------------------------------------------
# Main app layout
# ----------------------------------------------------------------------
def main():
    st.title("📰 News Topic Classifier")
    st.markdown(
        "Classify news headlines into **World**, **Sports**, **Business**, or "
        "**Sci/Tech** using a BERT model fine-tuned on the AG News dataset."
    )

    if not TRANSFORMERS_AVAILABLE:
        st.error(
            "Required packages are missing. Please install dependencies first:\n\n"
            "```\npip install -r requirements.txt\n```\n\n"
            f"Import error detail: {IMPORT_ERROR_MSG}"
        )
        st.stop()

    # Load model (cached after first run)
    try:
        with st.spinner("Loading fine-tuned BERT model... (first load may take a few seconds)"):
            model, tokenizer, id2label = load_model_and_tokenizer(MODEL_DIR)
    except FileNotFoundError as e:
        st.error(str(e))
        st.info(
            "Expected folder structure:\n\n"
            "```\nbert_agnews_model/\n"
            "  config.json\n"
            "  model.safetensors (or pytorch_model.bin)\n"
            "  tokenizer_config.json\n"
            "  vocab.txt\n"
            "  label_mapping.json (optional)\n```"
        )
        st.stop()
    except Exception as e:
        st.error(f"Failed to load the model: {e}")
        st.stop()

    st.success("Model loaded successfully ✅")

    st.divider()

    # ------------------------------------------------------------------
    # Input section
    # ------------------------------------------------------------------
    st.subheader("Enter a news headline")

    example_headlines = [
        "-- Select an example --",
        "Apple unveils new chip with record-breaking AI performance",
        "Manchester United wins dramatic match in final minutes",
        "Stock markets rally after central bank interest rate cut",
        "NASA announces new mission to study Jupiter's moons",
    ]
    selected_example = st.selectbox("Or pick an example headline:", example_headlines)

    default_text = "" if selected_example == example_headlines[0] else selected_example
    headline = st.text_area(
        "Headline text",
        value=default_text,
        placeholder="e.g. Tech company announces breakthrough in quantum computing",
        height=100,
    )

    classify_clicked = st.button("Classify Topic", type="primary")

    # ------------------------------------------------------------------
    # Prediction & results
    # ------------------------------------------------------------------
    if classify_clicked:
        cleaned_text = headline.strip()

        # --- Input validation / error handling ---
        if not cleaned_text:
            st.warning("⚠️ Please enter a headline before classifying.")
        elif len(cleaned_text) < 3:
            st.warning("⚠️ The input text is too short to classify meaningfully.")
        else:
            try:
                with st.spinner("Classifying..."):
                    top_label, top_confidence, label_probs = predict(
                        cleaned_text, model, tokenizer, id2label
                    )

                st.divider()
                st.subheader("Prediction Result")

                emoji = CATEGORY_EMOJI.get(top_label, "🏷️")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="Predicted Topic", value=f"{emoji} {top_label}")
                with col2:
                    st.metric(label="Confidence", value=f"{top_confidence * 100:.2f}%")

                st.markdown("##### Confidence breakdown across all categories")

                # Sort categories by descending probability for display
                sorted_items = sorted(label_probs.items(), key=lambda x: x[1], reverse=True)
                for label, prob in sorted_items:
                    label_emoji = CATEGORY_EMOJI.get(label, "🏷️")
                    st.write(f"{label_emoji} **{label}**: {prob * 100:.2f}%")
                    st.progress(min(max(prob, 0.0), 1.0))

                if top_confidence < 0.5:
                    st.info(
                        "ℹ️ The model's confidence is relatively low for this input — "
                        "the headline may be ambiguous or contain mixed topics."
                    )

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

    st.divider()
    with st.expander("About this app"):
        st.markdown(
            """
This app loads a **bert-base-uncased** model fine-tuned on the **AG News** dataset
(4 categories: World, Sports, Business, Sci/Tech) using the Hugging Face `Trainer` API.

- Model and tokenizer are loaded once and cached for the session.
- Inference runs on CPU.
- See `task1_bert_classifier.ipynb` for the full training pipeline.
            """
        )


if __name__ == "__main__":
    main()
