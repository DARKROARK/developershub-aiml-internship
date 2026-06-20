import os
os.environ["HF_HOME"] = os.path.abspath("./.hf_cache")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import random
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

def main():
    # Reproducibility
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Force CPU usage explicitly
    DEVICE = torch.device("cpu")
    print(f"Using device: {DEVICE}")
    print(f"Torch threads available: {torch.get_num_threads()}")

    print("Loading the AG News dataset...")
    # Load the full AG News dataset from Hugging Face
    raw_dataset = load_dataset("ag_news")
    print(raw_dataset)

    LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
    NUM_LABELS = len(LABEL_NAMES)
    id2label = {i: name for i, name in enumerate(LABEL_NAMES)}
    label2id = {name: i for i, name in enumerate(LABEL_NAMES)}

    print("Creating CPU-Friendly Subset...")
    USE_SUBSET = True       # Set to False to use the full AG News dataset
    TRAIN_SUBSET_SIZE = 10000
    TEST_SUBSET_SIZE = 2000

    if USE_SUBSET:
        # Stratified sampling: shuffle then take a balanced slice per class
        train_df = pd.DataFrame(raw_dataset["train"])
        test_df = pd.DataFrame(raw_dataset["test"])

        per_class_train = TRAIN_SUBSET_SIZE // NUM_LABELS
        per_class_test = TEST_SUBSET_SIZE // NUM_LABELS

        train_sampled = (
            train_df.groupby("label", group_keys=False)
            .apply(lambda x: x.sample(n=per_class_train, random_state=SEED))
            .reset_index(drop=True)
        )
        test_sampled = (
            test_df.groupby("label", group_keys=False)
            .apply(lambda x: x.sample(n=per_class_test, random_state=SEED))
            .reset_index(drop=True)
        )

        # Shuffle rows so classes are interleaved, not grouped
        train_sampled = train_sampled.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
        test_sampled = test_sampled.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

        from datasets import Dataset
        train_data = Dataset.from_pandas(train_sampled)
        test_data = Dataset.from_pandas(test_sampled)
    else:
        train_data = raw_dataset["train"]
        test_data = raw_dataset["test"]

    print(f"Train size: {len(train_data)}")
    print(f"Test size:  {len(test_data)}")

    print("Tokenizing data...")
    MODEL_NAME = "bert-base-uncased"
    MAX_LENGTH = 64  # short sequence length -> much faster CPU training

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding=False,          # dynamic padding
            truncation=True,
            max_length=MAX_LENGTH,
        )

    train_tokenized = train_data.map(tokenize_function, batched=True, remove_columns=["text"])
    test_tokenized = test_data.map(tokenize_function, batched=True, remove_columns=["text"])

    # Rename "label" -> "labels" as expected by Hugging Face Trainer
    train_tokenized = train_tokenized.rename_column("label", "labels")
    test_tokenized = test_tokenized.rename_column("label", "labels")

    # Drop any leftover index columns from the pandas conversion, if present
    for col in ["__index_level_0__"]:
        if col in train_tokenized.column_names:
            train_tokenized = train_tokenized.remove_columns([col])
        if col in test_tokenized.column_names:
            test_tokenized = test_tokenized.remove_columns([col])

    train_tokenized = train_tokenized.with_format("torch")
    test_tokenized = test_tokenized.with_format("torch")

    print("Train tokenized structure:", train_tokenized)
    
    # Dynamic padding collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    print("Loading pretrained BERT Model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=id2label,
        label2id=label2id,
    )
    model.to(DEVICE)
    print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average="weighted")
        return {
            "accuracy": acc,
            "f1": f1,
        }

    OUTPUT_DIR = "./bert_agnews_checkpoints"

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,                 # 2 epochs
        per_device_train_batch_size=8,       # batch size 8
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        use_cpu=True,                        # force CPU
        dataloader_num_workers=0,
        report_to="none",                    # disable wandb/tensorboard logging
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=test_tokenized,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting training (this may take 15-30 minutes on CPU)...")
    train_result = trainer.train()
    print("Training finished!")
    print(train_result)

    print("Running final evaluation...")
    eval_metrics = trainer.evaluate()
    print("Final evaluation metrics on the test set:")
    for k, v in eval_metrics.items():
        print(f"  {k}: {v}")

    print("Generating classification report...")
    predictions_output = trainer.predict(test_tokenized)
    y_pred = np.argmax(predictions_output.predictions, axis=-1)
    y_true = predictions_output.label_ids
    print(classification_report(y_true, y_pred, target_names=LABEL_NAMES, digits=4))

    print("Generating Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
        cbar=True,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix - BERT AG News Classifier")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close()
    print("Confusion Matrix saved as confusion_matrix.png")

    SAVE_DIR = "./bert_agnews_model"
    os.makedirs(SAVE_DIR, exist_ok=True)
    trainer.save_model(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)

    import json
    with open(os.path.join(SAVE_DIR, "label_mapping.json"), "w") as f:
        json.dump({"id2label": id2label, "label2id": label2id}, f, indent=2)

    print(f"Model, tokenizer, and label mapping saved to: {SAVE_DIR}")

    print("Running Inference Check:")
    def predict_topic(text, model, tokenizer, device=DEVICE):
        model.eval()
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LENGTH
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        pred_id = int(np.argmax(probs))
        return LABEL_NAMES[pred_id], float(probs[pred_id])

    sample_headlines = [
        "Apple unveils new chip with record-breaking AI performance",
        "Manchester United wins dramatic match in final minutes",
        "Stock markets rally after central bank interest rate cut",
        "NASA announces new mission to study Jupiter's moons",
    ]

    for headline in sample_headlines:
        label, confidence = predict_topic(headline, model, tokenizer)
        print(f"Headline: {headline}")
        print(f"  -> Predicted: {label} (confidence: {confidence:.4f})\n")

if __name__ == "__main__":
    main()
