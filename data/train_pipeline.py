import os
import re
import random
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_cosine_schedule_with_warmup,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings("ignore")

CFG = {
    "models": [
        "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "cardiffnlp/twitter-xlm-roberta-base-sentiment",
        "microsoft/deberta-v3-base",
    ],
    "max_len": 128,
    "batch_size": 32,
    "grad_accum": 1,
    "epochs": 5,
    "lr": 2e-5,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "n_folds": 5,
    "seed": 42,
    "fp16": True,
    "label_smoothing": 0.1,
    "pseudo_label_threshold": 0.90,
    "data_dir": Path("./"),
    "save_dir": Path("./models"),
}

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def clean_tweet(text: str) -> str:
    text = re.sub(r"https?://\S+|www\.\S+", "[URL]", text)
    text = re.sub(r"@\w+", "@user", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = text.replace("…", " ")
    return re.sub(r"\s+", " ", text).strip()


class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
        }
        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze()

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


def get_class_weights(labels: np.ndarray, num_classes: int = 3) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float)


def train_one_epoch(model, loader, optimizer, scheduler, scaler, criterion, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        with autocast(enabled=CFG["fp16"]):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            loss = criterion(outputs.logits, labels)

        scaler.scale(loss).backward()

        if (step + 1) % CFG["grad_accum"] == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        preds = outputs.logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, f1


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        with autocast(enabled=CFG["fp16"]):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            loss = criterion(outputs.logits, labels)

        total_loss += loss.item()
        preds = outputs.logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, f1, np.array(all_preds)


@torch.no_grad()
def predict_proba(model, loader, device):
    model.eval()
    all_probs = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        with autocast(enabled=CFG["fp16"]):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
        all_probs.append(probs)

    return np.vstack(all_probs)


def run_kfold(model_name, texts, labels, test_texts, device):
    CFG["save_dir"].mkdir(exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    class_weights = get_class_weights(labels).to(device)

    skf = StratifiedKFold(n_splits=CFG["n_folds"], shuffle=True, random_state=CFG["seed"])

    oof_probs = np.zeros((len(texts), 3))
    test_probs_folds = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
        print(f"\n  ── Fold {fold+1}/{CFG['n_folds']} ──")

        train_texts = [texts[i] for i in train_idx]
        train_labels = labels[train_idx]
        val_texts = [texts[i] for i in val_idx]
        val_labels = labels[val_idx]

        train_ds = TweetDataset(train_texts, train_labels, tokenizer, CFG["max_len"])
        val_ds = TweetDataset(val_texts, val_labels, tokenizer, CFG["max_len"])
        test_ds = TweetDataset(test_texts, None, tokenizer, CFG["max_len"])

        train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=CFG["batch_size"] * 2, shuffle=False, num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=CFG["batch_size"] * 2, shuffle=False, num_workers=2, pin_memory=True)

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3,
            ignore_mismatched_sizes=True,
        ).to(device)

        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=CFG["label_smoothing"])

        optimizer = AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
        total_steps = len(train_loader) * CFG["epochs"] // CFG["grad_accum"]
        warmup_steps = int(total_steps * CFG["warmup_ratio"])
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        scaler = GradScaler(enabled=CFG["fp16"])

        best_f1 = 0.0
        best_ckpt = CFG["save_dir"] / f"fold{fold}_best.pt"

        for epoch in range(CFG["epochs"]):
            train_loss, train_f1 = train_one_epoch(
                model, train_loader, optimizer, scheduler, scaler, criterion, device
            )
            val_loss, val_f1, _ = evaluate(model, val_loader, criterion, device)

            print(f"  Epoch {epoch+1}: train_loss={train_loss:.4f} train_f1={train_f1:.4f} | val_loss={val_loss:.4f} val_f1={val_f1:.4f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(model.state_dict(), best_ckpt)
                print(f"  ✓ Saved best model (f1={best_f1:.4f})")

        model.load_state_dict(torch.load(best_ckpt, map_location=device))

        oof_probs[val_idx] = predict_proba(model, val_loader, device)
        test_probs_folds.append(predict_proba(model, test_loader, device))

        del model
        torch.cuda.empty_cache()

    test_probs = np.mean(test_probs_folds, axis=0)

    oof_preds = oof_probs.argmax(axis=1)
    oof_f1 = f1_score(labels, oof_preds, average="macro")
    print(f"\n  OOF Macro F1: {oof_f1:.4f}")
    print(classification_report(labels, oof_preds, target_names=list(LABEL2ID.keys())))

    return oof_probs, test_probs, oof_f1


def add_pseudo_labels(train_texts, train_labels, test_texts, test_probs, threshold):
    max_probs = test_probs.max(axis=1)
    pseudo_mask = max_probs >= threshold
    pseudo_labels = test_probs.argmax(axis=1)[pseudo_mask]
    pseudo_texts = [test_texts[i] for i in np.where(pseudo_mask)[0]]

    n_pseudo = len(pseudo_texts)
    print(f"\nPseudo-labeling: adding {n_pseudo}/{len(test_texts)} test samples (threshold={threshold})")

    combined_texts = train_texts + pseudo_texts
    combined_labels = np.concatenate([train_labels, pseudo_labels])
    return combined_texts, combined_labels


def ensemble_and_submit(model_oof_probs, model_test_probs, model_f1s, train_labels, test_ids):
    weights = np.array(model_f1s)
    weights = weights / weights.sum()

    print("\n── Ensemble weights ──")
    for name, w, f1 in zip(CFG["models"], weights, model_f1s):
        print(f"  {name}: weight={w:.3f}, oof_f1={f1:.4f}")

    ensemble_oof = sum(w * p for w, p in zip(weights, model_oof_probs))
    ensemble_test = sum(w * p for w, p in zip(weights, model_test_probs))

    oof_preds = ensemble_oof.argmax(axis=1)
    ensemble_f1 = f1_score(train_labels, oof_preds, average="macro")
    print(f"\nEnsemble OOF Macro F1: {ensemble_f1:.4f}")
    print(classification_report(train_labels, oof_preds, target_names=list(LABEL2ID.keys())))

    test_preds = ensemble_test.argmax(axis=1)

    submission = pd.DataFrame({"id": test_ids, "target": test_preds})
    out_path = CFG["data_dir"] / "submission.csv"
    submission.to_csv(out_path, index=False)
    print(f"\nSubmission saved → {out_path}")
    return submission


def main():
    seed_everything(CFG["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_df = pd.read_csv(CFG["data_dir"] / "train.csv")
    test_df = pd.read_csv(CFG["data_dir"] / "test.csv")

    train_df["clean_tweet"] = train_df["tweet"].apply(clean_tweet)
    test_df["clean_tweet"] = test_df["tweet"].apply(clean_tweet)

    train_texts = train_df["clean_tweet"].tolist()
    train_labels = train_df["target"].map(LABEL2ID).values
    test_texts = test_df["clean_tweet"].tolist()
    test_ids = test_df["id"].tolist()

    print(f"\nTrain: {len(train_texts)}, Test: {len(test_texts)}")
    print(f"Class distribution: {dict(zip(LABEL2ID.keys(), np.bincount(train_labels)))}")

    model_oof_probs = []
    model_test_probs = []
    model_f1s = []

    for model_name in CFG["models"]:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        oof_probs, test_probs, oof_f1 = run_kfold(
            model_name=model_name,
            texts=train_texts,
            labels=train_labels,
            test_texts=test_texts,
            device=device,
        )

        model_oof_probs.append(oof_probs)
        model_test_probs.append(test_probs)
        model_f1s.append(oof_f1)

    best_model_idx = int(np.argmax(model_f1s))
    best_test_probs = model_test_probs[best_model_idx]
    best_model_name = CFG["models"][best_model_idx]

    aug_texts, aug_labels = add_pseudo_labels(
        train_texts, train_labels,
        test_texts, best_test_probs,
        threshold=CFG["pseudo_label_threshold"],
    )

    if len(aug_texts) > len(train_texts):
        print(f"\n── Re-training best model ({best_model_name}) with pseudo-labels ──")
        pseudo_oof, pseudo_test, pseudo_f1 = run_kfold(
            model_name=best_model_name,
            texts=aug_texts,
            labels=aug_labels,
            test_texts=test_texts,
            device=device,
        )
        model_oof_probs[best_model_idx] = pseudo_oof[:len(train_texts)]
        model_test_probs[best_model_idx] = pseudo_test
        model_f1s[best_model_idx] = f1_score(
            train_labels, pseudo_oof[:len(train_texts)].argmax(axis=1), average="macro"
        )

    print(f"\n{'='*60}")
    print("Building ensemble submission")
    print(f"{'='*60}")

    submission = ensemble_and_submit(
        model_oof_probs, model_test_probs, model_f1s, train_labels, test_ids
    )


if __name__ == "__main__":
    main()
