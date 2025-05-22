import os
import cv2
import argparse
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from datasets import load_dataset, concatenate_datasets, Dataset as HFDataset
import pandas as pd

# ------------------------
# Evaluation Metrics
# ------------------------

def edit_distance(str1, str2):
    len1, len2 = len(str1), len(str2)
    dp = [[0]*(len2+1) for _ in range(len1+1)]
    for i in range(len1+1):
        dp[i][0] = i
    for j in range(len2+1):
        dp[0][j] = j
    for i in range(1, len1+1):
        for j in range(1, len2+1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[len1][len2]

def compute_cer_wer(refs, preds):
    total_chars, char_errors = 0, 0
    total_words, word_errors = 0, 0
    for ref, pred in zip(refs, preds):
        if not ref:
            continue
        total_chars += len(ref)
        char_errors += edit_distance(ref, pred)
        ref_words = ref.split()
        pred_words = pred.split()
        total_words += len(ref_words)
        word_errors += edit_distance(ref_words, pred_words)
    cer = (char_errors / total_chars * 100) if total_chars > 0 else 0
    wer = (word_errors / total_words * 100) if total_words > 0 else 0
    return cer, wer

# ------------------------
# Dataset Class
# ------------------------

class OCRDataset(Dataset):
    def __init__(self, data, processor, max_length=128):
        self.data = data
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        image = self._preprocess_image(item['image'])
        tokenized = self.processor.tokenizer(text,
                                             max_length=self.max_length,
                                             padding="max_length",
                                             truncation=True,
                                             return_tensors="pt")
        labels = tokenized.input_ids.squeeze(0)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        return pixel_values, labels

    def _preprocess_image(self, image):
        np_img = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        resized = cv2.resize(denoised, (384, 384))
        rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(rgb)

def collate_fn(batch):
    images = [b[0] for b in batch]
    labels = [b[1] for b in batch]
    return torch.stack(images), pad_sequence(labels, batch_first=True, padding_value=-100)

# ------------------------
# Training Function
# ------------------------

def train_model(imgur_path=None, epochs=10, lr=5e-5, batch_size=8, patience=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten").to(device)
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")

    print("📦 Loading IAM dataset...")
    iam_dataset = load_dataset("teklia/iam_line", split="train")
    iam_dataset = iam_dataset.remove_columns([c for c in iam_dataset.column_names if c not in ['image', 'text']])

    if imgur_path:
        print("📦 Loading Imgur5K from:", imgur_path)
        ann_path = os.path.join(imgur_path, "annotations.csv")
        if os.path.exists(ann_path):
            df = pd.read_csv(ann_path)
            imgur_samples = []
            for _, row in df.iterrows():
                img_file = os.path.join(imgur_path, row['filename'])
                if os.path.exists(img_file):
                    img = Image.open(img_file).convert("RGB")
                    imgur_samples.append({'image': img, 'text': row['text']})
            imgur_dataset = HFDataset.from_list(imgur_samples)
            combined = concatenate_datasets([iam_dataset, imgur_dataset])
        else:
            print("⚠️ Annotations file missing. Using IAM only.")
            combined = iam_dataset
    else:
        combined = iam_dataset

    # Split train/val
    split = combined.train_test_split(test_size=0.1, seed=42)
    train_loader = DataLoader(OCRDataset(split['train'], processor), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(OCRDataset(split['test'], processor), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler()
    best_val_loss = float('inf')
    no_improve = 0

    print("🚀 Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                out = model(pixel_values=x, labels=y)
                loss = out.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        print(f"✅ Epoch {epoch+1}: Train Loss = {total_loss/len(train_loader):.4f}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                loss = model(pixel_values=x, labels=y).loss
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        print(f"🔍 Validation Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve = 0
            model.save_pretrained("fine_tuned_trocr")
            processor.save_pretrained("fine_tuned_trocr")
            print("💾 Best model saved.")
        else:
            no_improve += 1
            if no_improve >= patience:
                print("🛑 Early stopping triggered.")
                break

    # Final evaluation
    print("📈 Evaluating model on validation set...")
    model.eval()
    refs, preds = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            output_ids = model.generate(x)
            decoded_preds = processor.batch_decode(output_ids, skip_special_tokens=True)
            for ref_label in y:
                ref_ids = ref_label[ref_label != -100]
                ref_text = processor.tokenizer.decode(ref_ids, skip_special_tokens=True)
                refs.append(ref_text)
            preds.extend(decoded_preds)
    cer, wer = compute_cer_wer(refs, preds)
    print(f"\n📊 Final CER: {cer:.2f}% {'✅' if cer <= 7 else '❌'}")
    print(f"📊 Final WER: {wer:.2f}% {'✅' if wer <= 15 else '❌'}")

# ------------------------
# Inference Function
# ------------------------

def run_prediction(image_paths):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VisionEncoderDecoderModel.from_pretrained("fine_tuned_trocr").to(device)
    processor = TrOCRProcessor.from_pretrained("fine_tuned_trocr")

    for path in image_paths:
        img = Image.open(path).convert("RGB")
        np_img = np.array(img)
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        resized = cv2.resize(denoised, (384, 384))
        rgb_img = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        final_img = Image.fromarray(rgb_img)
        pixel_values = processor(images=final_img, return_tensors="pt").pixel_values.to(device)
        output_ids = model.generate(pixel_values)
        prediction = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        print(f"📝 Prediction for {os.path.basename(path)}:\n{prediction}\n")

# ------------------------
# Entry Point
# ------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR Fine-Tuning and Prediction with TrOCR")
    parser.add_argument("--train", action="store_true", help="Run training pipeline")
    parser.add_argument("--predict", nargs="+", help="Run OCR prediction on image(s)")
    parser.add_argument("--imgur_path", type=str, help="Path to Imgur5K dataset folder")
    args = parser.parse_args()

    if args.train:
        train_model(imgur_path=args.imgur_path)
    elif args.predict:
        run_prediction(args.predict)
    else:
        print("❌ Please specify either --train or --predict.")
