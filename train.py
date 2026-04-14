import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from tqdm import tqdm

# ============================================================
# 1. Muat Dataset
# ============================================================
df = pd.read_csv('dataset_final_balanced.csv')
df = df.dropna(subset=['text_clean', 'label']).copy()
df['label'] = df['label'].astype(int)

sentences = df['text_clean'].astype(str).tolist()
labels    = df['label'].values

# ============================================================
# 2. Split Data
# ============================================================
train_sentences, test_sentences, train_labels, test_labels = train_test_split(
    sentences,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

# ============================================================
# 3. Hitung Class Weight (jaring pengaman tambahan)
# ============================================================
# compute_class_weight akan menghitung bobot proporsional
# berdasarkan distribusi label di data training
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Menggunakan device: {device}")
print(f"Class weights — Label 0: {class_weights[0]:.4f} | Label 1: {class_weights[1]:.4f}")

# Konversi ke tensor dan pindahkan ke device
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

# Loss function dengan bobot — ini pengganti outputs.loss bawaan model
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

# ============================================================
# 4. Tokenizer IndoBERT
# ============================================================
MODEL_NAME = "indolem/indobert-base-uncased"
MAX_LENGTH = 128

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ============================================================
# 5. Custom Dataset Class
# ============================================================
class KomentarDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(
            texts,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids':      self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels':         self.labels[idx]
        }

BATCH_SIZE = 16

train_dataset = KomentarDataset(train_sentences, train_labels, tokenizer, MAX_LENGTH)
test_dataset  = KomentarDataset(test_sentences,  test_labels,  tokenizer, MAX_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

# ============================================================
# 6. Load Model
# ============================================================
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)
model.to(device)

# ============================================================
# 7. Optimizer & Scheduler
# ============================================================
EPOCHS        = 3
LEARNING_RATE = 2e-5

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

total_steps = len(train_loader) * EPOCHS
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=int(0.1 * total_steps),   # warmup 10% awal — lebih stabil dari 0
    num_training_steps=total_steps
)

# ============================================================
# 8. Training Loop
# ============================================================
def train_epoch(model, loader, optimizer, scheduler, device, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for batch in tqdm(loader, desc="Training"):
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['labels'].to(device)

        optimizer.zero_grad()

        # Tidak kirim 'labels' ke model agar loss tidak dihitung otomatis,
        # karena kita pakai criterion sendiri yang sudah berbobot
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits  = outputs.logits

        # Hitung loss dengan class weight
        loss = criterion(logits, labels)

        loss.backward()

        # Gradient clipping — mencegah exploding gradient saat fine-tuning BERT
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds       = torch.argmax(logits, dim=1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

    return total_loss / len(loader), correct / total


def eval_epoch(model, loader, device, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluasi"):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits  = outputs.logits

            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds       = torch.argmax(logits, dim=1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

    return total_loss / len(loader), correct / total


# ============================================================
# 9. Jalankan Training
# ============================================================
print("\nMemulai fine-tuning IndoBERT (PyTorch)...")

best_val_loss = float('inf')

for epoch in range(EPOCHS):
    print(f"\n=== Epoch {epoch + 1}/{EPOCHS} ===")

    train_loss, train_acc = train_epoch(model, train_loader, optimizer, lr_scheduler, device, criterion)
    val_loss,   val_acc   = eval_epoch(model, test_loader, device, criterion)

    print(f"Train  — Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f}")
    print(f"Val    — Loss: {val_loss:.4f}   | Accuracy: {val_acc:.4f}")

    # Simpan model terbaik berdasarkan val_loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        model.save_pretrained('./model_filterjudol_best')
        tokenizer.save_pretrained('./model_filterjudol_best')
        print(f"  ✓ Model terbaik disimpan (val_loss: {val_loss:.4f})")

# ============================================================
# 10. Simpan Model Final
# ============================================================
model.save_pretrained('./model_filterjudol')
tokenizer.save_pretrained('./model_filterjudol')
print("\nSelesai! Model disimpan di folder ./model_filterjudol")
print(f"Model terbaik tersimpan di ./model_filterjudol_best (val_loss: {best_val_loss:.4f})")
