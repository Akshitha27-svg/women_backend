import json
import torch
import os
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm

with open("data/legal_dataset.json", "r", encoding="utf-8") as f:
    legal_data = json.load(f)

texts = []
labels = []
intent_map = {}

for idx, item in enumerate(legal_data):
    intent_map[idx] = item["intent"]
    for example in item["examples"]:
        texts.append(example)
        labels.append(idx)

os.makedirs("models", exist_ok=True)

with open("models/legal_intent_map.json", "w") as f:
    json.dump(intent_map, f)

train_texts, _, train_labels, _ = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

class LegalDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True)
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = LegalDataset(train_texts, train_labels)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(intent_map)
)

optimizer = AdamW(model.parameters(), lr=2e-5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(3):
    model.train()
    for batch in tqdm(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

torch.save(model.state_dict(), "models/legal_distilbert_model.pt")

print("✅ Legal DistilBERT model saved.")