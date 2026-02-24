import torch
import torch.nn as nn
import torch.optim as optim
import random

# Expanded Symptom List
SYMPTOM_LIST = [
    "irregular periods", "missed periods", "acne", "weight gain",
    "excess facial hair", "hair thinning", "dark neck patches",
    "ovarian cyst history",
    "fatigue", "cold sensitivity", "heat intolerance",
    "hair loss", "weight change", "dry skin",
    "constipation", "depression",
    "mood swings", "bloating", "sleep issues",
    "anxiety", "pelvic pain", "heavy bleeding",
    "low energy"
]

NUM_CLASSES = 3  # 0=PCOS, 1=Thyroid, 2=Low Risk

class HealthClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

def generate_case(label):
    vector = [0] * len(SYMPTOM_LIST)

    if label == 0:  # PCOS
        key_symptoms = [
            "irregular periods", "acne", "weight gain",
            "excess facial hair", "hair thinning"
        ]
    elif label == 1:  # Thyroid
        key_symptoms = [
            "fatigue", "cold sensitivity", "hair loss",
            "weight change", "dry skin"
        ]
    else:
        key_symptoms = random.sample(SYMPTOM_LIST, 3)

    for s in key_symptoms:
        idx = SYMPTOM_LIST.index(s)
        vector[idx] = random.randint(2, 3)

    return vector

# Generate 90 cases
X = []
y = []

for _ in range(30):
    X.append(generate_case(0))
    y.append(0)

for _ in range(30):
    X.append(generate_case(1))
    y.append(1)

for _ in range(30):
    X.append(generate_case(2))
    y.append(2)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

model = HealthClassifier(len(SYMPTOM_LIST), NUM_CLASSES)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(300):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), "models/health_model.pt")
print("✅ Advanced Health Model Trained & Saved")