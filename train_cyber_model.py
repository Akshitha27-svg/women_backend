import pandas as pd
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load dataset
data = pd.read_csv("data/cyber_dataset.csv")

X = data["text"]
y = data["label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", LogisticRegression())
])

# Train
model.fit(X_train, y_train)

# Evaluate
pred = model.predict(X_test)
print(classification_report(y_test, pred))

# Save model
os.makedirs("models", exist_ok=True)

with open("models/cyber_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Cyberbullying model trained and saved.")