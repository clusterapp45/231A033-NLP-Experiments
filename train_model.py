"""
Quick-start training script.
IMPORTANT: This small built-in dataset is only for checking that the app works.
For valid academic results, replace it with a labelled, properly licensed dataset
and report only metrics measured on a held-out test set.
"""
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Small illustrative examples to make the starter project runnable immediately.
# 0 = non-toxic, 1 = toxic
texts = [
    "Thank you for your help", "Have a nice day", "I appreciate your effort",
    "That is a useful idea", "I respectfully disagree", "Could you explain this?",
    "Good job on the project", "I am happy to work with you", "Please be kind",
    "This answer helped me", "I think we should try again", "You did well",
    "You are an idiot", "I hate you", "Shut up you stupid fool",
    "Nobody likes you", "You are pathetic", "You are disgusting",
    "Get lost you moron", "You are worthless", "What a dumb thing to say",
    "You make me sick", "You are a horrible person", "Stop being such a jerk",
    "You are so stupid", "I will insult you every day", "You are a loser",
    "That was a thoughtful comment", "Let's discuss this calmly",
    "I disagree with your opinion", "Thanks for sharing your view",
    "Please stop bothering me", "This is not helpful", "I am frustrated by this",
    "You should be ashamed of yourself", "You are an awful idiot",
    "I hope you have a good week", "Can we solve this together?",
    "Your explanation is clear", "I do not agree with that"
]
labels = [
    0,0,0,0,0,0,0,0,0,0,0,0,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    0,0,0,0,0,0,0,1,1,0,0,0,0
]

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.25, random_state=42, stratify=labels
)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2))),
    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
])
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)

print("DEMO DATASET ONLY — do not use these metrics as final project results.")
print(f"Test accuracy: {accuracy_score(y_test, predictions):.3f}")
print("\nClassification report:")
print(classification_report(y_test, predictions, target_names=["Non-toxic", "Toxic"], zero_division=0))
print("Confusion matrix (rows=true, columns=predicted):")
print(confusion_matrix(y_test, predictions))

os.makedirs("model", exist_ok=True)
joblib.dump(pipeline, os.path.join("model", "toxic_model.joblib"))
print("\nSaved model to model/toxic_model.joblib")
