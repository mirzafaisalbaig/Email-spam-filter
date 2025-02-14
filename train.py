import pandas as pd
import re
import nltk
import pickle
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

nltk.download("stopwords")

# Load dataset 
df = pd.read_csv("./synthetic_phishing_dataset.csv")

# Function to clean email text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text) #remove special characters
    text = re.sub(r'\s+', ' ', text) #remove extra space
    text = ' '.join([word for word in text.split() if word not in stopwords.words("english")])
    return text
# Apply text cleaning
df["Sender"] = df["Sender"].apply(clean_text)

# Convert text into numerical features
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['Sender'])
y = df["Label"]

# Split dataset into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# save model & vectorizer

pickle.dump(model, open("phishing_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model and vectorizer saved successfully!")

