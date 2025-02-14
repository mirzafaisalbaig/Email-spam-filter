from flask import Flask, request, jsonify
import pickle
import re
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")

app = Flask(__name__)

#Load trained model and vectorized

model = pickle.load(open("phishing_model.pkl", "rb"))
Vectorizer = pickle.load(open("vectorizer.pkl", "rb"))  

#function to clean email text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = ' '.join([word for word in text.split() if word not in stopwords.words("english")])
    return text

@app.route("/predict", methods=["POST","GET"])

def predict():
    data = request.get_json()
    email_text = data.get("email"," ")

    #Preprocess email
    cleaned_email = clean_text(email_text)
    vectorized_email =Vectorizer.transform([cleaned_email])

    # Predict
    prediction = model.predict(vectorized_email)
    result = "phishing Email" if prediction == 1 else "Legitimate Email"

    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(debug=True)