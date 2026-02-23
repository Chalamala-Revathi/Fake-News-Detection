from flask import Flask, request, render_template
import pickle

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    news = request.form['news']

    news_vectorized = vectorizer.transform([news])
    prediction = model.predict(news_vectorized)

    if prediction[0] == 1:
        result = "Real News ✅"
    else:
        result = "Fake News ❌"

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)