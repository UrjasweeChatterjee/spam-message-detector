from flask import Flask, render_template, request
import pickle
import re

app = Flask(__name__)

# Load the saved pkl file
with open('spam_model.pkl', 'rb') as file:
    model_data = pickle.load(file)

# Load models
naive_bayes = model_data['naive_bayes']
logistic_regression = model_data['logistic_regression']
random_forest = model_data['random_forest']

# Load vectorizer and stopwords
vectorizer = model_data['vectorizer']
stop_words = model_data['stop_words']

# Load saved model metrics
model_metrics = model_data.get('model_metrics', {})


def clean_text(text):
    text = re.sub(r'\W', ' ', str(text))
    text = text.lower()
    text = text.split()
    text = [word for word in text if word not in stop_words]
    return ' '.join(text)


def get_prediction_details(model, vectorized_message, metric_key):
    prediction = model.predict(vectorized_message)[0]

    probability = model.predict_proba(vectorized_message)[0]
    not_spam_probability = round(probability[0] * 100, 2)
    spam_probability = round(probability[1] * 100, 2)

    result = "Spam" if prediction == 1 else "Not Spam"

    metrics = model_metrics.get(metric_key, {})

    return {
        "result": result,
        "spam_probability": spam_probability,
        "not_spam_probability": not_spam_probability,
        "accuracy": round(metrics.get("accuracy", 0) * 100, 2),
        "confusion_matrix": metrics.get("confusion_matrix", "Not available")
    }


@app.route('/', methods=['GET', 'POST'])
def index():
    final_result = None
    message = ""
    cleaned_message = ""
    model_results = None

    if request.method == 'POST':
        message = request.form['message']

        cleaned_message = clean_text(message)
        vectorized_message = vectorizer.transform([cleaned_message])

        nb_result = get_prediction_details(
            naive_bayes,
            vectorized_message,
            "naive_bayes"
        )

        lr_result = get_prediction_details(
            logistic_regression,
            vectorized_message,
            "logistic_regression"
        )

        rf_result = get_prediction_details(
            random_forest,
            vectorized_message,
            "random_forest"
        )

        model_results = {
            "Naive Bayes": nb_result,
            "Logistic Regression": lr_result,
            "Random Forest": rf_result
        }

        final_result = nb_result["result"]

    return render_template(
        'index.html',
        message=message,
        cleaned_message=cleaned_message,
        final_result=final_result,
        model_results=model_results
    )


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5555))
    app.run(debug=False, host='0.0.0.0', port=port)