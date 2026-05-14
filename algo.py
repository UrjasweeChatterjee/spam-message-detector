import pandas as pd
import pickle
import re
import nltk

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# Read CSV file
data = pd.read_csv('data_file.csv')

# Encode labels
data['label'] = data['label'].map({
    'spam': 1,
    'not spam': 0
})

# Remove rows with missing labels/messages
data = data.dropna(subset=['message', 'label'])

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Clean text function
def clean_text(text):
    text = re.sub(r'\W', ' ', str(text))
    text = text.lower()
    text = text.split()
    text = [word for word in text if word not in stop_words]
    return ' '.join(text)

# Clean messages
data['cleaned_text'] = data['message'].apply(clean_text)

print(data[['message', 'cleaned_text', 'label']].head())

# Vectorize text
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(data['cleaned_text'])
y = data['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

# -----------------------------
# Naive Bayes Model
# -----------------------------
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_predictions = nb_model.predict(X_test)

nb_accuracy = accuracy_score(y_test, nb_predictions)
nb_report = classification_report(y_test, nb_predictions, output_dict=True)
nb_confusion = confusion_matrix(y_test, nb_predictions)

print("Naive Bayes Accuracy:", nb_accuracy)
print(classification_report(y_test, nb_predictions))

# -----------------------------
# Logistic Regression Model
# -----------------------------
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)

lr_accuracy = accuracy_score(y_test, lr_predictions)
lr_report = classification_report(y_test, lr_predictions, output_dict=True)
lr_confusion = confusion_matrix(y_test, lr_predictions)

print("Logistic Regression Accuracy:", lr_accuracy)
print(classification_report(y_test, lr_predictions))

# -----------------------------
# Random Forest Model
# -----------------------------
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_report = classification_report(y_test, rf_predictions, output_dict=True)
rf_confusion = confusion_matrix(y_test, rf_predictions)

print("Random Forest Accuracy:", rf_accuracy)
print(classification_report(y_test, rf_predictions))
print("Confusion Matrix for Random Forest:")
print(rf_confusion)

# Prediction function using Naive Bayes
def predict_spam(message):
    cleaned_message = clean_text(message)
    vectorized_message = vectorizer.transform([cleaned_message])
    prediction = nb_model.predict(vectorized_message)
    probability = nb_model.predict_proba(vectorized_message)

    return (
        "Spam" if prediction[0] == 1 else "Not Spam",
        probability[0][1]
    )

# Test predictions
print(predict_spam("Don't forget about your dentist appointment next Tuesday. Limited-time offer! Buy one and get one free on all items. Shop Now!"))

print(predict_spam("Don't forget about your dentist appointment next Tuesday. Exclusive offer! You have been selected to win a $500 gift card. Act now!"))

# Label distribution
print(data['label'].value_counts())

# Save all models and details in one pkl file
model_data = {
    'naive_bayes': nb_model,
    'logistic_regression': lr_model,
    'random_forest': rf_model,
    'vectorizer': vectorizer,
    'stop_words': stop_words,

    'model_metrics': {
        'naive_bayes': {
            'accuracy': nb_accuracy,
            'classification_report': nb_report,
            'confusion_matrix': nb_confusion
        },
        'logistic_regression': {
            'accuracy': lr_accuracy,
            'classification_report': lr_report,
            'confusion_matrix': lr_confusion
        },
        'random_forest': {
            'accuracy': rf_accuracy,
            'classification_report': rf_report,
            'confusion_matrix': rf_confusion
        }
    }
}

with open('spam_model.pkl', 'wb') as file:
    pickle.dump(model_data, file)

print("All models saved successfully in one file: spam_model.pkl")