import os
import sys
import json
import joblib

from train import (
    clean_text,
    train_and_evaluate,
    MODEL_PATH,
    VECTORIZER_PATH,
    METRICS_PATH,
)


def load_or_train_model():
   
    if not (os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH) and os.path.exists(METRICS_PATH)):
        print("Model artifacts not found. Initiating automatic training...")
        train_and_evaluate(quiet=True)

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    return model, vectorizer, metrics["accuracy"], metrics["precision"]


def predict_message(text: str, model, vectorizer):
 
    cleaned = clean_text(text)
    if not cleaned:
        cleaned = text.lower().strip()

    vectorized = vectorizer.transform([cleaned])
    prediction_idx = model.predict(vectorized)[0]
    probabilities = model.predict_proba(vectorized)[0]

    confidence = float(probabilities[prediction_idx] * 100)
    label = "SPAM" if prediction_idx == 1 else "HAM"

    return label, confidence


def run_classifier():
    try:
        model, vectorizer, accuracy, precision = load_or_train_model()
    except Exception as e:
        print(f"Error loading/training model: {e}")
        sys.exit(1)

    print("========================================")
    print("       EMAIL SPAM CLASSIFIER")
    print("========================================")
    print()
    print(f"Model Accuracy : {accuracy:.2f}%")
    print(f"Precision      : {precision:.2f}%")
    print()

    while True:
        try:
            print("Enter email:")
            raw_input_text = input("> ").strip()

            if not raw_input_text:
                print("Email cannot be empty. Please enter some text.\n")
                continue

            prediction, confidence = predict_message(raw_input_text, model, vectorizer)

            print()
            print(f"Prediction : {prediction}")
            print(f"Confidence : {confidence:.2f}%")
            print()
            print("========================================")
            print()

            while True:
                choice = input("Enter another email? (y/n): ").strip().lower()
                if choice in ["y", "yes"]:
                    print()
                    break
                elif choice in ["n", "no"]:
                    print("\nThank you for using Email Spam Classifier. Exiting cleanly.\n")
                    return
                else:
                    print("Invalid option. Please enter 'y' for yes or 'n' for no.")

        except (KeyboardInterrupt, EOFError):
            print("\n\nSession terminated by user. Goodbye!\n")
            break


if __name__ == "__main__":
    run_classifier()
