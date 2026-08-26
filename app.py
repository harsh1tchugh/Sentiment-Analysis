from flask import Flask, render_template, request, jsonify
import joblib
import os
import json
import smtplib
from email.mime.text import MIMEText
from datetime import datetime
from dotenv import load_dotenv

from preprocessing import clean_text

load_dotenv()  # reads GMAIL_USER / GMAIL_APP_PASSWORD from a local .env file

app = Flask(__name__)

model = joblib.load("best_sentiment_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

GMAIL_USER = os.environ.get("GMAIL_USER")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD")
FEEDBACK_FILE = "feedback.json"

# --- Temporary debug output so we can see what .env actually loaded ---
print("DEBUG: GMAIL_USER =", GMAIL_USER)
print("DEBUG: GMAIL_APP_PASSWORD loaded =", bool(GMAIL_APP_PASSWORD))


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    confidence = None
    review_text = ""

    if request.method == "POST":
        review_text = request.form.get("review", "").strip()
        if review_text:
            cleaned = clean_text(review_text)
            vec = tfidf.transform([cleaned])
            pred = model.predict(vec)[0]

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(vec)[0]
                confidence = round(max(proba) * 100, 1)

            result = "positive" if pred == 1 else "negative"

    return render_template(
        "index.html", result=result, confidence=confidence, review_text=review_text
    )


def _save_feedback_to_file(data):
    """Append feedback to a local JSON file as a fallback when email isn't configured."""
    feedback_list = []
    if os.path.exists(FEEDBACK_FILE):
        try:
            with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
                feedback_list = json.load(f)
        except (json.JSONDecodeError, IOError):
            feedback_list = []

    data["timestamp"] = datetime.now().isoformat()
    feedback_list.append(data)

    with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
        json.dump(feedback_list, f, indent=2, ensure_ascii=False)


@app.route("/submit-feedback", methods=["POST"])
def submit_feedback():
    data = request.get_json(silent=True) or request.form
    name = data.get("name", "Anonymous")
    email = data.get("email", "Not provided")
    feedback_type = data.get("type", "Not specified")
    rating = data.get("rating", "Not rated")
    message = data.get("message", "")

    if not message or not message.strip():
        return jsonify({"ok": False, "error": "Message is required"}), 400

    feedback_record = {
        "name": name,
        "email": email,
        "type": feedback_type,
        "rating": rating,
        "message": message.strip(),
    }

    # Try sending email if credentials are configured
    if GMAIL_USER and GMAIL_APP_PASSWORD:
        try:
            body = (
                f"New feedback from {name} ({email})\n"
                f"Type: {feedback_type}\n"
                f"Rating: {rating}/5\n\n"
                f"Message:\n{message}"
            )
            msg = MIMEText(body)
            msg["Subject"] = f"SentiScan Feedback - {feedback_type}"
            msg["From"] = GMAIL_USER
            msg["To"] = GMAIL_USER

            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                server.starttls()
                server.login(GMAIL_USER, GMAIL_APP_PASSWORD)
                server.send_message(msg)

            print("DEBUG: Email sent successfully")
            return jsonify({"ok": True})
        except Exception as e:
            # Email failed — print why, then fall through to save to file
            print("DEBUG: Email send failed:", repr(e))
    else:
        print("DEBUG: Skipping email — GMAIL_USER or GMAIL_APP_PASSWORD missing")

    # Fallback: save feedback to local file
    try:
        _save_feedback_to_file(feedback_record)
        print("DEBUG: Feedback saved to feedback.json instead")
        return jsonify({"ok": True})
    except Exception as e:
        print("DEBUG: Failed to save feedback.json:", repr(e))
        return jsonify({"ok": False, "error": f"Failed to save feedback: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)