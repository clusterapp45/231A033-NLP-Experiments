from flask import Flask, render_template, request
import os
import joblib

app = Flask(__name__)
MODEL_PATH = os.path.join("model", "toxic_model.joblib")
model = None

def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    else:
        model = None

load_model()

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    comment = ""
    error = None

    if request.method == "POST":
        comment = request.form.get("comment", "").strip()
        if not comment:
            error = "Please enter a comment first."
        elif model is None:
            error = "Model not found. Run train_model.py first."
        else:
            label = model.predict([comment])[0]
            prediction = "Toxic" if int(label) == 1 else "Non-toxic"

    return render_template("index.html", prediction=prediction, comment=comment, error=error)

if __name__ == "__main__":
    app.run(debug=True)
