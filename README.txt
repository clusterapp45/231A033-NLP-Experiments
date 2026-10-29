CYBERBULLYING AND TOXIC COMMENT DETECTION — QUICK START

Requirements: Python installed on Windows.

1. Extract this ZIP.
2. Open the extracted toxic_comment_detector folder in VS Code.
3. In VS Code, choose Terminal > New Terminal.
4. Run these commands one by one:

   py -m venv .venv
   .venv\Scripts\activate
   python -m pip install -r requirements.txt
   python train_model.py
   python app.py

5. Open the local URL printed in the terminal, normally:
   http://127.0.0.1:5000

6. Try a polite comment and an insulting comment. Capture genuine screenshots
   of the running app and its results.

IMPORTANT ACADEMIC NOTE:
This starter uses a tiny illustrative dataset so the application runs immediately.
It is NOT sufficient for reliable real-world detection and its metrics must NOT
be presented as final project results. For your final report, train and evaluate
on a properly sourced labelled dataset (such as the Jigsaw Toxic Comment dataset),
record the actual metrics, and update the report honestly. The interface predicts
only toxic/non-toxic; it does not identify separate bullying categories.
