from __future__ import annotations

from pathlib import Path
import os
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)

# Instructor keeps the real test labels here
GROUND_TRUTH_FILE = BASE_DIR / "ml_bi_hackathon_test.csv"
LEADERBOARD_FILE = BASE_DIR / "leaderboard.csv"

ALLOWED_EXTENSIONS = {"csv"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024
app.secret_key = "change-this-secret-key"


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_leaderboard() -> pd.DataFrame:
    if LEADERBOARD_FILE.exists():
        df = pd.read_csv(LEADERBOARD_FILE)
    else:
        df = pd.DataFrame(columns=["team_name", "accuracy", "f1", "precision", "recall"])

    if not df.empty:
        df = df.sort_values(["f1", "accuracy"], ascending=False).reset_index(drop=True)
        df.index = df.index + 1
    return df


def score_submission(pred_file: Path) -> dict:
    truth = pd.read_csv(GROUND_TRUTH_FILE)
    pred = pd.read_csv(pred_file)

    required_cols = {"customer_id", "churn_prediction"}
    if not required_cols.issubset(pred.columns):
        raise ValueError("Submission must contain: customer_id, churn_prediction")

    merged = truth[["customer_id", "churned"]].merge(
        pred[["customer_id", "churn_prediction"]],
        on="customer_id",
        how="left",
        validate="one_to_one",
    )

    if merged["churn_prediction"].isna().any():
        missing_count = int(merged["churn_prediction"].isna().sum())
        raise ValueError(f"Submission is missing predictions for {missing_count} test rows.")

    y_true = merged["churned"].astype(int)
    y_pred = merged["churn_prediction"].astype(int)

    invalid = ~y_pred.isin([0, 1])
    if invalid.any():
        raise ValueError("churn_prediction must contain only 0 or 1 values.")

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
    }


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        team_name = request.form.get("team_name", "").strip()
        file = request.files.get("file")

        if not team_name:
            flash("Please enter a team name.", "danger")
            return redirect(url_for("index"))

        if file is None or file.filename == "":
            flash("Please upload a CSV file.", "danger")
            return redirect(url_for("index"))

        if not allowed_file(file.filename):
            flash("Only CSV files are allowed.", "danger")
            return redirect(url_for("index"))

        filename = secure_filename(f"{team_name}_{file.filename}")
        save_path = UPLOAD_FOLDER / filename
        file.save(save_path)

        try:
            scores = score_submission(save_path)
        except Exception as exc:
            flash(str(exc), "danger")
            return redirect(url_for("index"))

        leaderboard = pd.read_csv(LEADERBOARD_FILE) if LEADERBOARD_FILE.exists() else pd.DataFrame(
            columns=["team_name", "accuracy", "f1", "precision", "recall"]
        )

        leaderboard = leaderboard[leaderboard["team_name"] != team_name]
        new_row = pd.DataFrame([{ "team_name": team_name, **scores }])
        leaderboard = pd.concat([leaderboard, new_row], ignore_index=True)
        leaderboard.to_csv(LEADERBOARD_FILE, index=False)

        flash(f"Submission accepted for {team_name}.", "success")
        return redirect(url_for("index"))

    leaderboard = load_leaderboard()
    return render_template("index.html", leaderboard=leaderboard)


if __name__ == "__main__":
    app.run()
