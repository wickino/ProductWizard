import os
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime

def train_and_save_models(df, model_dir="model", backup_dir="backup"):
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(backup_dir, exist_ok=True)

    # Rozdelenie podľa výrobcu
    manufacturers = df["manufacturer"].unique()

    for manufacturer in manufacturers:
        df_mfr = df[df["manufacturer"] == manufacturer]
        if len(df_mfr) < 3:
            continue  # preskočíme ak má príliš málo dát

        model_path = os.path.join(model_dir, f"{manufacturer.lower()}.pkl")

        # Zálohovanie starého modelu
        if os.path.exists(model_path):
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            backup_path = os.path.join(backup_dir, f"{manufacturer.lower()}_{timestamp}.pkl")
            os.rename(model_path, backup_path)

        # Trénovanie nového modelu
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', LogisticRegression(max_iter=1000))
        ])

        X = df_mfr["manufacturer"] + " " + df_mfr["description"]
        y = df_mfr["category"]

        pipeline.fit(X, y)
        joblib.dump(pipeline, model_path)
