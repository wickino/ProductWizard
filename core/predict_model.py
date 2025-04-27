import os
import joblib

def predict_with_manufacturer(manufacturer, description, model_dir="model", return_top2=False):
    model_file = os.path.join(model_dir, f"{manufacturer.lower()}.pkl")

    if not os.path.exists(model_file):
        return None  # model pre výrobcu neexistuje

    model = joblib.load(model_file)

    if return_top2:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba([manufacturer + " " + description])[0]
            classes = model.classes_
            top_indices = probs.argsort()[-2:][::-1]
            top_preds = [classes[top_indices[0]]]

            if len(top_indices) > 1:
                top_preds.append(classes[top_indices[1]])
            return top_preds
        else:
            # fallback ak model nemá predict_proba
            pred = model.predict([manufacturer + " " + description])
            return [pred[0], ""]
    else:
        pred = model.predict([manufacturer + " " + description])
        return pred[0]
