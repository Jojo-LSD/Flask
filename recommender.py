import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model
import joblib
import math
import gdown

def load_npy_from_gdrive(file_id):
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "/tmp/temp.npy"
    gdown.download(url, output, quiet=False)
    return np.load(output, allow_pickle=True)

def gdrive_download_link(file_id):
    return f"https://drive.google.com/uc?export=download&id={file_id}"

cv_vectors = load_npy_from_gdrive("1V8Rv1ko9xlhjUhzHAeysCIkncYBP-yDH")

# Chargement des ressources
vectorizer = joblib.load("tfidf_vectorizer.joblib")
model = load_model("DL.keras")
encoded_cv = model.predict(cv_vectors)
model.summary()

# Chargement du CSV avec les colonnes: public_identifier, first_name, last_name
df_cv = pd.read_csv("Les_cv.csv")

def recommend_from_offer(offer_text: str, top_n: int = 5):
    print(f"\n📥 Offre reçue : '{offer_text}'")

    # Vectorisation + encodage
    X_offer = vectorizer.transform([offer_text]).toarray()
    encoded_offer = model.predict(X_offer)

    # Calcul des similarités
    similarities = cosine_similarity(encoded_offer, encoded_cv)[0]
    top_indices = np.argsort(similarities)[::-1][:top_n]
    top_scores = similarities[top_indices]

    print("🎯 Profils recommandés :")
    results = []
    for i, (idx, score) in enumerate(zip(top_indices, top_scores), start=1):
        first_name = df_cv["first_name"].iloc[idx]
        last_name = df_cv["last_name"].iloc[idx]
        public_id = df_cv["public_identifier"].iloc[idx]
        title = df_cv["occupation"].iloc[idx]
        languages = df_cv["languages"].iloc[idx]

        # Construction du lien LinkedIn, sauf si identifiant manquant
        linkedin_url = f"https://www.linkedin.com/in/{public_id}" if pd.notna(public_id) else ""

        if isinstance(title, float) and math.isnan(title):
            title= linkedin_url

        print(f"{i}. {first_name} {last_name} | {linkedin_url} | Similarité: {score:.4f} | Job Title: {title}")
        print(languages)

        results.append({
            "id": int(idx),
            "score": float(score),
            "full_name": f"{first_name} {last_name}",
            "linkedin": linkedin_url,
            "Job Title": title,
            "languages": languages,
        })

    return results