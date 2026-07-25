import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from transformers import pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("🔄 Initialisation du modèle NLP DistilBERT (Hugging Face)...")
# Pipeline NLP moderne (Deep Learning) pour l'analyse de sentiment
nlp_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# ==========================================
# ÉTAPE 2 : PIPELINE NLP & FEATURE EXTRACTION (DEEP LEARNING)
# ==========================================

def extract_nlp_features(df):
    """
    Extrait le score de sentiment finetuné de DistilBERT (-1 à +1)
    et la longueur du commentaire.
    """
    print("🧠 Analyse des commentaires avec DistilBERT...")
    sentiment_scores = []
    
    for text in df['comment_text'].astype(str):
        # Analyse de la phrase
        result = nlp_analyzer(text)[0]
        # DistilBERT renvoie 'POSITIVE' ou 'NEGATIVE' avec un score de confiance (0 à 1)
        # On convertit cela en une échelle linéaire continue de -1.0 à +1.0
        score = result['score']
        if result['label'] == 'NEGATIVE':
            score = -score
        sentiment_scores.append(score)
        
    df['sentiment_score'] = sentiment_scores
    df['comment_length'] = df['comment_text'].apply(lambda text: len(str(text)))
    return df

# ==========================================
# ÉTAPE 3 : ASSEMBLAGE, ENTRAÎNEMENT & EXPORT MLOPS
# ==========================================

def train_and_export_model(df):
    feature_cols = ['sentiment_score', 'comment_length', 'watch_time_pct', 'like_dislike', 'share_type']
    X = df[feature_cols]
    y = df['final_rating']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("🌲 Entraînement du classifieur Random Forest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Évaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print("\n==================================================")
    print(f"🎯 Précision Globale du Modèle (Accuracy) : {acc * 100:.2f}%")
    print("==================================================")
    print("\n--- Rapport de Classification ---")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Sauvegarde des artefacts pour la production
    print("💾 Sauvegarde du modèle dans 'streamwave_model.pkl'...")
    joblib.dump(model, 'streamwave_model.pkl')
    print("✅ Modèle exporté avec succès et prêt pour la production !")

    # Génération des visuels du portfolio
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=[1,2,3,4,5], yticklabels=[1,2,3,4,5])
    plt.title('StreamWave — Matrice de Confusion (DistilBERT + RF)')
    plt.xlabel('Note Prédite')
    plt.ylabel('Note Réelle')
    plt.savefig('confusion_matrix_sentiment.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    df = pd.read_csv("streamwave_data.csv")
    df = extract_nlp_features(df)
    train_and_export_model(df)
