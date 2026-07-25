import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline

# Configuration de la page Streamlit
st.set_page_config(page_title="StreamWave Moderator Dashboard", page_icon="🎬", layout="centered")

st.title("🎬 Moteur de Recommandation Multimodal StreamWave")
st.markdown("Prototype d'évaluation prédictive basé sur le comportement utilisateur et l'analyse de sentiment NLP (DistilBERT).")

# Chargement sécurisé des artefacts de modèle
@st.cache_resource
def load_production_artifacts():
    model = joblib.load('streamwave_model.pkl')
    nlp_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    return model, nlp_analyzer

try:
    model, nlp_analyzer = load_production_artifacts()
    st.success("✅ Modèles prédictifs chargés avec succès depuis la production !")
except FileNotFoundError:
    st.error("❌ Erreur : 'streamwave_model.pkl' introuvable. Veuillez exécuter 'python3 script.py' d'abord.")
    st.stop()

st.divider()

# AJOUT : Section Graphique d'analyse directement dans l'interface
st.subheader("📊 Métriques du Modèle en Production")
with st.expander("Visualiser l'importance des caractéristiques (Features Importance)"):
    feature_cols = ['sentiment_score', 'comment_length', 'watch_time_pct', 'like_dislike', 'share_type']
    # Extraction des importances calculées par le Random Forest sauvegardé
    importances = model.feature_importances_
    df_imp = pd.DataFrame({'Caractéristique': feature_cols, 'Importance': importances})
    df_imp = df_imp.sort_values(by='Importance', ascending=False)
    
    # Génération du graphique Matplotlib propre pour le dashboard
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.barplot(x='Importance', y='Caractéristique', data=df_imp, palette='Purples_r', ax=ax)
    ax.set_title("Poids des variables dans la décision de l'IA")
    st.pyplot(fig)

st.divider()
st.subheader("📥 Simuler une interaction utilisateur")

# Formulaire de saisie pour l'interface graphique
comment = st.text_area("Commentaire de l'utilisateur (en anglais) :", "")
watch_time = st.slider("Temps de visionnage (%) :", min_value=0, max_value=100, value=95)

col1, col2 = st.columns(2)
with col1:
    like_raw = st.radio("Interaction Bouton :", ["Mention J'aime (Like)", "Aucune / Je n'aime pas (Dislike)"])
    like = 1 if "J'aime" in like_raw else 0
with col2:
    share_raw = st.selectbox("Type de partage détecté :", ["Aucun partage", "Partage privé (Message)", "Partage public (Réseaux)"])
    share = 0 if "Aucun" in share_raw else (1 if "privé" in share_raw else 2)

if st.button("🚀 Analyser l'interaction et prédire"):
    # AJOUT : Gestion de l'erreur si le champ texte est vide
    if not comment.strip():
        st.warning("⚠️ Action requise : Veuillez entrer un commentaire avant de lancer l'analyse.")
    else:
        # Pipeline NLP temps réel
        with st.spinner("Analyse sémantique du texte en cours..."):
            nlp_result = nlp_analyzer(comment)
            sentiment_score = nlp_result[0]['score']
            if nlp_result[0]['label'] == 'NEGATIVE':
                sentiment_score = -sentiment_score
            comment_length = len(comment)

        # Préparation du vecteur pour le Random Forest
        input_data = pd.DataFrame([{
            'sentiment_score': sentiment_score,
            'comment_length': comment_length,
            'watch_time_pct': watch_time,
            'like_dislike': like,
            'share_type': share
        }])

        # Prédiction de la note
        predicted_rating = model.predict(input_data)[0]

        # Affichage des résultats graphiques
        st.divider()
        st.subheader("📊 Résultats de l'analyse")
        
        c1, c2 = st.columns(2)
        c1.metric(label="Score Polarité NLP", value=f"{sentiment_score:.2f}")
        c2.metric(label="Note Finale Estimée", value=f"⭐ {predicted_rating}/5")

        # Logique métier avec affichage stylisé (Badges)
        if predicted_rating >= 4:
            st.success(f"🔥 STATUS : RECOMMANDATION PRIORITAIRE ({predicted_rating}/5)")
            st.info("💡 Action système : Cette vidéo est immédiatement propulsée en tête des flux utilisateurs.")
        elif predicted_rating == 3:
            st.warning(f"⚠️ STATUS : RECOMMANDATION SECONDAIRE ({predicted_rating}/5)")
            st.info("💡 Action système : Vidéo gardée en réserve, diffusée uniquement en cas de pénurie de contenu.")
        else:
            st.error(f"❌ STATUS : CONTENU EXCLU ({predicted_rating}/5)")
            st.info("💡 Action système : Cette vidéo est retirée des algorithmes de recommandation automatique.")
