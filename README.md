# 🎬 StreamWave — Multimodal Sentiment Analysis & Recommendation Engine

![Subject](https://img.shields.io/badge/Subject-NLP_/_Multimodal_Classification-purple)
![Type](https://img.shields.io/badge/Type-Supervised_Learning_&_Deep_Learning-orange)
![Technology](https://img.shields.io/badge/AI_--_Transformers-DistilBERT-red)
![Status](https://img.shields.io/badge/Status-Portfolio_Ready-success)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0%2B-blue)


Moteur d'analyse de sentiment hybride pour la plateforme vidéo **StreamWave**. Le système combine du traitement automatique du langage naturel (NLP) sur les commentaires utilisateurs à l'aide de Transformers avec des métriques d'interaction (temps de visionnage, likes, partages) pour attribuer une note implicite de 1 à 5 et alimenter l'algorithme de recommandation.

---

## 📌 Vue d'ensemble

Plutôt que d'interrompre l'expérience utilisateur en lui demandant d'évaluer une vidéo, StreamWave calcule une note globale de satisfaction basée sur son comportement :
1. **Analyse NLP Avancée :** Extraction de la polarité contextuelle du commentaire via le modèle d'apprentissage profond DistilBERT (AI).
2. **Métriques comportementales :** Ratio de visionnage (`watch_time_pct`), mention J'aime (`like_dislike`), type de partage (`share_type`).
3. **Moteur de Recommandation :** Filtrage prédictif selon la note attribuée.

---

## 🛠 Stack Technique

* **Language :** Python 3.10+
* **AI & NLP Engine :** Hugging Face Transformers (`distilbert-base-uncased-finetuned-sst-2-english`)
* **Data Processing :** Pandas, NumPy
* **Machine Learning :** Scikit-Learn (`RandomForestClassifier`, `train_test_split`)
* **Production / MLOps :** Joblib (Sérialisation du modèle `.pkl`), Pytest (Tests unitaires)
* **Visualisation :** Matplotlib, Seaborn
* **Frontend Interface :** Dashboard interactif web complet (`streamlit`)

---

## 🛠️ Pipeline Machine Learning & Industrialisation

1. **Génération / Ingestion (`streamwave_data.csv`) :**
   * Traitement d'un jeu d'interactions simulé de 500 sessions utilisateur.
2. **Feature Extraction (NLP Deep Learning & Comportement) :**
   * Analyse sémantique via DistilBERT pour générer un `sentiment_score` linéaire continu (de -1.0 à +1.0).
   * Mesure de la longueur du texte (`comment_length`).
   * Consolidation avec les signaux d'engagement numériques.
3. **Modélisation & Sérialisation :**
   * Classification supervisée sur 5 classes de notes (1 à 5) via Random Forest.
   * Exportation du modèle entraîné au format de production `streamwave_model.pkl`.
4. **Déploiement applicatif & MLOps :**
   * Suite de tests unitaires automatisés pour valider la cohérence des prédictions.
   * Interface Web Streamlit chargeant à la volée le pipeline NLP et le modèle prédictif pour des inférences en temps réel.

---

## 📈 Métriques & Résultats

* **Accuracy Globale :** **91.00%** sur le jeu de test.
* **Recall Note 5/5 :** 93% (F1-score: 0.94).
* **Précision Note 1/5 :** 100% (F1-score: 1.00).

---

## 🎯 Analyse de la Robustesse

* **Excellence sur les extrêmes :** Le modèle est parfait sur la **note 1** (F1-score de 1.00) et excellent sur la **note 5** (0.94). Les signaux de frustration ou d'enthousiasme maximal sont très nets et faciles à isoler pour l'algorithme.
* **Nuance sur les notes moyennes :** Le modèle rencontre plus de difficultés sur la **note 3** (F1-score de 0.71) et la **note 4** (0.76). En IA, les zones grises et les comportements d'utilisateurs "tièdes" ou modérés présentent toujours des frontières de décision plus floues.

---

## 🏆 Importance des Features (Gini Importance)

1. **Sentiment Score (32.6%)** — La polarité NLP calculée par le Transformer reste le signal le plus discriminant.
2. **Watch Time % (28.7%)** — Le temps de visionnage confirme ou nuance le sentiment exprimé.
3. **Like / Dislike (18.5%)** — Signal binaire explicite fort.
4. **Share Type (11.9%)** — Le partage privé ou public reflète une forte adhésion.
5. **Comment Length (8.4%)** — Les avis très longs sont souvent corrélés aux notes extrêmes.

---

## 🔍 Évolution MLOps : Résolution du cas d'école (De VADER à DistilBERT)

Lors de la première itération du projet avec l'analyseur lexical VADER, une anomalie critique avait été détectée sur les expressions modernes de l'argot internet :
* **Commentaire testé :** `"This video is absolutely insane! Mindblown."` (Phrase ultra-positive).
* **Score VADER initial :** `-0.51 (Négatif)` en raison de l'interprétation statique et littérale du mot "insane".

### 🧠 La solution : Deep Learning & Contextualisation
Pour industrialiser le projet, le pipeline NLP a été migré vers **DistilBERT** (Hugging Face). Ce modèle basé sur l'architecture Transformer analyse le contexte global de la phrase plutôt que des mots isolés. Il a instantanément corrigé cette faille en attribuant un score de polarité de **1.00 (Positif Maximal)**.

### 📊 Robustesse du système Hybride Multimodal
L'intérêt majeur de cette approche réside dans sa complémentarité. Même si le pipeline NLP venait à rencontrer une ambiguïté textuelle, la structure multimodale du Random Forest sécurise l'expérience utilisateur : en croisant un texte avec **95% de watch time** et un **Like**, l'algorithme combine la puissance du Transformer et des signaux comportementaux pour prédire la note finale de **4/5**, déclenchant avec succès le statut de **Recommandation Prioritaire**.

---

## ⚙️ Logic Métier de Recommandation

| Note Prédite | Statut Système | Action de Recommandation |
| :--- | :--- | :--- |
| **4 / 5 – 5 / 5** | 🔥 Prioritaire | Proposée automatiquement après visionnage. |
| **3 / 5** | ⚠️ Secondaire | Affichée si épuisement du catalogue prioritaire. |
| **1 / 5 – 2 / 5** | ❌ Exclus | Masquée des recommandations automatiques. |

---

## 🚀 Installation & Lancement

1. **Installer les dépendances (y compris l'environnement AI/Deep Learning) :**
   ```bash
   pip3 install pandas matplotlib seaborn scikit-learn transformers torch streamlit joblib
   ```

2. **Générer le jeu de données d'entraînement :**
   ```bash
   python3 generate_data.py
   ```

3. **Exécuter le pipeline d'entraînement et exporter le modèle de production :**
   ```bash
   python3 script.py
   ```

4. **Lancer la suite de tests unitaires automatisés :**
   ```bash
   python3 test_model.py
   ```

5. **Lancer le tableau de bord web interactif Streamlit :**
   ```bash
   python3 -m streamlit run app.py
   ```