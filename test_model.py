import joblib
import pandas as pd

def test_model_predictions():
    # 1. Chargement du modèle de production
    model = joblib.load('streamwave_model.pkl')
    
    # 2. Cas de test 1 : Comportement idéal (doit donner une excellente note)
    good_user = pd.DataFrame([{
        'sentiment_score': 0.95,
        'comment_length': 40,
        'watch_time_pct': 98,
        'like_dislike': 1,
        'share_type': 2
    }])
    
    # 3. Cas de test 2 : Comportement toxique/négatif (doit donner une mauvaise note)
    bad_user = pd.DataFrame([{
        'sentiment_score': -0.95,
        'comment_length': 35,
        'watch_time_pct': 5,
        'like_dislike': 0,
        'share_type': 0
    }])
    
    pred_good = model.predict(good_user)[0]
    pred_bad = model.predict(bad_user)[0]
    
    # Assertions : On vérifie mathématiquement la cohérence
    assert pred_good >= 4, f"Erreur : Un utilisateur parfait ne devrait pas avoir la note {pred_good}"
    assert pred_bad <= 2, f"Erreur : Un détracteur ne devrait pas avoir la note {pred_bad}"
    print("\n✅ Tous les tests unitaires de cohérence du modèle sont validés avec succès !")

if __name__ == "__main__":
    test_model_predictions()
