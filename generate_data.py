import random
import pandas as pd

# Fixer la graine aléatoire pour la reproductibilité
random.seed(42)

# Exemples de commentaires représentatifs
POSITIVE_COMMENTS = [
    "Awesome video! Loved the editing and content.",
    "This was super helpful, thanks for sharing!",
    "Great work, keep making videos like this!",
    "Best video I have seen all week, absolutely fantastic.",
    "Amazing insights, really enjoyed watching this.",
    "Super interesting topic, well explained!",
]

NEUTRAL_COMMENTS = [
    "Okay video, nothing special though.",
    "It was fine, a bit long in the middle.",
    "Decent content, could be improved.",
    "Not bad, but I have seen better explanations.",
    "Just an average video on this subject.",
]

NEGATIVE_COMMENTS = [
    "Terrible quality, total waste of time.",
    "I really hated this video, super boring.",
    "Incorrect information, do not watch this.",
    "Bad audio and very dry content.",
    "Horrible video, completely useless.",
]


def generate_streamwave_dataset(filename="streamwave_data.csv", n_samples=500):
    data = []

    for _ in np_range if "np_range" in globals() else range(n_samples):
        # 1. Tirage du type de sentiment initial pour la cohérence
        sentiment_type = random.choices(
            ["positive", "neutral", "negative"], weights=[0.4, 0.3, 0.3]
        )[0]

        if sentiment_type == "positive":
            comment = random.choice(POSITIVE_COMMENTS)
            watch_time_pct = random.randint(60, 100)
            like_dislike = random.choices([1, 0], weights=[0.9, 0.1])[0]
            share_type = random.choices([0, 1, 2], weights=[0.3, 0.4, 0.3])[0]
        elif sentiment_type == "neutral":
            comment = random.choice(NEUTRAL_COMMENTS)
            watch_time_pct = random.randint(30, 70)
            like_dislike = random.choices([1, 0], weights=[0.5, 0.5])[0]
            share_type = random.choices([0, 1, 2], weights=[0.6, 0.3, 0.1])[0]
        else:
            comment = random.choice(NEGATIVE_COMMENTS)
            watch_time_pct = random.randint(5, 40)
            like_dislike = random.choices([1, 0], weights=[0.1, 0.9])[0]
            share_type = random.choices([0, 1, 2], weights=[0.8, 0.15, 0.05])[0]

        # 2. Calcul du score implicite pour attribuer une note globale cohérente (1 à 5)
        # Pondération : Watch Time (35%), Like (25%), Share (20%), Sentiment (20%)
        sentiment_val = (
            1.0
            if sentiment_type == "positive"
            else (0.0 if sentiment_type == "neutral" else -1.0)
        )

        score = (
            (watch_time_pct / 100.0) * 3.5
            + (like_dislike * 2.5)
            + (share_type * 1.0)
            + (sentiment_val * 1.5)
        )

        # Mappage vers une note de 1 à 5
        if score >= 5.5:
            final_rating = 5
        elif score >= 3.8:
            final_rating = 4
        elif score >= 2.2:
            final_rating = 3
        elif score >= 0.8:
            final_rating = 2
        else:
            final_rating = 1

        data.append(
            {
                "comment_text": comment,
                "watch_time_pct": watch_time_pct,
                "like_dislike": like_dislike,
                "share_type": share_type,
                "final_rating": final_rating,
            }
        )

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(
        f"✅ Dataset '{filename}' généré avec succès ({n_samples} lignes) !"
    )


if __name__ == "__main__":
    generate_streamwave_dataset()