import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class HybridRecommender:
    """
    A simple hybrid-style recommender that combines
    content features (genres + description) into a single
    TF-IDF feature space and uses cosine similarity to
    recommend similar movies.
    """

    def __init__(self, csv_path: str):
        self.movies = pd.read_csv(csv_path)
        # Create a combined text field
        self.movies["combined"] = (
            self.movies["title"].fillna("") + " " +
            self.movies["genres"].fillna("") + " " +
            self.movies["description"].fillna("")
        )

        # Build TF-IDF matrix
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.movies["combined"])
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)

    def _get_movie_index(self, title: str):
        """
        Fuzzy match movie title by simple case-insensitive containment.
        Returns the index of the first best match or None.
        """
        if not title:
            return None

        title_lower = title.strip().lower()
        # Exact match first
        exact_matches = self.movies[self.movies["title"].str.lower() == title_lower]
        if not exact_matches.empty:
            return int(exact_matches.index[0])

        # Then partial match
        contains_matches = self.movies[self.movies["title"].str.lower().str.contains(title_lower)]
        if not contains_matches.empty:
            return int(contains_matches.index[0])

        return None

    def recommend(self, title: str, top_n: int = 5):
        """
        Recommend top_n movies similar to the given title.
        Returns a list of dicts with movie information and a short explanation.
        """
        idx = self._get_movie_index(title)
        if idx is None:
            return []

        similarity_scores = list(enumerate(self.similarity_matrix[idx]))
        # Sort by similarity, excluding the movie itself
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        similarity_scores = [s for s in similarity_scores if s[0] != idx]

        recommendations = []
        for i, score in similarity_scores[:top_n]:
            row = self.movies.iloc[i]
            explanation = (
                f"Recommended because it is similar to '{self.movies.iloc[idx]['title']}' "
                f"in terms of genres and description."
            )
            recommendations.append(
                {
                    "movie_id": int(row["movie_id"]),
                    "title": row["title"],
                    "genres": row["genres"],
                    "description": row["description"],
                    "score": float(score),
                    "explanation": explanation,
                }
            )
        return recommendations

    def all_titles(self):
        return list(self.movies["title"].values)
