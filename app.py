from flask import Flask, render_template, request
from recommender import HybridRecommender
import os

app = Flask(__name__)

# Initialize recommender
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "movies.csv")
recommender = HybridRecommender(DATA_PATH)


@app.route("/", methods=["GET", "POST"])
def index():
    query_title = ""
    recommendations = []
    error_message = ""
    if request.method == "POST":
        query_title = request.form.get("movie_title", "").strip()
        if not query_title:
            error_message = "Please enter a movie title to get recommendations."
        else:
            recommendations = recommender.recommend(query_title, top_n=5)
            if not recommendations:
                error_message = (
                    "No close matches were found for that title. "
                    "Try a different or simpler movie name."
                )

    return render_template(
        "index.html",
        query_title=query_title,
        recommendations=recommendations,
        error_message=error_message,
        titles=recommender.all_titles(),
    )


if __name__ == "__main__":
    # For local testing only. In production, use a proper WSGI server.
    app.run(debug=True)
