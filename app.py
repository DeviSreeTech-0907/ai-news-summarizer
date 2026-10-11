
import os
import re
import requests
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()
app = Flask(__name__)

GNEWS_KEY = os.getenv("GNEWS_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

ai_client = InferenceClient(api_key=HF_TOKEN) if HF_TOKEN else None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/news")
def get_news():
    category = request.args.get("category", "general")
    allowed = {
        "general", "world", "nation", "business", "technology",
        "entertainment", "sports", "science", "health"
    }

    if category not in allowed:
        category = "general"

    if not GNEWS_KEY:
        return jsonify({"error": "Add GNEWS_API_KEY to your .env file."}), 400

    try:
        response = requests.get(
            "https://gnews.io/api/v4/top-headlines",
            params={
                "category": category,
                "lang": "en",
                "country": "in",
                "max": 10,
                "apikey": GNEWS_KEY
            },
            timeout=15
        )
        response.raise_for_status()
        return jsonify(response.json())

    except requests.RequestException:
        app.logger.exception("News API request failed")
        return jsonify({
            "error": "Unable to load news. Check your API key and internet connection."
        }), 502


@app.route("/api/search")
def search_news():
    query = request.args.get("q", "").strip()

    if not query:
        return jsonify({"articles": []})

    if not GNEWS_KEY:
        return jsonify({"error": "Add GNEWS_API_KEY to your .env file."}), 400

    try:
        response = requests.get(
            "https://gnews.io/api/v4/search",
            params={
                "q": query[:200],
                "lang": "en",
                "country": "in",
                "max": 10,
                "apikey": GNEWS_KEY
            },
            timeout=15
        )
        response.raise_for_status()
        return jsonify(response.json())

    except requests.RequestException:
        app.logger.exception("News search failed")
        return jsonify({"error": "News search failed. Please try again."}), 502


@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.get_json(silent=True) or {}
    article = data.get("article", "").strip()

    if not article:
        return jsonify({"error": "Please enter an article."}), 400

    if ai_client:
        try:
            result = ai_client.summarization(
                article[:10000],
                model="facebook/bart-large-cnn"
            )
            return jsonify({
                "summary": result.summary_text,
                "mode": "AI"
            })
        except Exception:
            app.logger.exception("AI summarization failed")

    sentences = re.split(r"(?<=[.!?])\s+", article)
    short_summary = " ".join(sentences[:2])

    return jsonify({
        "summary": short_summary,
        "mode": "Basic fallback"
    })


if __name__ == "__main__":
    app.run(debug=True)
