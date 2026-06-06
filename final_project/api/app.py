"""Flask API for the legal issue triage demo."""

from __future__ import annotations

import sys
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.model_service import ModelService


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)
    service = ModelService()

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok"})

    @app.post("/api/predict")
    def predict():
        payload = request.get_json(silent=True) or {}
        text = str(payload.get("text", "")).strip()
        if len(text) < 5:
            return jsonify({"status": "error", "message": "Please enter a longer Chinese legal scenario."}), 400
        return jsonify(service.predict(text))

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)
