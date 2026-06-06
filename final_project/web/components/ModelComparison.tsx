"use client";

import { useState } from "react";
import type { PredictResponse } from "@/lib/api";

const modelTabs = [
  { id: "rule", label: "Rule-based" },
  { id: "tfidf_svm", label: "TF-IDF/SVM" },
  { id: "bert", label: "BERT" },
];

type ModelComparisonProps = {
  result: PredictResponse | null;
};

export function ModelComparison({ result }: ModelComparisonProps) {
  const [active, setActive] = useState("rule");
  const predictions = result?.models?.[active] ?? [];

  return (
    <section className="card">
      <div className="card-header">
        <h2 className="card-title">Model comparison</h2>
        <p className="card-description">The demo is structured for comparing NLP methods.</p>
      </div>
      <div className="card-content">
        <div className="tabs" role="tablist" aria-label="Model comparison tabs">
          {modelTabs.map((tab) => (
            <button
              className={`tab-button ${active === tab.id ? "tab-button-active" : ""}`}
              key={tab.id}
              type="button"
              role="tab"
              aria-selected={active === tab.id}
              onClick={() => setActive(tab.id)}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {predictions.length > 0 ? (
          <div className="model-row">
            {predictions.map((prediction) => (
              <span className="badge badge-primary" key={`${active}-${prediction.id}`}>
                {prediction.name_zh} {Math.round(prediction.confidence * 100)}%
              </span>
            ))}
          </div>
        ) : (
          <p className="empty-state">
            {result
              ? "This model is not available yet; the interface reserves the comparison slot."
              : "Run a scenario to populate model outputs."}
          </p>
        )}
      </div>
    </section>
  );
}
