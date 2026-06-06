import { AlertTriangle, Scale } from "lucide-react";
import type { PredictResponse } from "@/lib/api";

type PredictionResultsProps = {
  result: PredictResponse | null;
  error: string | null;
};

export function PredictionResults({ result, error }: PredictionResultsProps) {
  if (error) {
    return (
      <section className="card" aria-live="polite">
        <div className="card-content">
          <div className="alert">
            <strong>Prediction failed.</strong> {error}
          </div>
        </div>
      </section>
    );
  }

  if (!result) {
    return (
      <section className="card">
        <div className="card-header">
          <h2 className="card-title">Prediction</h2>
          <p className="card-description">Run analysis to see issue labels and supporting laws.</p>
        </div>
        <div className="card-content empty-state">
          The output will compare model predictions and show a clear not-legal-advice disclaimer.
        </div>
      </section>
    );
  }

  return (
    <section className="card" aria-live="polite">
      <div className="card-header">
        <h2 className="card-title">Predicted legal issues</h2>
        <p className="card-description">Multi-label triage output for the submitted scenario.</p>
      </div>
      <div className="card-content">
        {result.explanation.needs_review ? (
          <div className="alert">
            <AlertTriangle size={18} aria-hidden="true" /> Needs review: the scenario may be too short
            or ambiguous. Treat labels as tentative.
          </div>
        ) : null}

        <div className="prediction-list">
          {result.final_prediction.length > 0 ? (
            result.final_prediction.map((prediction) => (
              <article className="prediction-item" key={prediction.id}>
                <div className="prediction-head">
                  <div>
                    <p className="issue-name">{prediction.name_zh}</p>
                    <p className="issue-en">{prediction.name_en}</p>
                  </div>
                  <span className="badge badge-primary">
                    {Math.round(prediction.confidence * 100)}%
                  </span>
                </div>
                <div className="progress" aria-label={`${prediction.name_en} confidence`}>
                  <div
                    className="progress-fill"
                    style={{ width: `${Math.round(prediction.confidence * 100)}%` }}
                  />
                </div>
                <div className="law-row">
                  {prediction.supporting_laws.map((law) => (
                    <span className="badge" key={`${prediction.id}-${law.law}-${law.article}`}>
                      <Scale size={13} aria-hidden="true" />
                      {law.law} §{law.article}
                    </span>
                  ))}
                </div>
              </article>
            ))
          ) : (
            <p className="empty-state">No issue label crossed the prediction threshold.</p>
          )}
        </div>

        <p className="small-copy" style={{ marginTop: 16 }}>
          {result.disclaimer}
        </p>
      </div>
    </section>
  );
}
