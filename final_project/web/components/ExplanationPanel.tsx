import { KeyRound } from "lucide-react";
import type { PredictResponse } from "@/lib/api";

type ExplanationPanelProps = {
  result: PredictResponse | null;
};

export function ExplanationPanel({ result }: ExplanationPanelProps) {
  const keywords = result?.explanation.matched_keywords ?? [];

  return (
    <section className="card">
      <div className="card-header">
        <h2 className="card-title">Why these labels?</h2>
        <p className="card-description">
          Shows interpretable evidence used by the current demo pipeline.
        </p>
      </div>
      <div className="card-content">
        {keywords.length > 0 ? (
          <>
            <p className="small-copy">Matched keywords from the ontology:</p>
            <div className="keyword-list">
              {keywords.map((keyword) => (
                <span className="badge" key={keyword}>
                  <KeyRound size={13} aria-hidden="true" />
                  {keyword}
                </span>
              ))}
            </div>
          </>
        ) : (
          <p className="empty-state">
            The explanation area will show matched keywords, TF-IDF features, or citation mappings.
          </p>
        )}
      </div>
    </section>
  );
}
