"use client";

import { useState } from "react";
import { Database, FlaskConical, ShieldCheck } from "lucide-react";
import { ExplanationPanel } from "@/components/ExplanationPanel";
import { ModelComparison } from "@/components/ModelComparison";
import { PredictionResults } from "@/components/PredictionResults";
import { ScenarioInput } from "@/components/ScenarioInput";
import { predictLegalIssues, type PredictResponse } from "@/lib/api";

export default function Home() {
  const [text, setText] = useState("");
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function analyze() {
    setLoading(true);
    setError(null);
    try {
      setResult(await predictLegalIssues(text));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="page">
      <div className="shell">
        <header className="topbar">
          <div>
            <p className="brand-kicker">NLP Final Project Demo</p>
            <h1 className="title">Chinese Legal Issue Triage</h1>
            <p className="subtitle">
              A focused text classification demo for routing everyday Chinese legal narratives into
              legal issue labels. It visualizes model outputs without generating legal advice.
            </p>
          </div>
          <div className="status-row" aria-label="Project status">
            <span className="badge badge-primary">
              <Database size={14} aria-hidden="true" />
              tw-legal-synthetic-qa
            </span>
            <span className="badge badge-primary">
              <FlaskConical size={14} aria-hidden="true" />
              Flask API
            </span>
            <span className="badge badge-warning">
              <ShieldCheck size={14} aria-hidden="true" />
              Not legal advice
            </span>
          </div>
        </header>

        <section className="grid">
          <ScenarioInput value={text} loading={loading} onChange={setText} onAnalyze={analyze} />
          <PredictionResults result={result} error={error} />
        </section>

        <section className="lower-grid">
          <ModelComparison result={result} />
          <ExplanationPanel result={result} />
        </section>
      </div>
    </main>
  );
}
