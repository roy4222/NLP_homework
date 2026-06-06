"use client";

import { useCallback, useEffect, useState } from "react";
import { ExplanationPanel } from "@/components/ExplanationPanel";
import { ModelComparison } from "@/components/ModelComparison";
import { PredictionResults } from "@/components/PredictionResults";
import { ScenarioInput } from "@/components/ScenarioInput";
import { predictLegalIssues, type PredictResponse } from "@/lib/api";

const PIPE_STEPS = [
  { zh: "斷詞 · jieba 法律詞典", en: "Tokenize" },
  { zh: "特徵向量化 · TF-IDF", en: "Vectorize" },
  { zh: "多標籤分類 · 模型推論", en: "Classify" },
];

function Masthead() {
  return (
    <header className="masthead">
      <div className="brand">
        <div className="brand-mark">
          <span />
        </div>
        <div className="brand-word">
          Lex<b>Tag</b>
        </div>
        <div className="brand-divider" />
        <div className="brand-tag">
          <div className="brand-tag-zh">法律議題前置分流</div>
          <div className="brand-sub">Legal Issue Triage · Multi-label NLP</div>
        </div>
      </div>
      <div className="masthead-meta">
        <div className="chip-row">
          <span className="chip">
            <span className="dot" style={{ background: "var(--m-tfidf)" }} />
            tw-legal-synthetic-qa
          </span>
          <span className="chip">
            <span className="dot" style={{ background: "var(--m-bert)" }} />
            tw-processed-law-article
          </span>
        </div>
        <div className="chip-row">
          <span className="chip" style={{ fontStyle: "normal" }}>
            Multi-label · 20 issues
          </span>
          <span className="chip">Flask + Next.js</span>
        </div>
      </div>
    </header>
  );
}

function LoadingPipeline({ stage }: { stage: number }) {
  return (
    <div className="pipeline">
      {PIPE_STEPS.map((step, i) => {
        const cls = i < stage ? "done" : i === stage ? "on" : "";
        return (
          <div key={step.en} className={`pipe-step ${cls}`}>
            <div className="pipe-num">{i < stage ? "✓" : i + 1}</div>
            <div className="pipe-label">
              {step.zh}
              <span className="en">{step.en}</span>
            </div>
            <div className="pipe-track">
              <div className="pipe-fill" />
            </div>
          </div>
        );
      })}
    </div>
  );
}

export default function Home() {
  const [text, setText] = useState("");
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [stage, setStage] = useState(0);

  useEffect(() => {
    if (!loading) return;
    const s1 = setTimeout(() => setStage(1), 300);
    const s2 = setTimeout(() => setStage(2), 620);
    return () => {
      clearTimeout(s1);
      clearTimeout(s2);
    };
  }, [loading]);

  const analyze = useCallback(async () => {
    if (text.trim().length < 8) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setStage(0);
    try {
      const data = await predictLegalIssues(text);
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
      setStage(0);
    }
  }, [text]);

  return (
    <main className="app">
      <Masthead />

      <div className="disclaimer" role="note">
        <span className="tag-en">Not legal advice</span>
        <span className="txt">
          本工具僅進行<b>法律議題分流</b>，幫助判斷情境「可能屬於哪類問題」；
          <b>不提供法律意見、不生成判決</b>，所附條文僅為查詢線索。
        </span>
      </div>

      <div className="grid-2">
        <ScenarioInput value={text} loading={loading} onChange={setText} onAnalyze={analyze} />
        {loading ? (
          <section className="panel" aria-label="Running">
            <div className="panel-head">
              <div className="panel-title">
                <h2>預測議題</h2>
                <span className="en">Predicted issues</span>
              </div>
              <span className="panel-note">推論中…</span>
            </div>
            <div className="panel-body">
              <LoadingPipeline stage={stage} />
            </div>
          </section>
        ) : (
          <PredictionResults result={result} error={error} />
        )}
      </div>

      <div className="full">
        <ModelComparison result={result} />
      </div>

      <div className="full">
        <ExplanationPanel result={result} />
      </div>
    </main>
  );
}
