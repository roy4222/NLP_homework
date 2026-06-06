import type { PredictResponse } from "@/lib/api";
import { LawCard } from "./LawCard";

type Props = {
  result: PredictResponse | null;
  error: string | null;
};

function EmptyState({ title, sub }: { title: string; sub: string }) {
  return (
    <div className="empty">
      <div className="ring">
        <span />
      </div>
      <div className="ttl">{title}</div>
      <div className="sub">{sub}</div>
    </div>
  );
}

export function PredictionResults({ result, error }: Props) {
  if (error) {
    return (
      <section className="panel" aria-label="Predicted issues" aria-live="polite">
        <div className="panel-head">
          <div className="panel-title">
            <h2>預測議題</h2>
            <span className="en">Predicted issues</span>
          </div>
        </div>
        <div className="panel-body">
          <div className="error-box">
            <h3>分析失敗 · Prediction failed</h3>
            <p>{error}（請確認後端 Flask API 是否在 http://127.0.0.1:5000 執行）</p>
          </div>
        </div>
      </section>
    );
  }

  if (!result) {
    return (
      <section className="panel" aria-label="Predicted issues">
        <div className="panel-head">
          <div className="panel-title">
            <h2>預測議題</h2>
            <span className="en">Predicted issues</span>
          </div>
        </div>
        <div className="panel-body">
          <EmptyState
            title="尚未分析"
            sub="輸入情境並按「執行分析」，即可看到多標籤議題、信心分數與具體條文內容。"
          />
        </div>
      </section>
    );
  }

  const preds = result.final_prediction;

  return (
    <section className="panel" aria-label="Predicted issues" aria-live="polite">
      <div className="panel-head">
        <div className="panel-title">
          <h2>預測議題</h2>
          <span className="en">Predicted issues</span>
        </div>
        <span className="panel-note">融合 Rule + TF-IDF·SVM · 取前 5 名</span>
      </div>
      <div className="panel-body">
        {result.explanation.needs_review ? (
          <div className="needs-review">
            <strong style={{ whiteSpace: "nowrap" }}>⚠ 需人工複核</strong>
            <span>情境可能太短或語意模糊，標籤僅供初步參考，請以人工判斷為準。</span>
          </div>
        ) : null}

        {preds.length === 0 ? (
          <EmptyState
            title="無高於閾值的議題"
            sub="沒有任何議題達到判定門檻。試著補充更多情境細節。"
          />
        ) : (
          <div className="pred-list">
            {preds.map((prediction, index) => {
              const pct = Math.round(prediction.confidence * 100);
              return (
                <article className="pred" key={prediction.id}>
                  <div className="pred-top">
                    <div className="pred-name">
                      <span className="pred-rank">{String(index + 1).padStart(2, "0")}</span>
                      <div>
                        <h3>{prediction.name_zh}</h3>
                        <span className="en">{prediction.name_en}</span>
                      </div>
                    </div>
                    <div className="conf-num">
                      {pct}
                      <small>%</small>
                    </div>
                  </div>
                  <div className="bar">
                    <div className="bar-fill" style={{ width: `${pct}%` }} />
                  </div>
                  <div className="laws">
                    {prediction.supporting_laws.map((law) => (
                      <LawCard key={`${prediction.id}-${law.law}-${law.article}`} law={law} />
                    ))}
                  </div>
                </article>
              );
            })}
          </div>
        )}
        <p className="lookup-note">{result.disclaimer}</p>
      </div>
    </section>
  );
}
