import type { PredictResponse, SupportingLaw } from "@/lib/api";
import { LawCard } from "./LawCard";

type Props = {
  result: PredictResponse | null;
};

export function ExplanationPanel({ result }: Props) {
  const keywords = result?.explanation.matched_keywords ?? [];

  // Collect unique supporting laws across the final predictions.
  const lawRows: (SupportingLaw & { issue: string })[] = [];
  const seen = new Set<string>();
  if (result) {
    for (const prediction of result.final_prediction) {
      for (const law of prediction.supporting_laws) {
        const key = `${law.law}-${law.article}`;
        if (seen.has(key)) continue;
        seen.add(key);
        lawRows.push({ ...law, issue: prediction.name_zh });
      }
    }
  }

  return (
    <section className="panel" aria-label="Why these labels">
      <div className="panel-head">
        <div className="panel-title">
          <h2>判讀依據</h2>
          <span className="en">Why these labels</span>
        </div>
        <span className="panel-note">可解釋性線索</span>
      </div>
      <div className="panel-body">
        {!result ? (
          <div className="empty">
            <div className="ring">
              <span />
            </div>
            <div className="ttl">尚無依據</div>
            <div className="sub">分析後顯示命中的關鍵字、各模型的判讀方式與具體條文內容。</div>
          </div>
        ) : (
          <>
            <div className="field-label" style={{ marginTop: 0 }}>
              命中關鍵字 · Matched keywords
            </div>
            {keywords.length > 0 ? (
              <div className="kw-list">
                {keywords.map((keyword) => (
                  <span className="kw" key={keyword}>
                    {keyword}
                  </span>
                ))}
              </div>
            ) : (
              <p className="muted-line">未從本體論詞典命中明顯關鍵字。</p>
            )}

            <div className="field-label">各模型依據 · Evidence by model</div>
            <div className="evidence-method">
              <div className="ev-row">
                <span className="sw" style={{ background: "var(--m-rule)" }} />
                <div className="body">
                  <h4>
                    規則比對<span className="en">Rule-based</span>
                  </h4>
                  <p>別名／關鍵字字典命中即觸發，可直接指出命中詞，但無法捕捉隱含語意。</p>
                </div>
              </div>
              <div className="ev-row">
                <span className="sw" style={{ background: "var(--m-tfidf)" }} />
                <div className="body">
                  <h4>
                    TF-IDF + SVM<span className="en">Features</span>
                  </h4>
                  <p>以字元 n-gram 權重作為特徵，由 LinearSVC 的決策分數轉換為信心分數。</p>
                </div>
              </div>
            </div>

            <div className="field-label">相關條文 · Statute lookup</div>
            {lawRows.length > 0 ? (
              <div className="laws">
                {lawRows.map((law) => (
                  <LawCard key={`${law.law}-${law.article}`} law={law} />
                ))}
              </div>
            ) : (
              <p className="muted-line">本情境的預測議題沒有對應的條文線索。</p>
            )}
            <p className="lookup-note">
              ⚖ 條文為查詢線索，非個案法律意見；實際適用請洽專業律師或法律扶助。
            </p>
          </>
        )}
      </div>
    </section>
  );
}
