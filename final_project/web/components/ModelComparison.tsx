"use client";

import { useEffect, useState } from "react";
import type { GridRow, PredictResponse } from "@/lib/api";

// Real headline metrics from report/results.md (test split).
const MODEL_META = {
  rule: { label: "Rule-based", zh: "規則比對", color: "var(--m-rule)", macroF1: 0.426, microF1: 0.6 },
  tfidf_svm: {
    label: "TF-IDF + SVM",
    zh: "傳統 ML",
    color: "var(--m-tfidf)",
    macroF1: 0.324,
    microF1: 0.76,
  },
} as const;

type ModelKey = keyof typeof MODEL_META;
const ORDER: ModelKey[] = ["rule", "tfidf_svm"];
const THRESHOLD = 0.5;

function valFor(row: GridRow, key: ModelKey): number {
  return key === "rule" ? (row.rule === 1 ? 1 : 0.03) : row.tfidf_svm;
}

function ConfidenceLineChart({ rows }: { rows: GridRow[] }) {
  const [drawn, setDrawn] = useState(false);
  useEffect(() => {
    const timer = setTimeout(() => setDrawn(true), 60);
    return () => clearTimeout(timer);
  }, []);

  const data = rows.slice(0, 6);
  const W = 680;
  const H = 250;
  const padL = 38;
  const padR = 16;
  const padT = 14;
  const padB = 58;
  const innerW = W - padL - padR;
  const innerH = H - padT - padB;
  const n = data.length;
  const xAt = (i: number) => padL + (n === 1 ? innerW / 2 : (innerW * i) / (n - 1));
  const yAt = (v: number) => padT + innerH * (1 - v);
  const yTicks = [0, 0.25, 0.5, 0.75, 1];

  return (
    <div className="chart-wrap">
      <div className="chart-head">
        <span className="lbl">信心趨勢 · Confidence by issue</span>
        <span className="hint">每條線為一種方法的逐議題信心；虛線 = 0.50 閾值</span>
      </div>
      <svg
        className="linechart"
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label="Per-issue confidence line chart"
      >
        {yTicks.map((v) => (
          <g key={v}>
            <line
              className={v === 0.5 ? "lc-thresh" : "lc-grid"}
              x1={padL}
              y1={yAt(v)}
              x2={W - padR}
              y2={yAt(v)}
            />
            <text className="lc-axis-lbl" x={padL - 8} y={yAt(v) + 3.5} textAnchor="end">
              {v.toFixed(2)}
            </text>
          </g>
        ))}
        <text className="lc-thresh-lbl" x={W - padR} y={yAt(0.5) - 5} textAnchor="end">
          threshold
        </text>

        {ORDER.map((key, mi) => {
          const pts = data.map((r, i) => `${xAt(i)},${yAt(valFor(r, key))}`).join(" ");
          return (
            <polyline
              key={key}
              className={`lc-line ${key} ${drawn ? "drawn" : ""}`}
              style={{ stroke: MODEL_META[key].color, transitionDelay: `${mi * 0.12}s` }}
              points={pts}
            />
          );
        })}

        {ORDER.map((key) =>
          data.map((r, i) => (
            <circle
              key={`${key}-${i}`}
              className={`lc-dot ${drawn ? "drawn" : ""}`}
              cx={xAt(i)}
              cy={yAt(valFor(r, key))}
              r={key === "rule" ? 3 : 3.6}
              fill={key === "rule" ? "var(--surface)" : MODEL_META[key].color}
              stroke={MODEL_META[key].color}
              strokeWidth={key === "rule" ? 2 : 1.5}
            />
          )),
        )}

        {data.map((r, i) => (
          <g key={`x-${i}`}>
            <text className="lc-xlbl" x={xAt(i)} y={H - padB + 22} textAnchor="middle">
              {r.name_zh.length > 6 ? `${r.name_zh.slice(0, 5)}…` : r.name_zh}
            </text>
            <text className="lc-xlbl-en" x={xAt(i)} y={H - padB + 38} textAnchor="middle">
              {r.name_en.length > 14 ? `${r.name_en.slice(0, 12)}…` : r.name_en}
            </text>
          </g>
        ))}
      </svg>
    </div>
  );
}

function Matrix({ rows }: { rows: GridRow[] }) {
  return (
    <table className="matrix">
      <thead>
        <tr>
          <th className="issue-col">議題 · Issue</th>
          {ORDER.map((key) => (
            <th key={key}>
              <span className="mhead">
                <span className="sw" style={{ background: MODEL_META[key].color }} />
                {MODEL_META[key].label}
              </span>
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr key={row.id}>
            <td>
              <div className="m-issue">
                <span className="zh">{row.name_zh}</span>
                <span className="en">{row.name_en}</span>
              </div>
            </td>
            <td>
              <div className="cell">
                <span className={`cell-flag ${row.rule === 1 ? "hit" : "miss"}`}>
                  {row.rule === 1 ? "✓" : "✕"}
                </span>
                <span className="cell-tag">{row.rule === 1 ? "keyword hit" : "miss"}</span>
              </div>
            </td>
            <td>
              <div className={`cell ${row.tfidf_svm < THRESHOLD ? "below" : ""}`}>
                <span className="cell-bar">
                  <i
                    style={{
                      width: `${Math.round(row.tfidf_svm * 100)}%`,
                      background: MODEL_META.tfidf_svm.color,
                    }}
                  />
                </span>
                <span
                  className="cell-val"
                  style={{
                    color: row.tfidf_svm < THRESHOLD ? "var(--muted)" : MODEL_META.tfidf_svm.color,
                  }}
                >
                  {row.tfidf_svm.toFixed(2)}
                </span>
              </div>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

type Props = {
  result: PredictResponse | null;
};

export function ModelComparison({ result }: Props) {
  const rows = result?.grid ?? [];

  return (
    <section className="panel" aria-label="Model comparison">
      <div className="panel-head" style={{ flexDirection: "column", alignItems: "stretch", gap: 14 }}>
        <div className="panel-title" style={{ justifyContent: "space-between", width: "100%" }}>
          <div style={{ display: "flex", alignItems: "baseline", gap: 10 }}>
            <h2>模型對照</h2>
            <span className="en">Model comparison</span>
          </div>
          <span className="panel-note">Rule-based · TF-IDF + SVM</span>
        </div>
        <div className="model-legend">
          {ORDER.map((key) => (
            <div className="model-stat" key={key}>
              <span className="sw" style={{ background: MODEL_META[key].color }} />
              <span className="nm">
                {MODEL_META[key].zh}
                <span className="en">{MODEL_META[key].label}</span>
              </span>
              <span className="f1">
                Macro-F1 <b>{MODEL_META[key].macroF1.toFixed(2)}</b>
              </span>
            </div>
          ))}
        </div>
      </div>
      <div className="panel-body">
        {!result ? (
          <div className="empty">
            <div className="ring">
              <span />
            </div>
            <div className="ttl">等待輸入</div>
            <div className="sub">執行分析後，這裡會逐一議題比較兩種方法的判斷與信心。</div>
          </div>
        ) : rows.length === 0 ? (
          <p className="muted-line">本情境沒有任何候選議題可供比較。</p>
        ) : (
          <>
            <ConfidenceLineChart key={result.input} rows={rows} />
            <Matrix rows={rows} />
            <p className="lookup-note">
              觀察重點：規則比對 (Rule-based) 只在命中關鍵字時觸發，<b>整體精確度較低</b>，
              但在罕見議題上反而 Macro-F1 較高 (0.43)；TF-IDF + SVM 的 Micro-F1 與精確度較好
              (Micro-F1 0.76)，但因稀有類別樣本太少而 Macro-F1 偏低 (0.32)。
              這正是類別不平衡下兩種指標彼此矛盾的真實情況。
            </p>
          </>
        )}
      </div>
    </section>
  );
}
