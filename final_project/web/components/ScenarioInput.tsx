"use client";

import { Eraser, Play, Wand2 } from "lucide-react";

const examples = [
  {
    label: "酒駕",
    text: "我酒後開車被警察攔下，酒測超標，想知道可能涉及什麼法律問題。",
  },
  {
    label: "妨害名譽",
    text: "有人在 Threads 上冒用我的照片發文罵人，現在對方說要告我妨害名譽。",
  },
  {
    label: "買賣瑕疵",
    text: "我買到的二手車交車後才發現引擎有重大問題，賣家不願意處理。",
  },
];

type ScenarioInputProps = {
  value: string;
  loading: boolean;
  onChange: (value: string) => void;
  onAnalyze: () => void;
};

export function ScenarioInput({
  value,
  loading,
  onChange,
  onAnalyze,
}: ScenarioInputProps) {
  return (
    <section className="card" aria-labelledby="scenario-title">
      <div className="card-header">
        <h2 className="card-title" id="scenario-title">
          Scenario input
        </h2>
        <p className="card-description">
          Paste or type an everyday Chinese legal situation. The demo predicts issue labels only.
        </p>
      </div>
      <div className="card-content">
        <textarea
          className="textarea"
          aria-label="Chinese legal scenario"
          value={value}
          onChange={(event) => onChange(event.target.value)}
          placeholder="請輸入一段中文法律情境，例如：我酒後開車被警察攔下..."
        />
        <div className="chips" aria-label="Example scenarios">
          {examples.map((example) => (
            <button
              className="button button-secondary"
              key={example.label}
              type="button"
              onClick={() => onChange(example.text)}
            >
              <Wand2 size={16} aria-hidden="true" />
              {example.label}
            </button>
          ))}
        </div>
        <div className="actions">
          <button
            className="button button-primary"
            type="button"
            onClick={onAnalyze}
            disabled={loading || value.trim().length < 5}
          >
            <Play size={16} aria-hidden="true" />
            {loading ? "Analyzing..." : "Analyze"}
          </button>
          <button className="button button-secondary" type="button" onClick={() => onChange("")}>
            <Eraser size={16} aria-hidden="true" />
            Clear
          </button>
        </div>
      </div>
    </section>
  );
}
