"use client";

const EXAMPLES = [
  {
    label: "Threads 冒名",
    text: "我被告妨害名譽，但其實是有人在 Threads 上冒用我的照片和名字發文罵人，我根本沒做這些事，現在對方說要告我，我該怎麼辦？",
  },
  {
    label: "酒駕攔檢",
    text: "我昨晚聚餐後開車回家，在路口被警察攔下做酒測，結果吐氣值超標，警察說我涉及公共危險，想知道這會有什麼法律問題。",
  },
  {
    label: "二手車瑕疵",
    text: "我向車行買了一台二手車，交車不到一週引擎就出現重大故障，賣家卻說一切正常不願處理也不退款，我可以要求退車嗎？",
  },
  {
    label: "投資詐騙",
    text: "網路上認識的人推薦我一個投資平台，說保證獲利，我陸續匯了三十多萬過去，現在平台打不開、人也聯絡不上，應該是被詐騙了。",
  },
];

type Props = {
  value: string;
  loading: boolean;
  onChange: (value: string) => void;
  onAnalyze: () => void;
};

export function ScenarioInput({ value, loading, onChange, onAnalyze }: Props) {
  const canRun = value.trim().length >= 8 && !loading;

  function onKey(event: React.KeyboardEvent<HTMLTextAreaElement>) {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter" && canRun) onAnalyze();
  }

  return (
    <section className="panel" aria-label="Scenario input">
      <div className="panel-head">
        <div className="panel-title">
          <h2>情境輸入</h2>
          <span className="en">Scenario input</span>
        </div>
        <span className="panel-note">自由文字 · 50–500 字典型</span>
      </div>
      <div className="panel-body">
        <div className="ta-wrap">
          <textarea
            className="textarea"
            value={value}
            onChange={(event) => onChange(event.target.value)}
            onKeyDown={onKey}
            placeholder="用白話描述你的法律情境，例如：我酒後開車被警察攔下，酒測超標……"
            aria-label="中文法律情境"
          />
          <span className="charcount">{value.length} 字</span>
        </div>

        <div className="field-label">試試看 · Examples</div>
        <div className="examples">
          {EXAMPLES.map((example) => (
            <button
              key={example.label}
              className="ex-chip"
              type="button"
              onClick={() => onChange(example.text)}
            >
              <span className="glyph">›</span>
              {example.label}
            </button>
          ))}
        </div>

        <div className="actions">
          <button className="btn btn-primary" type="button" onClick={onAnalyze} disabled={!canRun}>
            {loading ? "分析中…" : "執行分析"}
            {!loading && <span className="kbd">⌘↵</span>}
          </button>
          <button className="btn btn-ghost" type="button" onClick={() => onChange("")}>
            清除
          </button>
        </div>
      </div>
    </section>
  );
}
