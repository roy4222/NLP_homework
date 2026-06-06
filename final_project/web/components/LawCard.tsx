"use client";

import { useState } from "react";
import type { SupportingLaw } from "@/lib/api";

type Props = {
  law: SupportingLaw;
};

/** Renders a supporting statute as a card showing the actual article text. */
export function LawCard({ law }: Props) {
  const [open, setOpen] = useState(false);
  const text = law.text?.trim() ?? "";
  const long = text.length > 90;

  return (
    <div className="law-card">
      <div className="law-card-head">
        <span className="code">
          {law.law} §{law.article}
        </span>
        {law.full_law && law.full_law !== law.law ? (
          <span className="full">{law.full_law}</span>
        ) : null}
      </div>
      {text ? (
        <>
          <p className={`law-card-text${long && !open ? " clamp" : ""}`}>{text}</p>
          {long ? (
            <button type="button" className="law-card-toggle" onClick={() => setOpen((v) => !v)}>
              {open ? "收合條文 ▲" : "展開全文 ▼"}
            </button>
          ) : null}
        </>
      ) : (
        <p className="law-card-text">（查無條文內容）</p>
      )}
    </div>
  );
}
