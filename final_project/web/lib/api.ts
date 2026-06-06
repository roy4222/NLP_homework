export type SupportingLaw = {
  law: string;
  article: string;
  /** Full statute text, surfaced as a lookup clue (not legal advice). */
  text?: string;
  full_law?: string;
};

export type Prediction = {
  id: string;
  name_zh: string;
  name_en: string;
  confidence: number;
  supporting_laws: SupportingLaw[];
  evidence: string[];
};

/** One candidate issue scored across the trained models, for the comparison view. */
export type GridRow = {
  id: string;
  name_zh: string;
  name_en: string;
  rule: number; // 0 / 1 (keyword hit)
  tfidf_svm: number; // 0..1 confidence
};

export type PredictResponse = {
  input: string;
  status: string;
  final_prediction: Prediction[];
  models: Record<string, Prediction[]>;
  grid: GridRow[];
  explanation: {
    matched_keywords: string[];
    needs_review: boolean;
  };
  disclaimer: string;
};

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:5000";

export async function predictLegalIssues(text: string): Promise<PredictResponse> {
  const response = await fetch(`${API_BASE_URL}/api/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });

  if (!response.ok) {
    const payload = (await response.json().catch(() => ({}))) as { message?: string };
    throw new Error(payload.message ?? "Prediction failed");
  }

  return response.json() as Promise<PredictResponse>;
}
