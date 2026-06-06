# Deployment Guide

LexTag is split into two deployables:

| Part | What | Where |
|---|---|---|
| **Frontend** | Next.js static export (`web/out`) | **Cloudflare Workers** (static assets) |
| **Backend** | Flask + scikit-learn prediction API | **Any Docker host** (Render / Railway / Fly.io) |

> The Flask + scikit-learn backend **cannot** run on Cloudflare Workers (JS/WASM runtime). That is why the backend goes to a Python/Docker host and the frontend calls it cross-origin (CORS is already enabled).

Deploy the **backend first** so you know its public URL, then build + deploy the frontend pointing at it.

---

## 1. Backend → Render (Docker, free tier)

The `Dockerfile` rebuilds the TF-IDF+SVM model at build time from the committed
`data/train.jsonl` (no Hugging Face download needed), then serves with gunicorn.

**Option A — Render Blueprint (uses `final_project/render.yaml`):**
1. Push this repo to GitHub.
2. On render.com → **New → Blueprint** → pick the repo. It reads `render.yaml`.
3. Deploy. Note the URL, e.g. `https://lextag-api.onrender.com`.
4. Verify: open `https://<your-api>/api/health` → `{"status":"ok"}`.

**Option B — any Docker host (Railway / Fly.io / a VPS):**
```bash
cd final_project
docker build -t lextag-api .
docker run -p 8000:8000 lextag-api          # local test → http://localhost:8000/api/health
```
Railway/Fly: point the platform at `final_project/Dockerfile`. The container reads `$PORT`.

> Render's free plan sleeps after ~15 min idle; the first request after sleep takes ~50s (cold start). Fine for a demo — just warm it up before presenting.

---

## 2. Frontend → Cloudflare Workers (static assets)

Config is in `web/wrangler.jsonc` (serves `web/out`). The API URL is baked in at
**build time** via `NEXT_PUBLIC_API_BASE_URL`.

```bash
cd final_project/web
npm install                       # first time (installs wrangler locally)
npx wrangler login                # one-time browser OAuth to your Cloudflare account

# Build with the real backend URL, then deploy:
NEXT_PUBLIC_API_BASE_URL="https://lextag-api.onrender.com" npm run build
npx wrangler deploy
```

`wrangler deploy` prints the live URL, e.g. `https://lextag-demo.<account>.workers.dev`.

> Rebuild + redeploy whenever the backend URL changes (the URL is compiled into the
> static JS — it is not read at runtime).

---

## 3. Verify end-to-end

1. Open the Workers URL.
2. Click an example (e.g. 酒駕攔檢) → **執行分析**.
3. You should see predicted issues with confidence bars **and the full statute text**
   in the expandable law cards. If predictions fail, check that the backend `/api/health`
   is reachable and that you rebuilt the frontend with the correct `NEXT_PUBLIC_API_BASE_URL`.

---

## Notes

- **CORS:** the Flask app uses `flask-cors` allowing all origins — fine for a public demo.
  To lock it down, restrict origins in `api/app.py`.
- **Local dev** still works unchanged: `./start_demo.sh` (Flask on :5000 + static frontend),
  which builds with the default `http://127.0.0.1:5000`.
- **wrangler in WSL:** use the repo-local Linux build (`web/node_modules/.bin/wrangler`
  or `npx wrangler`). The Windows-global `wrangler` on the PATH does not work in WSL.
