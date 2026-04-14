# Kenya Presidential Dashboard — Setup Guide

## Project structure
```
kenya-dashboard/
├── .env                  ← your secrets (never commit this)
├── requirements.txt
├── 01_supabase_schema.sql
├── 02_etl.py
└── 03_app.py
```

## Step 1 — Create your .env file
```
SUPABASE_URL=https://xxxx.supabase.co
SUPABASE_KEY=your-anon-key-from-supabase-dashboard
GEMINI_API_KEY=your-gemini-api-key
```

Get Supabase keys: Project Settings → API → anon public key
Get Gemini key: https://aistudio.google.com/app/apikey (free)

## Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

## Step 3 — Run schema in Supabase
1. Open Supabase dashboard → SQL Editor
2. Paste entire contents of 01_supabase_schema.sql
3. Click Run

## Step 4 — Run ETL to pull World Bank data
```bash
python 02_etl.py
```
This fetches ~40 indicators for Kenya 1998–2024.
Takes about 2–3 minutes. Run once, then weekly to refresh.

## Step 5 — Launch the app
```bash
streamlit run 03_app.py
```

## Step 6 — Deploy free on Streamlit Cloud
1. Push project to GitHub (add .env to .gitignore!)
2. Go to https://share.streamlit.io
3. Connect your GitHub repo
4. Add secrets in Streamlit Cloud dashboard (same as .env)
5. Deploy → get a public URL

---

## Adding more data sources (optional next steps)

### World Happiness Report (life satisfaction)
- Download CSV from: https://worldhappiness.report/data/
- Add a column: source = 'Happiness Report'
- Load with pandas and upsert same indicators table

### Transparency International (corruption index)
- Download from: https://www.transparency.org/en/cpi
- Indicator code: TI.CPI.PERCN (or load manually)

### KNBS (Kenya National Bureau of Statistics)
- https://www.knbs.or.ke
- Download poverty and unemployment CSVs
- Parse and upsert same way as World Bank data

---

## Supabase free tier limits
- 500 MB storage (more than enough)
- 2 GB bandwidth/month
- Unlimited API calls
- No credit card required
