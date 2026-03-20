# BizGen AI — AI-Powered Business Report Generator

An AI-driven Streamlit web application that generates professional, investor-ready business feasibility reports by combining GPT-4 with live market data, financial indicators, news scraping, Reddit sentiment, and RAG (Retrieval-Augmented Generation).

---

## What It Does

You enter a business idea — the industry, target market, goal, and budget — and the app produces a full analyst-grade report by pulling from multiple real-world data sources simultaneously:

- **GPT-4** synthesizes everything into a structured, formal report
- **Google Trends** shows 12-month search interest for the industry
- **Yahoo Finance (ETF data)** shows how the sector has performed over the last 6 months
- **Industry growth CSV** provides static growth rate and competitor data
- **Investopedia** is scraped for an expert industry definition
- **Google News & financial sites** (Bloomberg, CNBC, Forbes, etc.) provide live headlines
- **Reddit** posts are fetched and indexed with FAISS for semantic context retrieval (RAG)
- **PDF export** lets you download the full report with embedded charts

---

## Report Structure (Full Mode)

1. Executive Market Summary
2. SWOT Analysis
3. Financial Forecast (3–6 month outlook)
4. Industry Scale & Growth Trends (TAM/SAM/SOM, CAGR)
5. Competitive Landscape
6. Tactical Recommendations (go-to-market strategy)
7. Pre-launch Readiness & Capital Planning
8. Final Assessment & Feasibility Verdict

---

## Project Structure

```
ai-report-generator/
├── app.py                   # Version 1 — original Streamlit app (no Reddit/RAG)
├── app2.py                  # Version 2 — full app with Reddit RAG, multi-page UI
├── try.py                   # Version 3 — further-refined variant of app2
├── news_scraper.py          # Scrapes Bloomberg, CNBC, Forbes, etc. for headlines
├── rag_vector_DB.py         # Builds FAISS vector store and retrieves relevant Reddit docs
├── reddit_rag_scraper.py    # Fetches Reddit posts via PRAW for a given keyword
├── requirements.txt         # All Python dependencies
├── industry_growth.csv      # Static dataset: industry names, growth rates, competitors
├── company_with_sectors.csv # Static dataset: company names mapped to sectors
└── .env                     # API keys (never commit this file)
```

### File Descriptions

| File | Purpose |
|---|---|
| `app.py` | Original version. Single-page Streamlit UI. Generates summary and full reports using GPT-4, Google Trends, ETF data, and news scraping. No Reddit integration. |
| `app2.py` | Current main version. Adds Reddit RAG pipeline, a multi-page UI (Home → Generated Report), visual tabs for each data source, and PDF download with charts. |
| `try.py` | Refined version of app2 with cleaner error handling and docstrings. Can be used as an alternative entry point. |
| `news_scraper.py` | Scrapes article links from major business news homepages, extracts full text using `newspaper3k`, infers sector from content using keyword matching and a company-sector CSV, and returns structured article data. |
| `rag_vector_DB.py` | Takes a list of text strings (Reddit posts), wraps them in LangChain `Document` objects, embeds them with OpenAI embeddings, and stores them in a FAISS index for similarity search. |
| `reddit_rag_scraper.py` | Uses the PRAW library to search Reddit (all subreddits) for posts matching a keyword. Returns post titles and body text for downstream RAG. |

---

## Setup

### 1. Prerequisites

- Python 3.11
- [Homebrew](https://brew.sh) (macOS) — needed for Cairo (PDF rendering dependency)

Install Cairo if not already installed:
```bash
brew install cairo pkg-config
```

### 2. Clone and create a virtual environment

```bash
git clone <your-repo-url>
cd ai-report-generator

python3.11 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the spaCy language model

```bash
python -m spacy download en_core_web_sm
```

### 5. Configure environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
REDDIT_USER_AGENT=your_app_name:v1.0 (by u/your_reddit_username)
```

> **Never commit your `.env` file.** Make sure `.env` is listed in `.gitignore`.

#### How to get API keys

**OpenAI:**
- Go to [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- Create a new secret key
- Your account needs access to GPT-4

**Reddit (PRAW):**
- Go to [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps)
- Click "create another app" → choose **script**
- Copy the client ID (under the app name) and the client secret

### 6. Prepare the data files

The app requires two CSV files in the project root:

**`industry_growth.csv`** — must have these columns:

```
Industry, GrowthRate, TopCompetitors
```

Example:
```csv
Industry,GrowthRate,TopCompetitors
Fintech,12.5,"Stripe, Square, Plaid"
Healthcare,9.3,"UnitedHealth, CVS, Teladoc"
Technology,15.2,"Apple, Microsoft, Google"
```

**`company_with_sectors.csv`** — must have these columns:

```
company_name, sector
```

Used for fuzzy-matching company names in news headlines to sector ETFs.

---

## Running the App

```bash
source venv/bin/activate
streamlit run app2.py
```

The app opens at `http://localhost:8501` in your browser.

To run the original simpler version:
```bash
streamlit run app.py
```

---

## How It Works — Data Flow

```
User Input (industry, market, goal, budget)
         │
         ├─── GPT-4 ──────────────────────────────────────────────────┐
         ├─── Google Trends (pytrends) ──── 12-month interest          │
         ├─── Yahoo Finance (yfinance) ──── ETF 6-month price          │
         ├─── industry_growth.csv ────────── growth rate + competitors  ├──► GPT-4 Prompt
         ├─── Investopedia (scraped) ─────── expert definition          │
         ├─── Google News (scraped) ──────── top 5 headlines            │
         ├─── Financial news sites ───────── Bloomberg/CNBC/Forbes      │
         └─── Reddit (PRAW + FAISS RAG) ─── community sentiment ───────┘
                                                      │
                                               GPT-4 Response
                                                      │
                                      ┌───────────────┴────────────────┐
                                 Report Text                     Visual Charts
                              (8-section report)         (Trends / ETF / Growth)
                                      │                               │
                                      └───────────────┬───────────────┘
                                                 PDF Export
```

### ETF Mapping

Industries are mapped to sector ETFs for market performance data. Examples:

| Industry | ETF |
|---|---|
| Technology | XLK |
| Fintech | FINX |
| Healthcare | XLV |
| Cybersecurity | CIBR |
| Renewable Energy | ICLN |
| Blockchain | BLOK |
| Semiconductors | SOXX |
| Aerospace | ITA |

If an exact match isn't found, the app uses fuzzy matching (70% similarity threshold) against the ETF map and the company-sector CSV.

### RAG Pipeline (Reddit)

1. Reddit posts are fetched for the input industry keyword via PRAW
2. Each post (title + body) is wrapped as a LangChain `Document`
3. OpenAI embeddings are generated and stored in a FAISS in-memory vector index
4. The top 3 most semantically relevant posts are retrieved and injected into the GPT-4 prompt

---

## Dependencies

| Package | Purpose |
|---|---|
| `openai` | GPT-4 API calls |
| `langchain-openai` | OpenAI embeddings for RAG |
| `langchain-community` | FAISS vector store integration |
| `langchain-core` | LangChain base types (Document, etc.) |
| `faiss-cpu` | Fast vector similarity search |
| `streamlit` | Web UI framework |
| `yfinance` | Yahoo Finance ETF price data |
| `pytrends` | Google Trends API wrapper |
| `praw` | Reddit API client |
| `newspaper3k` | Article text extraction from URLs |
| `beautifulsoup4` | HTML parsing for web scraping |
| `spacy` | NLP — keyword and named entity extraction |
| `textblob` | Sentiment analysis |
| `pandas` | Data manipulation and CSV handling |
| `matplotlib` | Chart rendering |
| `xhtml2pdf` | HTML-to-PDF export |
| `python-dotenv` | Load environment variables from `.env` |

---

## Known Limitations

- Google Trends may rate-limit requests; there is a 2-second delay built in
- Investopedia scraping may break if their HTML structure changes
- Financial news sites (Bloomberg, FT) may block scraping; articles may be paywalled
- The `industry_growth.csv` is static — growth rates reflect whatever data you populate it with
- ETF mapping only covers ~20 predefined sectors; niche industries may not match
- Reddit RAG uses an in-memory FAISS index — it is rebuilt on every report generation

---

## Security Notes

- **Do not commit `.env`** — add it to `.gitignore` immediately
- **Rotate your API keys** if they have ever been pushed to a public repository
- The app makes outbound HTTP requests to Investopedia, Google News, and financial news sites — respect their terms of service and rate limits
