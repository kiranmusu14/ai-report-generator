# BizGen AI — AI-Powered Business Report Generator

An AI-driven Streamlit web application that generates professional, investor-ready business feasibility reports by combining GPT-4 with live market data, financial indicators, news scraping, Reddit community intelligence, and RAG (Retrieval-Augmented Generation).

---

## What It Does

Enter a business idea — the industry, target market, goal, and budget — and the app produces a full analyst-grade business plan by pulling from multiple real-world data sources simultaneously:

- **GPT-4** synthesizes everything into a 10-section structured, actionable report specific to your idea
- **Google Trends** shows 12-month search interest for the industry
- **Yahoo Finance (ETF data)** shows live sector performance and 5-year historical growth
- **Live growth projection** uses ETF historical YoY returns + GPT-4 to forecast the next 3 years
- **Investopedia** is scraped for an expert industry definition
- **Google News & financial sites** (Bloomberg, CNBC, Forbes, etc.) provide live headlines
- **Reddit** posts are fetched using a business-specific query, indexed with FAISS, and summarised by GPT-4 for community sentiment
- **PDF export** generates an in-memory downloadable report

---

## Report Structure (Full Mode — 10 Sections)

1. **Business Concept Overview** — what the business does, the problem it solves, the value proposition
2. **Market Opportunity** — TAM / SAM / SOM estimates, market trends, growth rate
3. **Business Model & Revenue Streams** — revenue model, pricing, CAC, LTV, gross margin
4. **Financial Projections (12 months)** — budget allocation, month-by-month targets, breakeven, ROI
5. **Competitive Landscape** — named competitors, their weaknesses, your differentiators
6. **Go-to-Market Strategy** — first 100 customers plan, acquisition channels, marketing budget split
7. **90-Day Execution Roadmap** — week-by-week actions from validation to launch
8. **Tech Stack & Operational Requirements** — recommended tools, team, APIs, compliance
9. **Risks & Mitigation** — top 4 specific risks with mitigation plans
10. **Final Verdict** — feasibility score (X/10), critical success factors, next 3 steps

---

## Project Structure

```
ai-report-generator/
├── app2.py                  # Main Streamlit app — entry point
├── news_scraper.py          # Scrapes Bloomberg, CNBC, Forbes, etc. for headlines
├── rag_vector_DB.py         # FAISS vector store builder and retriever
├── reddit_rag_scraper.py    # Reddit scraper using PRAW with goal-specific queries
├── requirements.txt         # All Python dependencies
├── industry_growth.csv      # Industry names, growth rates, and key competitors
├── company_with_sectors.csv # Company names mapped to sectors (for ETF matching)
└── .env                     # API keys — never commit this file
```

### File Descriptions

| File | Purpose |
|---|---|
| `app2.py` | Main application. Multi-page Streamlit UI (Home → Report). Handles all data fetching, GPT-4 report generation, chart rendering, and in-memory PDF export. Session state caching prevents re-generation on rerenders. |
| `news_scraper.py` | Scrapes article links from major financial news homepages, extracts full text via `newspaper3k`, infers sector from content, and returns structured article data. |
| `rag_vector_DB.py` | Wraps text strings in LangChain `Document` objects, generates OpenAI embeddings, and stores them in an in-memory FAISS index for semantic similarity search. |
| `reddit_rag_scraper.py` | Fetches Reddit posts using a business-specific query (industry + goal + market) across targeted subreddits. Deduplicates, filters by quality, and returns both raw text (for RAG) and structured metadata (for UI display). |

---

## Setup

### 1. Prerequisites

- Python 3.11
- [Homebrew](https://brew.sh) (macOS) — needed for Cairo (PDF rendering dependency)

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

Create a `.env` file in the project root — **no quotes, no spaces around the `=`**:

```env
OPENAI_API_KEY=your_openai_api_key_here
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
REDDIT_USER_AGENT=your_app_name:v1.0 (by u/your_reddit_username)
```

> **Never commit your `.env` file.** It is already listed in `.gitignore`.

#### How to get API keys

**OpenAI:**
- Go to [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- Create a new secret key
- Your account needs access to GPT-4

**Reddit (PRAW):**
- Go to [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps)
- Click "create another app" → choose **script**
- Copy the client ID (shown under the app name) and the client secret

### 6. Run the app

```bash
source venv/bin/activate
streamlit run app2.py
```

Opens at `http://localhost:8501`.

---

## How It Works — Data Flow

```
User Input (industry, market, goal, budget)
         │
         ├─── Google Trends (pytrends) ────── 12-month search interest
         ├─── Yahoo Finance (yfinance) ─────── ETF sector price + 5-year YoY history
         ├─── GPT-4 growth projection ──────── 3-year forecast from ETF + trends data
         ├─── Investopedia (scraped) ──────────expert industry definition
         ├─── Google News (scraped) ─────────── top headlines
         ├─── Financial news sites ──────────── Bloomberg / CNBC / Forbes articles
         └─── Reddit (PRAW + FAISS RAG) ─────── goal-specific posts → GPT-4 summary
                          │
                 All data compiled into GPT-4 prompt
                          │
                   GPT-4 (10-section report)
                          │
          ┌───────────────┴────────────────────┐
     Report Text                         Visual Charts
  (specific to your idea)       (Trends / ETF / Growth Projection)
          │                                     │
          └─────────────────┬───────────────────┘
                     In-memory PDF Export
```

### Reddit Intelligence Pipeline

1. A specific search query is built from `industry + goal + target market` (not just the industry name)
2. Posts are fetched across targeted industry subreddits + r/all, filtered by quality (score ≥ 2)
3. Posts are embedded with OpenAI embeddings and stored in a FAISS index
4. The top semantically relevant posts are retrieved using the full business goal as the query
5. GPT-4 summarises the community sentiment into: Overall Sentiment, Top Themes, Opportunities, and Risks

### ETF Industry Mapping

Industries are matched to sector ETFs for live market data. Examples:

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
| Clean Energy | PBW |

Falls back to fuzzy matching (70% threshold) against the ETF map and company-sector CSV if no exact match.

---

## Visual Market Insights (4 Tabs)

| Tab | What It Shows |
|---|---|
| 📈 Google Trends | 12-month search interest line chart for the industry |
| 💹 ETF Performance | 6-month sector ETF price trend |
| 📊 Industry Growth | Historical YoY ETF returns (bars) + GPT-4 projected growth for next 3 years (dashed line) |
| 💬 Reddit Community Pulse | GPT-4 intelligence summary + individual post cards with upvotes, comments, and links |

---

## Dependencies

| Package | Purpose |
|---|---|
| `openai` | GPT-4 API calls |
| `langchain-openai` | OpenAI embeddings for RAG |
| `langchain-community` | FAISS vector store integration |
| `langchain-core` | LangChain base types |
| `faiss-cpu` | Fast vector similarity search |
| `streamlit` | Web UI framework |
| `yfinance` | Yahoo Finance ETF price data |
| `pytrends` | Google Trends API wrapper |
| `praw` | Reddit API client |
| `newspaper3k` | Article text extraction |
| `beautifulsoup4` | HTML parsing |
| `spacy` | NLP keyword and entity extraction |
| `textblob` | Sentiment analysis |
| `pandas` | Data manipulation |
| `matplotlib` | Chart rendering |
| `xhtml2pdf` | In-memory HTML-to-PDF export |
| `python-dotenv` | Load environment variables from `.env` |

---

## Known Limitations

- Google Trends may rate-limit requests; a 2-second delay is built in
- Investopedia and news site scraping may break if their HTML structure changes
- Bloomberg and FT articles may be paywalled
- ETF mapping covers ~20 predefined sectors; niche industries may fall back to fuzzy matching
- Reddit search relevance depends on how specific the business goal description is — more detail = better results
- The FAISS index is rebuilt in-memory on each report generation

---

## Security Notes

- **Do not commit `.env`** — it is in `.gitignore`
- **Rotate your API keys** if they have ever been pushed to a public repository
- The app makes outbound HTTP requests to external sites — respect their terms of service and rate limits
