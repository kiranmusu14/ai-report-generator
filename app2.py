import openai
from openai import OpenAIError
import pandas as pd
from pytrends.request import TrendReq
import yfinance as yf
import spacy
from xhtml2pdf import pisa
from io import BytesIO
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
from difflib import get_close_matches
from textblob import TextBlob
from news_scraper import scrape_latest_business_news, company_sector_df
import requests
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv
from reddit_rag_scraper import get_reddit_posts, get_reddit_posts_with_metadata
from rag_vector_DB import build_vector_db_from_texts, retrieve_relevant_docs

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── NLP ────────────────────────────────────────────────────────────────────────
nlp = spacy.load("en_core_web_sm")

# ── Utility functions ──────────────────────────────────────────────────────────

def scrape_investopedia_definition(industry_term):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        url = f"https://www.investopedia.com/search?q={industry_term.replace(' ', '+')}"
        soup = BeautifulSoup(requests.get(url, headers=headers).text, "html.parser")
        result = soup.select_one("a[data-analytics-label='search-result']")
        if not result:
            return f"No Investopedia article found for '{industry_term}'."
        article_soup = BeautifulSoup(
            requests.get(result["href"], headers=headers).text, "html.parser"
        )
        p = article_soup.find("p")
        return p.text.strip() if p else "No summary found."
    except Exception as e:
        return f"Error scraping Investopedia: {e}"


def scrape_google_news(industry):
    try:
        url = f"https://news.google.com/search?q={industry.replace(' ', '+')}&hl=en-US&gl=US&ceid=US:en"
        soup = BeautifulSoup(
            requests.get(url, headers={"User-Agent": "Mozilla/5.0"}).text, "html.parser"
        )
        articles = soup.select("article h3")
        return [a.get_text() for a in articles[:5]] if articles else ["No Google News results."]
    except Exception as e:
        return [f"Google News scrape failed: {e}"]


def get_statista_placeholder(industry):
    return (
        f"According to Statista, the {industry} industry is expected to grow steadily, "
        "with digital adoption and AI integration being key drivers."
    )


def normalize_industry_term(term):
    mapping = {
        "ai-driven micro-investing": "fintech",
        "credit access": "fintech",
        "wealth-building": "personal finance",
        "financial inclusion": "emerging markets finance",
        "credit scoring": "fintech",
    }
    term = term.lower()
    for k, v in mapping.items():
        if k in term:
            return v
    return term


FALLBACK_ETF_MAP = {
    "technology": "XLK",
    "energy": "XLE",
    "healthcare": "XLV",
    "financial": "XLF",
    "real estate": "XLRE",
    "consumer discretionary": "XLY",
    "utilities": "XLU",
    "industrials": "XLI",
    "materials": "XLB",
    "communications": "XLC",
    "fintech": "FINX",
    "emerging markets finance": "EMFM",
    "blockchain": "BLOK",
    "personal finance": "ARKF",
    "robo advisors": "BOTZ",
    "sustainable packaging": "PKB",
    "aerospace": "ITA",
    "semiconductors": "SOXX",
    "cybersecurity": "CIBR",
    "renewable energy": "ICLN",
    "clean energy": "PBW",
}


def map_industry_to_etf(industry):
    ic = industry.lower().strip()
    if ic in FALLBACK_ETF_MAP:
        return FALLBACK_ETF_MAP[ic]
    matches = get_close_matches(ic, FALLBACK_ETF_MAP.keys(), n=1, cutoff=0.7)
    if matches:
        return FALLBACK_ETF_MAP[matches[0]]
    if "clean_sector" in company_sector_df.columns:
        sector_matches = get_close_matches(
            ic, company_sector_df["clean_sector"].str.lower().unique(), n=1, cutoff=0.7
        )
        if sector_matches and sector_matches[0] in FALLBACK_ETF_MAP:
            return FALLBACK_ETF_MAP[sector_matches[0]]
    return None


def get_market_insights(industry):
    df = pd.read_csv("industry_growth.csv")
    if "GrowthRate" not in df.columns or "Industry" not in df.columns:
        return "Industry data format error.", None
    filtered = df[df["Industry"].str.lower() == industry.lower()]
    if filtered.empty:
        return f"No industry insights available for '{industry}'.", df
    stats = filtered.iloc[0]
    return (
        f"The {industry} industry has an annual growth rate of {stats['GrowthRate']}%. "
        f"Key players include {stats.get('TopCompetitors', 'N/A')}",
        df,
    )


def get_google_trends(industry):
    try:
        pytrends = TrendReq(hl="en-US", tz=360)
        pytrends.build_payload([industry], timeframe="today 12-m")
        time.sleep(2)
        interest = pytrends.interest_over_time()
        if interest.empty:
            return f"No trend data available for {industry}.", None
        return (
            f"Search interest in '{industry}' is {interest[industry].iloc[-1]} (last recorded week).",
            interest,
        )
    except Exception as e:
        return f"Google Trends error: {e}", None


def get_industry_market_summary(industry):
    ticker = map_industry_to_etf(industry)
    if not ticker:
        return f"No ETF mapping found for '{industry}'.", None
    try:
        hist = yf.Ticker(ticker).history(period="6mo")
        if hist.empty:
            return f"No historical data for ETF '{ticker}'.", None
        start, end = hist["Close"].iloc[0], hist["Close"].iloc[-1]
        change = round(((end - start) / start) * 100, 2)
        return (
            f"The {industry} sector via ETF **{ticker}** changed **{change}%** over 6 months "
            f"(${start:.2f} → ${end:.2f}).",
            hist,
        )
    except Exception as e:
        return f"Error retrieving ETF data: {e}", None


def extract_keywords(text):
    doc = nlp(text)
    return list(set(t.text for t in doc if t.pos_ in ["NOUN", "PROPN"] and not t.is_stop))


def get_sentiment(text):
    return TextBlob(text).sentiment


# ── Reddit AI summary ──────────────────────────────────────────────────────────

def summarize_reddit_posts(posts_meta, industry):
    """Use GPT-4 to distil Reddit posts into structured community intelligence."""
    if not posts_meta:
        return None
    posts_text = "\n\n".join(
        f"[r/{p['subreddit']}] {p['title']}\n{p['body']}"
        for p in posts_meta[:10]
    )
    prompt = f"""Analyze these Reddit discussions about the {industry} industry and provide a concise intelligence briefing:

1. **Overall Sentiment** — Bullish / Bearish / Neutral with a one-line reason
2. **Top 3 Trending Themes** — What topics dominate the conversation
3. **Community Opportunities** — What upside the community sees
4. **Key Risks & Concerns** — What worries are being raised

Reddit data:
{posts_text}

Keep each point to 1-2 sentences. Be direct and analytical. Use the exact bold headers above."""
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=400,
        )
        return response.choices[0].message.content.strip()
    except OpenAIError:
        return None


# ── Live industry growth projection ───────────────────────────────────────────

def get_industry_growth_projection(industry):
    """
    Compute historical YoY growth from ETF data and use GPT to project
    growth rates for the next 3 years.
    Returns (hist_years, hist_rates, proj_years, proj_rates, outlook_text).
    """
    import json
    from datetime import datetime

    ticker = map_industry_to_etf(industry)
    hist_years, hist_rates = [], []

    if ticker:
        try:
            hist = yf.Ticker(ticker).history(period="5y")
            if not hist.empty:
                hist["Year"] = hist.index.year
                annual_close = hist.groupby("Year")["Close"].last()
                yoy = annual_close.pct_change().dropna() * 100
                hist_years = [str(y) for y in yoy.index]
                hist_rates = [round(float(v), 1) for v in yoy.values]
        except Exception:
            pass

    trend_summary, _ = get_google_trends(industry)
    news_text = "\n".join(scrape_google_news(industry)[:3])
    hist_context = (
        f"Historical ETF YoY returns: {dict(zip(hist_years, hist_rates))}"
        if hist_years
        else "No historical ETF data available."
    )
    current_year = datetime.now().year

    prompt = f"""You are a market analyst. Based on the data below, estimate the {industry} industry's annual growth rate.

Data:
- {hist_context}
- Google Trends signal: {trend_summary}
- Recent news: {news_text}

Respond ONLY with valid JSON (no markdown, no extra text):
{{"current_growth": <float>, "year1": <float>, "year2": <float>, "year3": <float>, "outlook": "<2 sentence summary>"}}

year1={current_year + 1}, year2={current_year + 2}, year3={current_year + 3}. All values are percentages."""

    proj_years, proj_rates, outlook = [], [], ""
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200,
        )
        text = response.choices[0].message.content.strip()
        data = json.loads(text[text.find("{") : text.rfind("}") + 1])
        proj_years = [str(current_year), str(current_year + 1), str(current_year + 2), str(current_year + 3)]
        proj_rates = [
            data.get("current_growth", 0),
            data.get("year1", 0),
            data.get("year2", 0),
            data.get("year3", 0),
        ]
        outlook = data.get("outlook", "")
    except Exception:
        pass

    return hist_years, hist_rates, proj_years, proj_rates, outlook


# ── Report generators ──────────────────────────────────────────────────────────

def get_business_feasibility_summary(industry, target_market, goal, budget):
    prompt = f"""You are a startup advisor. A founder has shared this business idea:

Industry: {industry}
Idea: {goal}
Target Market: {target_market}
Budget: {budget}

Give a concise but specific feasibility summary with these 5 sections. Everything must be specific to THIS idea — no generic advice.

1. WHAT THIS BUSINESS IS
   Plain-English description of what this does, the problem it solves, and who it serves.

2. MARKET OPPORTUNITY
   Market size estimate, key growth trends, and why the timing is right.

3. REVENUE MODEL
   How this makes money, suggested pricing, and rough unit economics (CAC, LTV).

4. KEY RISKS
   The 3 biggest specific risks for this idea and how to mitigate them.

5. VERDICT & NEXT STEPS
   Feasibility score (X/10), one critical success factor, and 3 immediate actions the founder should take this week."""
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=900,
        )
        return response.choices[0].message.content.strip()
    except OpenAIError as e:
        return f"❌ Error generating report: {e}"


def generate_full_report(industry, target_market, goal, budget):
    industry = normalize_industry_term(industry)
    static_insight, _ = get_market_insights(industry)
    trend_data, _ = get_google_trends(industry)
    market_summary, _ = get_industry_market_summary(industry)
    news_articles = scrape_latest_business_news()
    investopedia_insight = scrape_investopedia_definition(industry)
    if "Error" in investopedia_insight or "No" in investopedia_insight:
        investopedia_insight = "No expert insight available."
    google_news = scrape_google_news(industry)
    news_section = (
        "\n".join(f"- {a['News']}" for a in news_articles[:5])
        if news_articles
        else "No recent news available."
    )
    google_headlines = "\n".join(f"- {h}" for h in google_news)

    # Pass the full business context to Reddit for relevant posts
    reddit_posts = get_reddit_posts(industry, goal=goal, target_market=target_market)
    st.session_state["_reddit_raw"] = reddit_posts
    reddit_db = build_vector_db_from_texts(reddit_posts)
    # Retrieve using the specific goal as the query, not just industry
    reddit_context = "\n".join(
        doc.page_content for doc in retrieve_relevant_docs(reddit_db, f"{industry} {goal}")
    )

    prompt = f"""You are a senior startup advisor and business strategist. A founder has come to you with a specific business idea. Your job is to produce a comprehensive, specific, and immediately actionable business plan report.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THE BUSINESS IDEA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Industry: {industry}
Business Idea / Goal: {goal}
Target Market: {target_market}
Available Budget: {budget}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LIVE MARKET DATA (incorporate these into your analysis)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Industry Growth: {static_insight}
Google Trends Signal: {trend_data}
Sector ETF Performance: {market_summary}
Industry Definition: {investopedia_insight}
Recent News Headlines: {google_headlines}
Financial News: {news_section}
Community Sentiment (Reddit): {reddit_context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INSTRUCTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Generate a detailed report with EXACTLY these 10 sections. Every section must be SPECIFIC to the business idea above — never give generic advice. Reference the actual goal, market, and budget throughout.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. BUSINESS CONCEPT OVERVIEW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- What this business does in 2-3 plain sentences
- The core problem it solves and for whom (be specific about the customer pain)
- The proposed solution and how it works
- Unique value proposition — what makes this different from existing solutions
- Why this idea is relevant right now (market timing)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. MARKET OPPORTUNITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Total Addressable Market (TAM) with $ estimate
- Serviceable Addressable Market (SAM) — the realistic slice for this business
- Serviceable Obtainable Market (SOM) — Year 1 target with reasoning
- 3 key market trends and tailwinds supporting this business right now
- Industry growth rate and what is driving it

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. BUSINESS MODEL & REVENUE STREAMS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Recommended revenue model (SaaS / marketplace / transactional / freemium / etc.) with justification
- Specific pricing strategy with suggested price points
- 2-3 revenue streams if applicable
- Unit economics: estimated CAC (customer acquisition cost), LTV (lifetime value), gross margin %
- How the business makes money from Day 1

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. FINANCIAL PROJECTIONS (12 months)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Budget allocation for {budget}:
  - Product / Tech development: X%
  - Marketing & customer acquisition: X%
  - Operations & team: X%
  - Reserve / contingency: X%

Timeline:
  - Month 1-3: Setup costs, key hires, expected spend, early revenue potential
  - Month 4-6: Growth targets, customer count goals, revenue range
  - Month 7-12: Scale targets, monthly revenue goal, cumulative revenue
  - Estimated breakeven point (month number)
  - Projected ROI by end of month 12

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5. COMPETITIVE LANDSCAPE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Name 3-4 direct competitors (real companies) with their weaknesses and gaps
- How this business exploits those gaps
- Key differentiators and sustainable competitive moat
- Market positioning: where this business sits vs. competitors (price, features, segment)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
6. GO-TO-MARKET STRATEGY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Exact plan to acquire the first 100 customers (step by step)
- Top 3 acquisition channels with specific tactics for each
- Content / community / partnership strategy
- How to allocate the marketing portion of {budget} across channels

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
7. 90-DAY EXECUTION ROADMAP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Week 1-2: Immediate validation steps and setup actions
- Week 3-4: MVP definition and early build
- Month 2: Key development milestones and first user tests
- Month 3: Soft launch plan, beta targets, feedback loops

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
8. TECH STACK & OPERATIONAL REQUIREMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Recommended tech stack for the MVP (tools, platforms, frameworks)
- Minimum team required to launch (roles, whether to hire or outsource)
- Key vendors, tools, or APIs needed
- Compliance or regulatory requirements for this industry

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
9. RISKS & MITIGATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Top 4 specific risks for THIS business idea
- Concrete mitigation plan for each risk
- Biggest assumption that could invalidate the model and how to test it early

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
10. FINAL VERDICT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Feasibility Score: X/10 (with brief justification)
- Confidence Level: High / Medium / Low
- The 3 things the founder absolutely must get right for this to succeed
- Recommended next 3 concrete steps to take this week

Be specific, direct, and data-driven throughout. Every number, recommendation, and strategy must relate to THIS business idea, THIS target market, and THIS budget. Do not give generic startup advice."""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=2800,
        )
        return response.choices[0].message.content.strip()
    except OpenAIError as e:
        return f"❌ Error generating full report: {e}"


# ── PDF export (in-memory, no disk write) ─────────────────────────────────────

def export_to_pdf_bytes(report_text, industry):
    formatted = report_text.replace("\n", "<br>")
    html = f"""
    <html>
    <head><style>
        body {{ font-family: Arial, sans-serif; padding: 32px; font-size: 12px;
                line-height: 1.7; color: #1a1a1a; }}
        h1 {{ color: #1f4e79; font-size: 20px; border-bottom: 2px solid #1f4e79;
               padding-bottom: 8px; margin-bottom: 16px; }}
        p {{ margin: 4px 0; }}
        .meta {{ color: #555; font-size: 11px; margin-bottom: 20px; }}
    </style></head>
    <body>
        <h1>AI-Generated Business Feasibility Report</h1>
        <p class="meta"><strong>Industry:</strong> {industry}</p>
        <hr>
        <div>{formatted}</div>
    </body>
    </html>
    """
    buf = BytesIO()
    pisa.CreatePDF(html, dest=buf)
    buf.seek(0)
    return buf


# ── Chart helpers (return fig, no disk write) ──────────────────────────────────

CHART_BG  = "#ffffff"
CHART_AX  = "#f8fafc"
BORDER    = "#e2e8f0"
BLUE      = "#4f46e5"
PURPLE    = "#7c3aed"
GREEN     = "#16a34a"
ORANGE    = "#ea580c"
TEXT      = "#0f172a"
MUTED     = "#64748b"


def _apply_light_style(fig, ax, title):
    fig.patch.set_facecolor(CHART_BG)
    ax.set_facecolor(CHART_AX)
    ax.set_title(title, color=TEXT, pad=12, fontsize=11, fontweight="bold")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=BORDER, linewidth=0.7, linestyle="--", alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def make_trends_chart(trend_df, industry):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    col = trend_df.columns[0]
    ax.plot(trend_df.index, trend_df[col], color=BLUE, linewidth=2.5)
    ax.fill_between(trend_df.index, trend_df[col], alpha=0.1, color=BLUE)
    ax.set_ylabel("Search Interest", color=MUTED, fontsize=9)
    _apply_light_style(fig, ax, f"Google Search Interest — '{industry}' (12 months)")
    plt.tight_layout()
    return fig


def make_etf_chart(hist, industry):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(hist.index, hist["Close"], color=PURPLE, linewidth=2.5)
    ax.fill_between(hist.index, hist["Close"].min(), hist["Close"], alpha=0.1, color=PURPLE)
    ax.set_ylabel("Price (USD)", color=MUTED, fontsize=9)
    _apply_light_style(fig, ax, f"{industry.title()} ETF — 6-Month Performance")
    plt.tight_layout()
    return fig


def make_growth_projection_chart(hist_years, hist_rates, proj_years, proj_rates, industry):
    fig, ax = plt.subplots(figsize=(10, 4.5))

    if hist_years and hist_rates:
        ax.bar(
            hist_years, hist_rates,
            color=BLUE, alpha=0.75, label="Historical ETF YoY Return (%)",
            width=0.5, zorder=2,
        )

    if proj_years and proj_rates:
        ax.plot(
            proj_years, proj_rates,
            color=ORANGE, linewidth=2.5, marker="o", markersize=8,
            label="GPT-4 Projected Growth (%)", linestyle="--", zorder=5,
        )
        for y, r in zip(proj_years, proj_rates):
            ax.annotate(
                f"{r:.1f}%", (y, r),
                textcoords="offset points", xytext=(0, 11),
                ha="center", color=ORANGE, fontsize=8, fontweight="bold",
            )

    ax.axhline(0, color="#cbd5e1", linewidth=1)
    ax.set_ylabel("Growth / Return (%)", color=MUTED, fontsize=9)
    ax.legend(facecolor=CHART_BG, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)
    _apply_light_style(fig, ax, f"{industry.title()} — Historical & Projected Growth Rate")
    plt.tight_layout()
    return fig


# ── Streamlit config ───────────────────────────────────────────────────────────

st.set_page_config(
    page_title="BizGen AI — Business Report Generator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
[data-testid="stAppViewContainer"] {
    background: #f1f5f9 !important;
}
[data-testid="stHeader"] {
    background: #ffffff !important;
    border-bottom: 1px solid #e2e8f0;
}
[data-testid="stMain"], .main {
    background: #f1f5f9 !important;
}
.block-container {
    padding-top: 2.5rem;
    max-width: 1100px;
}

/* Force readable text everywhere */
p, li, span, div, label {
    color: #1e293b !important;
}

/* ── Hero ── */
.hero-wrap {
    background: linear-gradient(135deg, #1e3a8a 0%, #4f46e5 50%, #7c3aed 100%);
    border-radius: 20px;
    padding: 3rem 2rem 2.5rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 4px 24px rgba(79,70,229,0.18);
}
.hero-title {
    font-size: 3rem; font-weight: 900; color: #ffffff !important;
    line-height: 1.15; margin-bottom: 0.5rem; letter-spacing: -1px;
}
.hero-sub {
    color: #c7d2fe !important; font-size: 1.05rem; margin-bottom: 1.5rem;
}
.badges { text-align: center; }
.badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    color: #ffffff !important;
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 20px;
    padding: 5px 14px; font-size: 0.78rem; margin: 3px; font-weight: 500;
    backdrop-filter: blur(4px);
}

/* Streamlit input overrides */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    background: #f8fafc !important;
    color: #0f172a !important;
    border: 1.5px solid #cbd5e1 !important;
    border-radius: 8px !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: #4f46e5 !important;
    box-shadow: 0 0 0 3px rgba(79,70,229,0.12) !important;
}
[data-testid="stSelectbox"] > div {
    background: #f8fafc !important;
    border: 1.5px solid #cbd5e1 !important;
    border-radius: 8px !important;
    color: #0f172a !important;
}
label, .stTextInput label, .stTextArea label, .stSelectbox label {
    color: #374151 !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
}

/* ── Report block ── */
.report-block {
    background: #ffffff;
    color: #1e293b !important;
    padding: 2rem 2.5rem;
    border-radius: 16px;
    border: 1px solid #e2e8f0;
    line-height: 1.9;
    font-size: 0.94rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.05);
}
.report-block * { color: #1e293b !important; }

/* ── Reddit cards ── */
.reddit-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #ff4500;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.reddit-title { color: #0f172a !important; font-weight: 700; font-size: 0.93rem; margin-bottom: 5px; }
.reddit-body  { color: #475569 !important; font-size: 0.84rem; margin-bottom: 8px; line-height: 1.55; }
.reddit-meta  { color: #94a3b8 !important; font-size: 0.77rem; }
.reddit-meta a { color: #4f46e5 !important; text-decoration: none; font-weight: 500; }
.reddit-meta a:hover { text-decoration: underline; }

/* ── Section header ── */
.section-header {
    color: #1e293b !important;
    font-size: 1.25rem; font-weight: 800;
    margin: 1.5rem 0 0.75rem 0;
    padding-bottom: 8px;
    border-bottom: 2px solid #4f46e5;
    letter-spacing: -0.3px;
}

/* ── Buttons ── */
div.stButton > button {
    border-radius: 9px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    border: 1.5px solid #cbd5e1 !important;
    background: #ffffff !important;
    color: #374151 !important;
    transition: all 0.18s !important;
}
div.stButton > button:hover {
    border-color: #4f46e5 !important;
    color: #4f46e5 !important;
    box-shadow: 0 2px 8px rgba(79,70,229,0.15) !important;
}
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: #ffffff !important;
    border: none !important;
    font-size: 1rem !important;
    padding: 0.6rem 1.5rem !important;
    box-shadow: 0 4px 14px rgba(79,70,229,0.35) !important;
}
div.stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 20px rgba(79,70,229,0.45) !important;
    color: #ffffff !important;
}
div.stDownloadButton > button {
    background: linear-gradient(135deg, #16a34a, #15803d) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 9px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 0.55rem 1.2rem !important;
    box-shadow: 0 3px 10px rgba(22,163,74,0.3) !important;
}

/* ── Tabs ── */
button[data-baseweb="tab"] {
    font-weight: 600 !important;
    font-size: 0.87rem !important;
    color: #64748b !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #4f46e5 !important;
    border-bottom: 3px solid #4f46e5 !important;
}

/* ── Status / spinner ── */
[data-testid="stStatusWidget"] { border-radius: 10px !important; }

/* ── Captions ── */
[data-testid="stCaptionContainer"] p { color: #64748b !important; font-size: 0.8rem !important; }

hr { border-color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state init ─────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "Home"

# ═══════════════════════════════════════════════════════════════════════════════
# HOME PAGE
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "Home":

    st.markdown("""
        <div class='hero-wrap'>
            <p class='hero-title'>BizGen AI</p>
            <p class='hero-sub'>
                Generate investor-ready business feasibility reports powered by GPT-4 and live market data.
            </p>
            <div class='badges'>
                <span class='badge'>📈 Google Trends</span>
                <span class='badge'>💹 Live ETF Data</span>
                <span class='badge'>🤖 GPT-4 Analysis</span>
                <span class='badge'>💬 Reddit Intelligence</span>
                <span class='badge'>📰 News Scraping</span>
                <span class='badge'>📄 PDF Export</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 2.2, 1])
    with col:
        st.markdown("#### 🏢 Enter Business Details")

        c1, c2 = st.columns(2)
        industry      = c1.text_input("Industry", placeholder="e.g. Fintech, SaaS, Healthcare")
        target_market = c2.text_input("Target Market", placeholder="e.g. Gen Z in North America")
        goal          = st.text_area("Business Goal", placeholder="Describe your product or idea", height=100)

        c3, c4 = st.columns(2)
        budget      = c3.text_input("Estimated Budget", placeholder="e.g. $50,000")
        report_type = c4.selectbox("Report Type", ["Full Report", "Quick Summary"])

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.button("Generate Report →", type="primary", use_container_width=True)

    if submitted:
        if not all([industry, target_market, goal, budget]):
            st.error("Please fill in all required fields.")
        else:
            # Clear any previous report cache
            for key in ["result", "trend_summary", "trend_df", "market_summary",
                        "hist", "reddit_posts_meta", "reddit_summary",
                        "growth_projection", "_reddit_raw"]:
                st.session_state.pop(key, None)

            st.session_state.update({
                "page":         "Generated Report",
                "generated":    True,
                "industry":     industry,
                "target_market": target_market,
                "goal":         goal,
                "budget":       budget,
                "report_type":  report_type,
            })
            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# REPORT PAGE
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "Generated Report" and st.session_state.get("generated"):

    industry     = st.session_state.industry
    target_market = st.session_state.target_market
    goal         = st.session_state.goal
    budget       = st.session_state.budget
    report_type  = st.session_state.report_type

    # ── Top nav ───────────────────────────────────────────────────────────────
    col_back, col_title = st.columns([1, 6])
    with col_back:
        if st.button("← New Report"):
            for key in ["result", "trend_summary", "trend_df", "market_summary",
                        "hist", "reddit_posts_meta", "reddit_summary",
                        "growth_projection", "_reddit_raw", "generated"]:
                st.session_state.pop(key, None)
            st.session_state.page = "Home"
            st.rerun()
    with col_title:
        st.markdown(
            f"<h2 style='color:#4fa3e0; margin:0; padding-top:4px;'>"
            f"📊 {industry.title()} — Business Feasibility Report</h2>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Generate report once, cache result ────────────────────────────────────
    if "result" not in st.session_state:
        with st.status("Generating your report...", expanded=True) as status:
            st.write("🔍 Scraping news and market data...")
            st.write("📡 Fetching Google Trends and ETF performance...")
            st.write("💬 Gathering Reddit community insights...")
            st.write("🤖 Running GPT-4 analysis...")

            if report_type.startswith("Full"):
                result = generate_full_report(industry, target_market, goal, budget)
            else:
                result = get_business_feasibility_summary(industry, target_market, goal, budget)

            st.session_state.result = result
            status.update(label="Report ready!", state="complete", expanded=False)
    else:
        result = st.session_state.result

    # ── Report display ────────────────────────────────────────────────────────
    st.markdown("<p class='section-header'>Report</p>", unsafe_allow_html=True)
    formatted = result.replace("\n", "<br>")
    st.markdown(f"<div class='report-block'>{formatted}</div>", unsafe_allow_html=True)

    # ── PDF download (in-memory, no disk write) ───────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    pdf_bytes = export_to_pdf_bytes(result, industry)
    st.download_button(
        label="⬇ Download PDF Report",
        data=pdf_bytes,
        file_name=f"{industry.lower().replace(' ', '_')}_report.pdf",
        mime="application/pdf",
    )

    st.markdown("---")
    st.markdown("<p class='section-header'>Visual Market Insights</p>", unsafe_allow_html=True)

    tabs = st.tabs(["📈 Google Trends", "💹 ETF Performance", "📊 Industry Growth", "💬 Reddit Community Pulse"])

    # ── Tab 1 — Google Trends ─────────────────────────────────────────────────
    with tabs[0]:
        if "trend_df" not in st.session_state:
            with st.spinner("Loading trend data..."):
                summary, df = get_google_trends(industry)
                st.session_state.trend_summary = summary
                st.session_state.trend_df = df
        else:
            summary = st.session_state.trend_summary
            df      = st.session_state.trend_df

        st.caption(f"📊 {summary}")
        if df is not None and not df.empty:
            fig = make_trends_chart(df, industry)
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No Google Trends data available for this industry.")

    # ── Tab 2 — ETF Performance ───────────────────────────────────────────────
    with tabs[1]:
        if "hist" not in st.session_state:
            with st.spinner("Loading ETF data..."):
                mkt_summary, hist = get_industry_market_summary(industry)
                st.session_state.market_summary = mkt_summary
                st.session_state.hist = hist
        else:
            mkt_summary = st.session_state.market_summary
            hist        = st.session_state.hist

        st.caption(f"💹 {mkt_summary}")
        if hist is not None and not hist.empty:
            fig = make_etf_chart(hist, industry)
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No ETF data available for this industry.")

    # ── Tab 3 — Industry Growth (live scrape + GPT projection) ───────────────
    with tabs[2]:
        if "growth_projection" not in st.session_state:
            with st.spinner("Scraping market data and projecting growth..."):
                hy, hr, py, pr, outlook = get_industry_growth_projection(industry)
                st.session_state.growth_projection = (hy, hr, py, pr, outlook)
        else:
            hy, hr, py, pr, outlook = st.session_state.growth_projection

        if outlook:
            st.markdown(f"**Market Outlook:** {outlook}")
            st.markdown("<br>", unsafe_allow_html=True)

        if hy or py:
            fig = make_growth_projection_chart(hy, hr, py, pr, industry)
            st.pyplot(fig)
            plt.close(fig)
            st.caption(
                "Blue bars = historical ETF year-over-year return (proxy for sector growth). "
                "Orange dashed line = GPT-4 projected growth rate."
            )
        else:
            st.info(f"Could not compute growth data for '{industry}'. Try a broader industry name.")

    # ── Tab 4 — Reddit Community Pulse ────────────────────────────────────────
    with tabs[3]:
        if "reddit_posts_meta" not in st.session_state:
            with st.spinner("Fetching and analysing Reddit community discussions..."):
                posts_meta = get_reddit_posts_with_metadata(industry, goal=goal, target_market=target_market)
                st.session_state.reddit_posts_meta = posts_meta
                reddit_summary = summarize_reddit_posts(posts_meta, industry)
                st.session_state.reddit_summary = reddit_summary
        else:
            posts_meta     = st.session_state.reddit_posts_meta
            reddit_summary = st.session_state.get("reddit_summary")

        # GPT summary at the top
        if reddit_summary:
            st.markdown(
                f"<div class='report-block' style='margin-bottom:1.5rem;'>{reddit_summary.replace(chr(10), '<br>')}</div>",
                unsafe_allow_html=True,
            )

        if posts_meta:
            st.markdown(
                f"**{len(posts_meta)} source discussions** from Reddit about *{industry}*",
            )
            st.markdown("<br>", unsafe_allow_html=True)
            for p in posts_meta:
                if p["score"] >= 100:
                    dot = "🟢"
                elif p["score"] >= 20:
                    dot = "🟡"
                else:
                    dot = "🔴"
                body_html = f"<div class='reddit-body'>{p['body']}</div>" if p["body"] else ""
                st.markdown(
                    f"""
                    <div class='reddit-card'>
                        <div class='reddit-title'>{p['title']}</div>
                        {body_html}
                        <div class='reddit-meta'>
                            {dot}&nbsp; r/{p['subreddit']}
                            &nbsp;·&nbsp; ▲ {p['score']} upvotes
                            &nbsp;·&nbsp; 💬 {p['comments']} comments
                            &nbsp;·&nbsp; <a href='{p['url']}' target='_blank'>View on Reddit ↗</a>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No Reddit posts found for this industry.")

# ── Fallback ───────────────────────────────────────────────────────────────────
else:
    st.markdown("<br><br>", unsafe_allow_html=True)
    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.info("No report generated yet. Please go to the Home page to get started.")
        if st.button("Go to Home →", type="primary", use_container_width=True):
            st.session_state.page = "Home"
            st.rerun()
