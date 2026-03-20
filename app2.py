# ✅ FIXED & UPDATED VERSION OF app.py (with GrowthRate fix, ETF mapping, and improved prompt)

import openai
from openai import OpenAIError
import pandas as pd
from pytrends.request import TrendReq
import yfinance as yf
import spacy
from xhtml2pdf import pisa
import streamlit as st
import matplotlib.pyplot as plt
import time
from difflib import get_close_matches
from textblob import TextBlob
from news_scraper import scrape_latest_business_news, company_sector_df
import requests
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv
from reddit_rag_scraper import get_reddit_posts
from rag_vector_DB import build_vector_db_from_texts, retrieve_relevant_docs

load_dotenv() 
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
def scrape_investopedia_definition(industry_term):
    search_query = industry_term.replace(" ", "+")
    url = f"https://www.investopedia.com/search?q={search_query}"

    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")

        # Get first result link
        result = soup.select_one("a[data-analytics-label='search-result']")
        if not result:
            return f"No Investopedia article found for '{industry_term}'."

        article_url = result["href"]
        article_response = requests.get(article_url, headers=headers)
        article_soup = BeautifulSoup(article_response.text, "html.parser")

        # Extract summary paragraph
        paragraph = article_soup.find("p")
        return paragraph.text.strip() if paragraph else "No summary found."

    except Exception as e:
        return f"⚠️ Error scraping Investopedia: {e}"


# 🔹 Web Scraping: Google News Headlines
def scrape_google_news(industry):
    try:
        query = industry.replace(" ", "+")
        url = f"https://news.google.com/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        articles = soup.select("article h3")
        return [a.get_text() for a in articles[:5]] if articles else ["No Google News results."]
    except Exception as e:
        return [f"⚠️ Google News scrape failed: {e}"]

# 🔹 Web Scraping: Statista Placeholder (since real-time scraping often blocked)
def get_statista_placeholder(industry):
    # Simulate or return static sample for now
    return f"According to Statista, the {industry} industry is expected to grow steadily, with digital adoption and AI integration being key drivers."


def normalize_industry_term(term):
    mapping = {
        "ai-driven micro-investing": "fintech",
        "credit access": "fintech",
        "wealth-building": "personal finance",
        "financial inclusion": "emerging markets finance",
        "credit scoring": "fintech"
    }
    term = term.lower()
    for k, v in mapping.items():
        if k in term:
            return v
    return term

fallback_etf_map = {
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
    "clean energy": "PBW"
}

# 🔹 Step 3: GPT-4 Summary Report
def get_business_feasibility_summary(industry, target_market, goal, budget):
    prompt = f"""
    Analyze the business idea below and provide a concise business feasibility summary.
    Business Details:
    - Industry: {industry}
    - Target Market: {target_market}
    - Business Goal: {goal}
    - Estimated Budget: {budget}
    Respond with the following sections:
    1. Executive Summary
    2. Market Opportunity
    3. Key Challenges or Risks
    4. Feasibility Assessment
    5. Recommendation
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except OpenAIError as e:
        return f"\u274c Error generating report: {e}"


# 🔹 Step 4: Static Market Data from CSV
def get_market_insights(industry):
    df = pd.read_csv("industry_growth.csv")
    if 'GrowthRate' not in df.columns or 'Industry' not in df.columns:
        return "Industry data format error: Missing 'GrowthRate' or 'Industry' column.", None
    filtered = df[df['Industry'].str.lower() == industry.lower()]
    if filtered.empty:
        return f"No industry insights available for '{industry}'.", df
    stats = filtered.iloc[0]
    return (
        f"The {industry} industry has an annual growth rate of {stats['GrowthRate']}%. Key players include {stats.get('TopCompetitors', 'N/A')}", df
    )


# 🔹 Step 5: Google Trends
def get_google_trends(industry):
    try:
        pytrends = TrendReq(hl='en-US', tz=360)
        pytrends.build_payload([industry], timeframe='today 12-m')
        time.sleep(2)
        interest = pytrends.interest_over_time()
        if interest.empty:
            return f"No trend data available for {industry}.", None
        return (
            f"Search interest in '{industry}' is {interest[industry].iloc[-1]} (last recorded week).",
            interest
        )
    except Exception as e:
        return f"Google Trends error: {e}", None


def map_industry_to_etf(industry):
    industry_clean = industry.lower().strip()

    # 1. Try direct match in fallback ETF map
    if industry_clean in fallback_etf_map:
        return fallback_etf_map[industry_clean]

    # 2. Try fuzzy match in fallback ETF map
    matches = get_close_matches(industry_clean, fallback_etf_map.keys(), n=1, cutoff=0.7)
    if matches:
        st.info(f"Fuzzy matched input **'{industry}'** to **'{matches[0]}'**, using ETF **{fallback_etf_map[matches[0]]}**.")
        return fallback_etf_map[matches[0]]

    # 3. Try match from sector CSV (inferred from company_sector_df)
    if "clean_sector" in company_sector_df.columns:
        sector_matches = get_close_matches(industry_clean, company_sector_df['clean_sector'].str.lower().unique(), n=1, cutoff=0.7)
        if sector_matches:
            sector = sector_matches[0]
            if sector in fallback_etf_map:
                st.info(f"Matched industry **'{industry}'** to CSV sector **'{sector}'**, using ETF **{fallback_etf_map[sector]}**.")
                return fallback_etf_map[sector]

    # 4. If nothing matched
    st.warning(f"No ETF mapping found for **'{industry}'**. Try using a more common or broad sector name.")
    return None



def get_industry_market_summary(industry):
    ticker = map_industry_to_etf(industry)
    if not ticker:
        return f"No ETF mapping found for '{industry}'. Try using a common sector.", None
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="6mo")
        if hist.empty:
            return f"No historical data found for ETF '{ticker}'.", None
        start_price = hist["Close"].iloc[0]
        end_price = hist["Close"].iloc[-1]
        change = round(((end_price - start_price) / start_price) * 100, 2)
        return f"The {industry} sector via ETF **{ticker}** changed **{change}%** in 6 months (from ${start_price:.2f} to ${end_price:.2f}).", hist
    except Exception as e:
        return f"Error retrieving ETF data: {e}", None


# 🔹 Step 7: NLP
nlp = spacy.load("en_core_web_sm")

def extract_keywords(text):
    doc = nlp(text)
    return list(set(token.text for token in doc if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop))

def extract_named_entities(text):
    return [(ent.text, ent.label_) for ent in nlp(text).ents]

def get_sentiment(text):
    return TextBlob(text).sentiment

# 🔹 Step 8: Full GPT Report with Real-Time News
def generate_full_report(industry, target_market, goal, budget):
    industry = normalize_industry_term(industry)
    static_insight, _ = get_market_insights(industry)
    trend_data, _ = get_google_trends(industry)
    market_summary, _ = get_industry_market_summary(industry)
    news_articles = scrape_latest_business_news()
    investopedia_insight = scrape_investopedia_definition(industry)
    # fallback check for investopedia insight
    if "Error" in investopedia_insight or "No" in investopedia_insight:
        investopedia_insight = "No expert insight available. Please consult additional sources."
    google_news = scrape_google_news(industry)
    statista_insight = get_statista_placeholder(industry)
    news_section = "\n".join(f"- {a['News']}" for a in news_articles[:5]) if news_articles else "No recent news available."
    google_headlines = "\n".join(f"- {headline}" for headline in google_news)
    reddit_posts = get_reddit_posts(industry)
    reddit_db = build_vector_db_from_texts(reddit_posts)
    reddit_context = "\n".join([doc.page_content for doc in retrieve_relevant_docs(reddit_db, industry)])


    prompt = f"""
    You are an expert AI business analyst with deep knowledge of industry trends, competitive analysis, and financial modeling.

    Based on the user inputs and real-world market data provided below, generate a professional and comprehensive business feasibility report. 
    The report should be structured, detailed, and suitable for investor and stakeholder presentations.

    🔹 User Inputs:
    - Industry Focus: {industry}
    - Target Market: {target_market}
    - Business Goal: {goal}
    - Estimated Budget: {budget}
    
    🔹 Growth Rate/Market Insights:
    {static_insight}

    🔹 Google Trends/Real-Time Consumer Trends:
    {trend_data}

    🔹 ETF Market Behavior/ETF-Based Financial Market Analysis:
    {market_summary}

    🔹 Expert Insight from Investopedia:
    {investopedia_insight}
    
    🔹 Statista-style Industry Summary:
    {statista_insight}

    🔹 Google News Headlines:
    {google_headlines}

    🔹 Latest Sector Headlines/Real-World Industry Headlines (scraped from financial news sites):
    {news_section}
    
    🔹 Reddit & LinkedIn Trend Insight:
    {reddit_context}   
    
    ✅ Please include:
    - Competitor Overview with at least two similar startups or platforms
    - Specific numeric assumptions in financial projections (costs, growth, breakeven)
    - Strategic recommendations for MVP launch, capital efficiency, and user traction

    📊 Report Structure:
    1. Executive Market Summary (highlight industry scope and relevance)
    2. SWOT Analysis (Strengths, Weaknesses, Opportunities, Threats)
    3. Financial Forecast (3-6 month outlook, with budget assumptions)
    4. Industry Scale and Growth Trends (macro view with statistics if possible, Industry CAGR, expansion potential, TAM/SAM/SOM)
    5. Competitive Landscape (direct/indirect alternatives)
    6. Tactical Recommendations (go-to-market strategy)
    7. Pre-launch Readiness & Capital Planning (tech stack, hiring, phase rollout)
    8. Final Assessment (Contingencies, compliance, fallback paths, is this feasible, what next?)

    Use a formal tone and make it suitable for investor and stakeholder presentations.
    Use formal business tone and back insights with estimated figures where possible. Make it suitable for investors, incubators, or executive briefings.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=1200
        )
        return response.choices[0].message.content.strip()
    except OpenAIError as e:
        return f"\u274c Error generating full report: {e}"

# (All other functions like trends chart, bar chart, Streamlit UI remain unchanged — but now this code will return actual ETF + growth rate results if the CSV and input are aligned.)
# 🔹 Step 9: Export Report to PDF (Formatted)
def export_to_pdf(report_content, output_file):
    formatted_content = report_content.replace("\n", "<br>")
    html_content = f"""
    <html>
    <head><style>body {{ font-family: Arial; padding: 30px; }}</style></head>
    <body><h1>AI-Generated Business Feasibility Report</h1>{formatted_content}</body>
    </html>"""
    with open(output_file, "w+b") as f:
        pisa_status = pisa.CreatePDF(src=html_content, dest=f)
    return pisa_status.err

# 🔹 Step 10: Charts
def plot_trends_chart(trend_df, industry):
    if trend_df is None or trend_df.empty:
        st.warning("No Google Trends data to plot.")
        return
    try:
        keyword_col = trend_df.columns[0]
        fig, ax = plt.subplots()
        trend_df[keyword_col].plot(ax=ax, title=f"Google Trends for '{keyword_col}' (12 months)", figsize=(10, 3))
        ax.set_ylabel("Search Interest")
        ax.set_xlabel("Date")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error in Google Trends chart: {e}")

    
def plot_etf_price_trend(hist, industry):
    if hist is None or hist.empty:
        st.warning("No ETF price data available.")
        return
    fig, ax = plt.subplots()
    hist["Close"].plot(ax=ax, title=f"{industry.capitalize()} Sector - ETF Price Trend", figsize=(10, 3))
    ax.set_ylabel("Price (USD)")
    st.pyplot(fig)


def plot_growth_bar(df):
    fig, ax = plt.subplots()
    top_df = df.sort_values(by="GrowthRate", ascending=False).head(10)
    ax.barh(top_df["Industry"], top_df["GrowthRate"])
    ax.set_title("Top 10 Fastest-Growing Industries")
    ax.set_xlabel("Growth Rate (%)")
    ax.invert_yaxis()
    st.pyplot(fig)

# stream lit 
import streamlit as st
import matplotlib.pyplot as plt
from xhtml2pdf import pisa

st.set_page_config(page_title="AI Business Report Generator", layout="wide")

if "page" not in st.session_state:
    st.session_state.page = "Home"

page = st.session_state.page

# JavaScript and CSS for button styling and sound
st.markdown("""
    <script>
        function playSound() {
            var audio = new Audio("https://www.soundjay.com/buttons/sounds/button-29.mp3");
            audio.play();
        }
    </script>
    <style>
        div.stButton > button:first-child {
            height: 3em;
            font-weight: bold;
            border-radius: 8px;
            border: none;
            background-color: #dc3545;
            color: white;
            transition: all 0.3s ease;
        }
        div.stButton > button.enabled {
            background-color: #28a745 !important;
        }
    </style>
""", unsafe_allow_html=True)

# Home Page
if page == "Home":
    st.markdown("""
        <h1 style='text-align: center; color: #1f4e79;'>AI-Powered Business Report Generator</h1>
        <h4 style='text-align: center; color: #ccc;'>Generate investor-ready business reports using GPT-4 and live market data.</h4>
    """, unsafe_allow_html=True)

    with st.form("report_form"):
        st.markdown("### Enter Business Details")
        industry = st.text_input("Industry", placeholder="e.g. Fintech")
        target_market = st.text_input("Target Market", placeholder="e.g. Gen Z in North America")
        goal = st.text_area("Business Goal", placeholder="Describe your product or idea")
        budget = st.text_input("Estimated Budget", placeholder="$10,000")
        report_type = st.selectbox("Report Type", ["Summary", "Full"])

        form_complete = all([industry, target_market, goal, budget])
        submit = st.form_submit_button("Generate Report", type="primary")

        st.markdown(f"""
            <script>
                const btn = window.parent.document.querySelector('button[type="submit"]');
                if (btn) {{
                    btn.classList.toggle("enabled", {str(form_complete).lower()});
                }}
            </script>
        """, unsafe_allow_html=True)

    if submit:
        st.session_state.generated = True
        st.session_state.industry = industry
        st.session_state.target_market = target_market
        st.session_state.goal = goal
        st.session_state.budget = budget
        st.session_state.report_type = report_type
        st.session_state.page = "Generated Report"
        st.components.v1.html("<script>playSound();</script>", height=0)
        st.rerun()

# Generated Report Page
elif page == "Generated Report" and st.session_state.get("generated"):
    industry = st.session_state.industry
    target_market = st.session_state.target_market
    goal = st.session_state.goal
    budget = st.session_state.budget
    report_type = st.session_state.report_type

    if report_type == "Summary":
        result = get_business_feasibility_summary(industry, target_market, goal, budget)
    else:
        result = generate_full_report(industry, target_market, goal, budget)

    formatted_result = result.replace('\n', '<br>')
    st.markdown("""
        <style>
            .report-block {
                background-color: #0e1117;
                color: #f0f2f6;
                padding: 1.5em;
                border-radius: 12px;
                border: 1px solid #444;
            }
        </style>
    """, unsafe_allow_html=True)
    st.markdown("<h2 style='color:#1f4e79;'>Business Report</h2>", unsafe_allow_html=True)
    st.markdown(f"<div class='report-block'>{formatted_result}</div>", unsafe_allow_html=True)

    # Visual Market Insights Tabs
    st.markdown("---")
    st.markdown("<h3 style='color:#1f4e79;'>Visual Market Insights</h3>", unsafe_allow_html=True)
    tabs = st.tabs(["📈 Google Trends", "💹 ETF Price Trend", "📊 Industry Growth",  "💬 Reddit Trends"])

    # Tab 1: Google Trends
    with tabs[0]:
        trend_summary, trend_df = get_google_trends(industry)
        st.markdown(f"*Google Trends Summary*: {trend_summary}")
        if trend_df is not None:
            fig1, ax1 = plt.subplots()
            trend_df[trend_df.columns[0]].plot(ax=ax1, title=f"Google Trends for {industry}")
            st.pyplot(fig1)
            fig1.savefig("trend_chart.png")

    # Tab 2: ETF Market Behavior
    with tabs[1]:
        market_summary, hist = get_industry_market_summary(industry)
        st.markdown(f"*ETF Market Summary*: {market_summary}")
        if hist is not None:
            fig2, ax2 = plt.subplots()
            hist["Close"].plot(ax=ax2, title=f"ETF Price Trend - {industry}")
            st.pyplot(fig2)
            fig2.savefig("etf_chart.png")

    # Tab 3: Industry Growth Rate
    with tabs[2]:
        _, growth_df = get_market_insights(industry)
        if growth_df is not None:
            fig3, ax3 = plt.subplots()
            top_df = growth_df.sort_values(by="GrowthRate", ascending=False).head(10)
            ax3.barh(top_df["Industry"], top_df["GrowthRate"])
            ax3.set_title("Top 10 Growing Industries")
            ax3.invert_yaxis()
            st.pyplot(fig3)
            fig3.savefig("growth_chart.png")
            
    # Tab 4: Reddit Trends
    with tabs[3]:
        reddit_posts = get_reddit_posts(industry)
        reddit_db = build_vector_db_from_texts(reddit_posts)
        reddit_context = "\n".join([doc.page_content for doc in retrieve_relevant_docs(reddit_db, industry)])
        st.markdown(f"*Reddit Trends*: {reddit_context}")
    st.markdown("---")
    st.markdown("<h3 style='color:#1f4e79;'>Download Report</h3>", unsafe_allow_html=True)      
    st.markdown("Click the button below to download your report as a PDF.")
    st.markdown("This report includes charts and insights based on your inputs.")
    st.markdown("**Note**: The report is generated based on the latest data available and may include estimates.")
    st.markdown("**Disclaimer**: This report is for informational purposes only and should not be considered financial advice.")

    # Export to PDF
    def export_to_pdf(report_text, file_name):
        formatted_text = report_text.replace("\n", "<br>")
        html_content = f"""
        <html>
        <body>
            <h1>AI-Generated Business Report</h1>
            <div>{formatted_text}</div>
            <img src='trend_chart.png'><br>
            <img src='growth_chart.png'><br>
            <img src='etf_chart.png'><br>
        </body>
        </html>
        """
        with open(file_name, "w+b") as f:
            pisa.CreatePDF(html_content, dest=f)

    # st.markdown("### Download Report")
    export_to_pdf(result, "business_report.pdf")
    with open("business_report.pdf", "rb") as f:
        st.download_button("Download PDF with Charts", f, "business_report.pdf", mime="application/pdf")

    # Back button
    if st.button("Back to Home"):
        st.session_state.page = "Home"
        st.rerun()

# Fallback
else:
    st.info("Please generate a report from the Home page first.")
    