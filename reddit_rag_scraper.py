import praw
import os
from dotenv import load_dotenv

load_dotenv()

reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT"),
)

# Industry → relevant subreddits
INDUSTRY_SUBREDDITS = {
    "fintech":          ["fintech", "personalfinance", "investing", "startups", "entrepreneur"],
    "technology":       ["technology", "startups", "tech", "programming", "entrepreneur"],
    "healthcare":       ["healthcare", "healthIT", "medicine", "startups", "biotech"],
    "renewable energy": ["RenewableEnergy", "solar", "sustainability", "CleanEnergy", "environment"],
    "blockchain":       ["CryptoCurrency", "ethereum", "Bitcoin", "blockchain", "web3"],
    "cybersecurity":    ["netsec", "cybersecurity", "AskNetsec", "hacking", "sysadmin"],
    "ecommerce":        ["ecommerce", "entrepreneur", "Shopify", "startups", "dropshipping"],
    "ai":               ["artificial", "MachineLearning", "deeplearning", "ChatGPT", "startups"],
    "real estate":      ["realestate", "investing", "landlord", "REI", "realestateinvesting"],
    "saas":             ["SaaS", "startups", "entrepreneur", "microsaas", "indiehackers"],
    "retail":           ["retail", "ecommerce", "entrepreneur", "smallbusiness", "startups"],
    "food":             ["food", "restaurantowners", "entrepreneur", "smallbusiness", "FoodTech"],
    "education":        ["education", "edtech", "startups", "entrepreneur", "learnprogramming"],
    "logistics":        ["logistics", "supplychain", "entrepreneur", "startups", "business"],
    "crypto":           ["CryptoCurrency", "ethereum", "Bitcoin", "defi", "web3"],
    "gaming":           ["gamedev", "gaming", "indiegaming", "startups", "entrepreneur"],
    "marketing":        ["marketing", "digitalmarketing", "entrepreneur", "startups", "SEO"],
    "insurance":        ["insurance", "personalfinance", "investing", "startups", "fintech"],
    "travel":           ["travel", "solotravel", "entrepreneur", "startups", "tourism"],
    "media":            ["media", "journalism", "contentcreation", "startups", "entrepreneur"],
}

DEFAULT_SUBREDDITS = ["startups", "entrepreneur", "investing", "business", "smallbusiness"]


def _get_targeted_subreddits(industry):
    industry_lower = industry.lower()
    for key, subs in INDUSTRY_SUBREDDITS.items():
        if key in industry_lower or industry_lower in key:
            seen, merged = set(), []
            for s in subs + DEFAULT_SUBREDDITS:
                if s not in seen:
                    seen.add(s)
                    merged.append(s)
            return merged
    return DEFAULT_SUBREDDITS


def _get_top_comment(post):
    try:
        post.comments.replace_more(limit=0)
        comments = [c for c in post.comments.list() if hasattr(c, "score") and c.score > 2]
        if not comments:
            return ""
        best = max(comments, key=lambda c: c.score)
        return best.body[:300].strip()
    except Exception:
        return ""


def _format_post(post, with_comment=True):
    parts = [f"[r/{post.subreddit.display_name}] {post.title}"]
    body = (post.selftext or "").strip()
    if len(body) > 30:
        parts.append(body[:600])
    if with_comment:
        comment = _get_top_comment(post)
        if comment:
            parts.append(f"Top comment: {comment}")
    parts.append(f"(Score: {post.score} | Comments: {post.num_comments})")
    return "\n".join(parts)


def build_reddit_query(industry, goal="", target_market=""):
    """
    Build a specific, focused search query from the business context.
    Combines the most meaningful terms from industry + goal to find relevant posts.
    """
    # Extract key noun phrases from the goal (first 100 chars to stay focused)
    goal_snippet = goal[:100].strip() if goal else ""

    # Build a targeted query — specific enough to get relevant results
    if goal_snippet:
        query = f"{industry} {goal_snippet}"
    else:
        query = industry

    # Trim to Reddit's practical query length limit
    return query[:200]


def get_reddit_posts(industry, goal="", target_market="", limit=25):
    """Returns list of formatted text strings for RAG embedding."""
    # Use a specific query, not just the industry name
    specific_query = build_reddit_query(industry, goal, target_market)
    # Also prepare a broader fallback query
    broad_query = industry

    posts_data = []
    seen_ids = set()

    def add(post):
        if post.stickied or post.id in seen_ids or post.score < 2:
            return
        seen_ids.add(post.id)
        posts_data.append(_format_post(post))

    targeted = _get_targeted_subreddits(industry)
    sub_str = "+".join(targeted[:6])

    # 1. Most specific query — exact business context on r/all
    try:
        for post in reddit.subreddit("all").search(
            specific_query, sort="relevance", limit=limit, time_filter="year"
        ):
            add(post)
    except Exception:
        pass

    # 2. Specific query in targeted subreddits (hot)
    try:
        for post in reddit.subreddit(sub_str).search(specific_query, sort="hot", limit=15):
            add(post)
    except Exception:
        pass

    # 3. Broader industry query in targeted subreddits if we need more results
    if len(posts_data) < 10:
        try:
            for post in reddit.subreddit(sub_str).search(
                broad_query, sort="top", limit=20, time_filter="month"
            ):
                add(post)
        except Exception:
            pass

    return posts_data[:35] if posts_data else [f"No Reddit discussions found for '{specific_query}'."]


def get_reddit_posts_with_metadata(industry, goal="", target_market="", limit=20):
    """Returns structured post dicts for UI display."""
    specific_query = build_reddit_query(industry, goal, target_market)
    broad_query = industry

    posts = []
    seen_ids = set()

    targeted = _get_targeted_subreddits(industry)
    sub_str = "+".join(targeted[:6])

    sources = [
        ("all",   specific_query, "relevance", limit, "year"),
        (sub_str, specific_query, "hot",       15,    None),
        (sub_str, broad_query,    "top",       15,    "month"),
    ]

    for sub, query, sort, lim, time_f in sources:
        try:
            kwargs = {"sort": sort, "limit": lim}
            if time_f:
                kwargs["time_filter"] = time_f
            for post in reddit.subreddit(sub).search(query, **kwargs):
                if post.stickied or post.id in seen_ids or post.score < 2:
                    continue
                seen_ids.add(post.id)
                posts.append({
                    "title":     post.title,
                    "subreddit": post.subreddit.display_name,
                    "score":     post.score,
                    "comments":  post.num_comments,
                    "body":      (post.selftext or "")[:300].strip(),
                    "url":       f"https://reddit.com{post.permalink}",
                })
        except Exception:
            continue

    posts.sort(key=lambda x: x["score"], reverse=True)
    return posts[:15]
