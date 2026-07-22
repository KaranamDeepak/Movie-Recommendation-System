import streamlit as st
import pandas as pd
import ast
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineMatch — AI Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── CUSTOM CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif !important; }

.stApp {
    background: linear-gradient(135deg, #0a0a1a 0%, #12082a 50%, #0d1117 100%);
    min-height: 100vh;
}

/* Hide streamlit default elements */
#MainMenu, footer { visibility: hidden; }
.stDeployButton { display: none; }

/* Hero Section */
.hero {
    text-align: center;
    padding: 3rem 1rem 1.5rem;
}
.hero-title {
    font-size: 3.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a78bfa, #ec4899, #f97316);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
    line-height: 1.1;
}
.hero-sub {
    color: rgba(255,255,255,0.5);
    font-size: 1.1rem;
    font-weight: 400;
}

/* Stats Cards */
.stat-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.2rem;
    text-align: center;
    backdrop-filter: blur(20px);
}
.stat-num {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #a78bfa, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.stat-label {
    color: rgba(255,255,255,0.45);
    font-size: 0.82rem;
    margin-top: 2px;
}

/* Selected Movie Card */
.selected-card {
    background: linear-gradient(135deg, rgba(167,139,250,0.1), rgba(236,72,153,0.08));
    border: 1px solid rgba(167,139,250,0.35);
    border-radius: 20px;
    padding: 1.8rem;
    margin: 1.5rem 0;
    position: relative;
    overflow: hidden;
}
.selected-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #a78bfa, #ec4899, #f97316);
}
.selected-badge {
    font-size: 0.7rem;
    font-weight: 600;
    color: #a78bfa;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 0.5rem;
}
.movie-title-lg {
    font-size: 1.5rem;
    font-weight: 700;
    color: #fff;
    margin-bottom: 0.6rem;
}
.movie-title-sm {
    font-size: 1.05rem;
    font-weight: 600;
    color: #fff;
    margin-bottom: 0.4rem;
}
.movie-meta {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
    margin-bottom: 0.7rem;
}
.rating-badge {
    background: linear-gradient(135deg, #a78bfa, #ec4899);
    color: white;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 700;
}
.year-badge {
    background: rgba(255,255,255,0.1);
    color: rgba(255,255,255,0.7);
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
}
.genre-tag {
    background: rgba(167,139,250,0.15);
    color: #c4b5fd;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.75rem;
    border: 1px solid rgba(167,139,250,0.25);
}
.movie-overview {
    color: rgba(255,255,255,0.65);
    font-size: 0.9rem;
    line-height: 1.6;
}

/* Recommendation Cards */
.rec-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.4rem;
    margin-bottom: 1rem;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
.rec-card:hover {
    background: rgba(255,255,255,0.08);
    border-color: rgba(167,139,250,0.35);
    transform: translateY(-3px);
    box-shadow: 0 20px 40px rgba(0,0,0,0.4);
}

/* Section Header */
.section-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: #fff;
    padding-bottom: 0.6rem;
    margin: 2rem 0 1rem;
    border-bottom: 1px solid rgba(167,139,250,0.25);
}

/* Rank badge */
.rank-badge {
    display: inline-block;
    width: 26px;
    height: 26px;
    background: linear-gradient(135deg, #a78bfa, #ec4899);
    border-radius: 50%;
    text-align: center;
    line-height: 26px;
    font-size: 0.75rem;
    font-weight: 700;
    color: white;
    margin-right: 8px;
    flex-shrink: 0;
}

/* Selectbox */
.stSelectbox label {
    color: rgba(255,255,255,0.7) !important;
    font-weight: 500 !important;
}
div[data-baseweb="select"] > div {
    background-color: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 12px !important;
    color: white !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #a78bfa, #ec4899) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.65rem 1.5rem !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    width: 100% !important;
    transition: all 0.3s ease !important;
    letter-spacing: 0.3px !important;
}
.stButton > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 30px rgba(167,139,250,0.35) !important;
}

/* Spinner */
.stSpinner > div {
    border-top-color: #a78bfa !important;
}

/* Divider */
.divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.08);
    margin: 1.5rem 0;
}

/* Footer */
.footer {
    text-align: center;
    color: rgba(255,255,255,0.25);
    font-size: 0.8rem;
    padding: 2rem 0 1rem;
}
</style>
""", unsafe_allow_html=True)


# ─── DATA LOADING & MODEL BUILDING ───────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_build_model():
    """Load CSV and build cosine similarity model — runs once, then cached."""
    df = pd.read_csv('tmdb_5000_movies.csv')

    def safe_extract(text):
        """Safely extract names from JSON-like string columns."""
        try:
            items = ast.literal_eval(str(text))
            if isinstance(items, list):
                return ' '.join([
                    item['name'].replace(' ', '').lower()
                    for item in items if isinstance(item, dict) and 'name' in item
                ])
        except Exception:
            pass
        return ''

    # Build feature tags from genres + keywords + overview
    df['genres_clean']   = df['genres'].apply(safe_extract)
    df['keywords_clean'] = df['keywords'].apply(safe_extract)
    df['overview_clean'] = df['overview'].fillna('').str.lower()
    df['tags']           = (
        df['overview_clean'] + ' ' +
        df['genres_clean'] + ' ' +
        df['keywords_clean']
    )

    # Vectorise and compute cosine similarity
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors    = cv.fit_transform(df['tags']).toarray()
    similarity = cosine_similarity(vectors)

    return df.reset_index(drop=True), similarity


# ─── HELPER FUNCTIONS ─────────────────────────────────────────────────────────
def get_genres(text, max_n=3):
    try:
        items = ast.literal_eval(str(text))
        return [i['name'] for i in items if isinstance(i, dict) and 'name' in i][:max_n]
    except Exception:
        return []

def get_year(date_str):
    try:
        return str(date_str)[:4] if pd.notna(date_str) and str(date_str) != 'nan' else 'N/A'
    except Exception:
        return 'N/A'

def truncate(text, length=160):
    text = str(text) if pd.notna(text) else 'No overview available.'
    return text if len(text) <= length else text[:length].rsplit(' ', 1)[0] + '…'

def recommend_movies(title, df, similarity, n=8):
    mask = df['title'].str.lower() == title.lower()
    if not mask.any():
        return []
    idx       = df[mask].index[0]
    distances = sorted(enumerate(similarity[idx]), key=lambda x: x[1], reverse=True)
    return [df.iloc[i] for i, _ in distances[1:n + 1]]

def render_meta(genres, year, rating):
    rating_html = f'<span class="rating-badge">⭐ {rating:.1f}</span>' if rating else ''
    year_html   = f'<span class="year-badge">📅 {year}</span>'
    genre_html  = ''.join(f'<span class="genre-tag">{g}</span>' for g in genres)
    return f'<div class="movie-meta">{rating_html}{year_html}{genre_html}</div>'


# ─── UI ───────────────────────────────────────────────────────────────────────

# Hero
st.markdown("""
<div class="hero">
    <div class="hero-title">🎬 CineMatch</div>
    <div class="hero-sub">AI-powered movie recommendations — discover your next favourite film</div>
</div>
""", unsafe_allow_html=True)

# Load model
with st.spinner("⚙️  Building recommendation engine…"):
    df, similarity = load_and_build_model()

# ── Stats Row ──
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-num">{len(df):,}</div>
        <div class="stat-label">Movies in Database</div>
    </div>""", unsafe_allow_html=True)
with c2:
    avg = df['vote_average'].mean()
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-num">{avg:.1f} ⭐</div>
        <div class="stat-label">Average Rating</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-num">20+</div>
        <div class="stat-label">Genre Categories</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Search + Button ──
movie_list     = sorted(df['title'].dropna().unique().tolist())
col_sel, col_btn = st.columns([5, 1])
with col_sel:
    selected = st.selectbox("🔍  Search a movie to get recommendations", movie_list)
with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    get_recs = st.button("✨ Recommend")

# ── Selected Movie Info ──
if selected:
    row    = df[df['title'] == selected].iloc[0]
    genres = get_genres(row['genres'])
    year   = get_year(row.get('release_date'))
    rating = row.get('vote_average', 0)
    meta   = render_meta(genres, year, rating)

    st.markdown(f"""
    <div class="selected-card">
        <div class="selected-badge">📽 Selected Movie</div>
        <div class="movie-title-lg">{row['title']}</div>
        {meta}
        <div class="movie-overview">{row.get('overview', 'No overview available.')}</div>
    </div>
    """, unsafe_allow_html=True)

# ── Recommendations ──
if get_recs and selected:
    with st.spinner("🎯  Finding best matches for you…"):
        recs = recommend_movies(selected, df, similarity)

    if recs:
        st.markdown('<div class="section-title">🎯 Recommended Movies</div>', unsafe_allow_html=True)
        left, right = st.columns(2)
        for i, movie in enumerate(recs):
            genres = get_genres(movie['genres'])
            year   = get_year(movie.get('release_date'))
            rating = movie.get('vote_average', 0)
            meta   = render_meta(genres, year, rating)
            card   = f"""
            <div class="rec-card">
                <div style="display:flex;align-items:flex-start;gap:10px;">
                    <div class="rank-badge">{i+1}</div>
                    <div style="flex:1;">
                        <div class="movie-title-sm">{movie['title']}</div>
                        {meta}
                        <div class="movie-overview">{truncate(movie.get('overview'))}</div>
                    </div>
                </div>
            </div>"""
            (left if i % 2 == 0 else right).markdown(card, unsafe_allow_html=True)
    else:
        st.warning("⚠️ No recommendations found. Try a different movie!")

# Footer
st.markdown("""
<hr class="divider">
<div class="footer">
    Built with ❤️ using Streamlit &amp; scikit-learn &nbsp;|&nbsp; Data: TMDB 5000 Movies
</div>
""", unsafe_allow_html=True)
