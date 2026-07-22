# 🎬 CineMatch — AI Movie Recommendation System

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas)

A content-based **Movie Recommendation System** built using Machine Learning, TF-IDF Vectorization, and Cosine Similarity — deployed as a beautiful interactive web app using **Streamlit**.

> 🔗 **Live Demo:** [Click here to try the app](https://your-app-link.streamlit.app)

---

## ✨ Features

- 🎯 Content-based movie recommendations
- 🤖 TF-IDF Vectorization + Cosine Similarity algorithm
- ⚡ No pre-built model files needed — builds automatically at runtime
- 🎨 Beautiful dark-themed UI with glassmorphism design
- 📊 Shows rating, year, genres & overview for each recommendation
- 🔍 Search from 4,800+ movies

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data loading & preprocessing |
| Scikit-learn | TF-IDF Vectorizer + Cosine Similarity |
| Streamlit | Web app framework & deployment |
| TMDB Dataset | Movie data (genres, keywords, overview) |

---

## 📂 Project Structure

```
Movie-Recommendation-System/
│
├── app.py                  # Streamlit app (main file)
├── tmdb_5000_movies.csv    # TMDB movie dataset
├── requirements.txt        # Python dependencies
└── README.md
```

---

## ⚙️ How It Works

1. **Data Loading** — Reads `tmdb_5000_movies.csv` (4,800+ movies)
2. **Feature Engineering** — Combines `overview`, `genres`, and `keywords` into a single `tags` column
3. **Vectorization** — Applies `CountVectorizer` (top 5000 features, English stop-words removed)
4. **Similarity** — Computes **Cosine Similarity** between all movie vectors
5. **Caching** — Uses `@st.cache_data` so the model builds only once per session
6. **Recommendation** — Returns top 8 most similar movies for any selected title

---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/KaranamDeepak/Movie-Recommendation-System.git
cd Movie-Recommendation-System

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

Open your browser at **http://localhost:8501** 🎉

---

## 📦 Requirements

```
streamlit
pandas
scikit-learn
numpy
```

---

## 📸 Screenshots

> *(Add a screenshot of your running app here)*

---

## 🙋‍♂️ Author

**Karanam Deepak**  
[![GitHub](https://img.shields.io/badge/GitHub-KaranamDeepak-181717?style=flat&logo=github)](https://github.com/KaranamDeepak)
