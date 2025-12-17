# app_ui_redesign.py
# FINAL PROJECT - Recommendation System (Books) - CLEAN DATA ONLY
#
# Run:
#   pip install streamlit pandas numpy scikit-learn matplotlib
#   streamlit run app_ui_redesign.py
#
# Expected files:
#   ./data/books_clean.csv
#   ./data/ratings_clean.csv
#
# Note:
# - Preprocessing / cleaning is assumed to be done in a separate script.
# - This Streamlit app ONLY loads the cleaned files and trains/evaluates the model.
#
# UI: Redesigned dashboard-style layout (cards, modern tabs, clean sidebar)
# Logic/model: unchanged (Item-Item CF cosine)

import os
import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = "data"
BOOKS_CLEAN_CSV = os.path.join(DATA_DIR, "books_clean.csv")
RATINGS_CLEAN_CSV = os.path.join(DATA_DIR, "ratings_clean.csv")

RANDOM_SEED = 42


# ---------------------------
# Helpers
# ---------------------------
def _safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def normalize_text(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).lower().strip()
    return " ".join(s.split())


@st.cache_data(show_spinner=False)
def load_clean_data():
    books = _safe_read_csv(BOOKS_CLEAN_CSV)
    ratings = _safe_read_csv(RATINGS_CLEAN_CSV)

    # basic schema checks
    if "book_id" not in books.columns:
        raise ValueError("books_clean.csv must contain column: book_id")
    for c in ["user_id", "book_id", "rating"]:
        if c not in ratings.columns:
            raise ValueError("ratings_clean.csv must contain columns: user_id, book_id, rating")

    # ensure numeric ids (avoid silent merge mismatch)
    books["book_id"] = pd.to_numeric(books["book_id"], errors="coerce")
    ratings["book_id"] = pd.to_numeric(ratings["book_id"], errors="coerce")
    ratings["user_id"] = pd.to_numeric(ratings["user_id"], errors="coerce")

    books = books.dropna(subset=["book_id"]).copy()
    ratings = ratings.dropna(subset=["book_id", "user_id"]).copy()

    books["book_id"] = books["book_id"].astype(int)
    ratings["book_id"] = ratings["book_id"].astype(int)
    ratings["user_id"] = ratings["user_id"].astype(int)

    # rating bounds safety (in case preprocessing missed)
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce").fillna(0).astype(int)
    ratings = ratings[(ratings["rating"] >= 1) & (ratings["rating"] <= 5)]
    ratings = ratings.drop_duplicates(subset=["user_id", "book_id"], keep="last").reset_index(drop=True)

    return books, ratings


@st.cache_data(show_spinner=False)
def build_tfidf(books: pd.DataFrame):
    # Prefer precomputed item_text if exists, else build from available columns
    if "item_text" in books.columns:
        item_text = books["item_text"].fillna("").astype(str)
    else:
        # Robust: if a column is missing, use empty strings of same length
        empty = pd.Series([""] * len(books), index=books.index)

        title = books["title"].map(normalize_text) if "title" in books.columns else empty
        authors = books["authors"].map(normalize_text) if "authors" in books.columns else empty
        tags_text = books["tags_text"].map(normalize_text) if "tags_text" in books.columns else empty

        item_text = (title + " " + authors + " " + tags_text).astype(str).str.strip()

    vectorizer = TfidfVectorizer(max_features=20000, stop_words="english", ngram_range=(1, 2))
    tfidf = vectorizer.fit_transform(item_text.values)
    return tfidf


def plot_hist(series, title, bins=20):
    fig = plt.figure()
    plt.hist(series, bins=bins)
    plt.title(title)
    plt.xlabel(series.name if series.name else "")
    plt.ylabel("Count")
    st.pyplot(fig)
    plt.close(fig)


def plot_rating_bar(ratings):
    fig = plt.figure(figsize=(6, 3))  # thu nhỏ chiều dọc
    counts = ratings.value_counts().sort_index()

    plt.bar(counts.index, counts.values, width=0.6)

    plt.xticks(
        [0, 1, 2, 3, 4, 5],
        ["0", "1", "2", "3", "4", "5"],
    )
    plt.xlim(0.5, 5.5)

    plt.xlabel("Rating")
    plt.ylabel("Count")
    plt.title("Rating distribution")

    st.pyplot(fig)
    plt.close(fig)



def plot_bar(x, y, title, xlabel="", ylabel=""):
    fig = plt.figure()
    plt.bar(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------
# Model: Item-Item CF (cosine)
# ---------------------------
@st.cache_data(show_spinner=False)
def train_test_split_by_user(ratings: pd.DataFrame, test_size=0.2, min_per_user=5):
    rng = np.random.default_rng(RANDOM_SEED)
    train_rows = []
    test_rows = []

    for uid, grp in ratings.groupby("user_id"):
        if len(grp) < min_per_user:
            train_rows.append(grp)
            continue
        idx = np.arange(len(grp))
        rng.shuffle(idx)
        n_test = max(1, int(round(len(grp) * test_size)))
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]
        test_rows.append(grp.iloc[test_idx])
        train_rows.append(grp.iloc[train_idx])

    train_df = pd.concat(train_rows, ignore_index=True) if train_rows else ratings.copy()
    test_df = pd.concat(test_rows, ignore_index=True) if test_rows else ratings.iloc[0:0].copy()
    return train_df, test_df


@st.cache_data(show_spinner=False)
def build_user_item_matrix(train_ratings: pd.DataFrame):
    users = np.sort(train_ratings["user_id"].unique())
    items = np.sort(train_ratings["book_id"].unique())

    user_index = {u: i for i, u in enumerate(users)}
    item_index = {b: j for j, b in enumerate(items)}

    R = np.zeros((len(users), len(items)), dtype=np.float32)

    for row in train_ratings.itertuples(index=False):
        ui = user_index[row.user_id]
        ij = item_index[row.book_id]
        R[ui, ij] = float(row.rating)

    return R, user_index, item_index, users, items


@st.cache_data(show_spinner=False)
def compute_item_similarity(R: np.ndarray):
    R2 = R.copy()
    row_sum = R2.sum(axis=1)
    row_cnt = (R2 > 0).sum(axis=1)
    row_mean = np.divide(row_sum, np.maximum(row_cnt, 1), dtype=np.float32)
    for i in range(R2.shape[0]):
        mask = R2[i, :] > 0
        if mask.any():
            R2[i, mask] = R2[i, mask] - row_mean[i]
    S = cosine_similarity(R2.T)
    np.fill_diagonal(S, 0.0)
    return S


def predict_rating_itemcf(user_row: int, item_col: int, R: np.ndarray, S: np.ndarray, k_neighbors: int = 50):
    user_ratings = R[user_row, :]
    rated_mask = user_ratings > 0
    if not rated_mask.any():
        return 0.0

    sims = S[item_col, :] * rated_mask

    if k_neighbors is not None and k_neighbors > 0:
        top_idx = np.argsort(sims)[::-1][:k_neighbors]
    else:
        top_idx = np.argsort(sims)[::-1]

    top_sims = sims[top_idx]
    top_r = user_ratings[top_idx]

    denom = np.abs(top_sims).sum()
    if denom <= 1e-8:
        return float(np.clip(user_ratings[rated_mask].mean(), 1.0, 5.0))

    pred = float((top_sims * top_r).sum() / denom)
    return float(np.clip(pred, 1.0, 5.0))


def recommend_for_user(
    user_id: int,
    R: np.ndarray,
    S: np.ndarray,
    user_index: dict,
    items: np.ndarray,
    k: int = 10,
    k_neighbors: int = 50,
):
    if user_id not in user_index:
        return []

    urow = user_index[user_id]
    user_ratings = R[urow, :]
    seen = set(np.where(user_ratings > 0)[0].tolist())

    preds = []
    for j in range(R.shape[1]):
        if j in seen:
            continue
        p = predict_rating_itemcf(urow, j, R, S, k_neighbors=k_neighbors)
        preds.append((j, p))

    preds.sort(key=lambda x: x[1], reverse=True)
    top = preds[:k]
    return [(int(items[j]), float(score)) for j, score in top]


@st.cache_data(show_spinner=False)
def evaluate_model(train_ratings: pd.DataFrame, test_ratings: pd.DataFrame, k: int = 20,
                   relevant_threshold: float = 4.0, k_neighbors: int = 50):
    R, user_index, item_index, users, items = build_user_item_matrix(train_ratings)
    S = compute_item_similarity(R)

    # RMSE/MAE
    y_true, y_pred = [], []
    for row in test_ratings.itertuples(index=False):
        if row.user_id not in user_index:
            continue
        if row.book_id not in item_index:
            continue
        u = user_index[row.user_id]
        i = item_index[row.book_id]
        pred = predict_rating_itemcf(u, i, R, S, k_neighbors=k_neighbors)
        y_true.append(float(row.rating))
        y_pred.append(float(pred))

    if len(y_true) == 0:
        rmse = float("nan")
        mae = float("nan")
    else:
        err = np.array(y_true) - np.array(y_pred)
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mae = float(np.mean(np.abs(err)))

    # Precision@K / Recall@K
    test_rel = test_ratings[test_ratings["rating"] >= relevant_threshold]
    rel_by_user = test_rel.groupby("user_id")["book_id"].apply(set).to_dict()

    precisions = []
    recalls = []
    for uid, rel_items in rel_by_user.items():
        recs = recommend_for_user(int(uid), R, S, user_index, items, k=k, k_neighbors=k_neighbors)
        rec_items = [bid for bid, _ in recs]
        if len(rec_items) == 0:
            continue
        hit = len(set(rec_items) & set(rel_items))
        precisions.append(hit / float(k))
        recalls.append(hit / float(len(rel_items)) if len(rel_items) > 0 else 0.0)

    p_at_k = float(np.mean(precisions)) if len(precisions) else float("nan")
    r_at_k = float(np.mean(recalls)) if len(recalls) else float("nan")

    return {"rmse": rmse, "mae": mae, "p_at_k": p_at_k, "r_at_k": r_at_k}


# ---------------------------
# Streamlit UI (Redesigned)
# ---------------------------
st.set_page_config(page_title="Book Recommendation System (ItemCF)", layout="wide")

st.markdown(
    """
<style>
/* App background */
.stApp{
    background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 45%, #f8fafc 100%);
}

/* Sidebar */
section[data-testid="stSidebar"]{
    background: #ffffff;
    border-right: 1px solid #e5e7eb;
}

/* Header card */
.hero{
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 18px;
    padding: 18px 18px 10px 18px;
    box-shadow: 0 10px 30px rgba(17, 24, 39, 0.06);
    margin-bottom: 12px;
}
.hero h1{
    font-size: 34px;
    font-weight: 800;
    margin: 0;
    color: #111827;
}
.hero p{
    margin: 6px 0 0 0;
    color: #6b7280 !important;
    font-size: 15px;
}

/* Metric cards */
.metric-grid{
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 12px;
    margin: 10px 0 16px 0;
}
.metric-card{
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    padding: 14px 14px 10px 14px;
    box-shadow: 0 8px 20px rgba(17, 24, 39, 0.06);
}
.metric-label{
    font-size: 12px;
    color: #6b7280 !important;
    letter-spacing: .02em;
    text-transform: uppercase;
}
.metric-value{
    font-size: 26px;
    font-weight: 800;
    margin-top: 4px;
    color: #111827;
}
.metric-sub{
    font-size: 12px;
    color: #6b7280 !important;
    margin-top: 4px;
}

/* Tabs */
.stTabs [role="tab"]{
    font-size: 15px;
    padding: 10px 12px;
}

/* Dataframe */
div[data-testid="stDataFrame"]{
    background: #ffffff;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
    padding: 6px;
}

/* Buttons */
.stButton > button{
    border-radius: 12px;
    padding: 10px 14px;
    font-weight: 700;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <h1>📚 Book Recommendation System</h1>
</div>
""",
    unsafe_allow_html=True,
)

# Sidebar (model/eval settings only)
st.sidebar.header("⚙️ Settings (Model/Eval)")
test_size = st.sidebar.slider("Test size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
top_k = st.sidebar.selectbox("Top-K recommendations", [5, 10, 20, 30, 50], index=2)
k_neighbors = st.sidebar.selectbox("K neighbors", [20, 50, 100, 200], index=1)
relevant_threshold = st.sidebar.selectbox("Relevant threshold", [3.0, 3.5, 4.0, 4.5], index=2)

# Load clean data
try:
    books, ratings = load_clean_data()
except Exception as e:
    st.error(str(e))
    st.stop()

# Optional TF-IDF (requirement / demo)
_ = build_tfidf(books)

# KPI cards
users_count = int(ratings["user_id"].nunique()) if "user_id" in ratings.columns else 0
items_in_ratings = int(ratings["book_id"].nunique()) if "book_id" in ratings.columns else 0
avg_rating = float(ratings["rating"].mean()) if "rating" in ratings.columns and len(ratings) else float("nan")

st.markdown(
    f"""
<div class="metric-grid">
  <div class="metric-card">
    <div class="metric-label">Books (clean)</div>
    <div class="metric-value">{len(books):,}</div>
    <div class="metric-sub">Số lượng item</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Ratings (clean)</div>
    <div class="metric-value">{len(ratings):,}</div>
    <div class="metric-sub">user–item interactions</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Users</div>
    <div class="metric-value">{users_count:,}</div>
    <div class="metric-sub">unique user_id</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Avg Rating</div>
    <div class="metric-value">{(avg_rating if not math.isnan(avg_rating) else 0):.2f}</div>
    <div class="metric-sub">trung bình rating</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["📊 Khám phá dữ liệu", "🎯 Gợi ý sách", "📏 Đánh giá mô hình"])

with tab1:

    st.markdown("**Phân bố rating**")
    plot_rating_bar(ratings["rating"])

    # Top 10 sách có rating trung bình cao nhất
    # (ưu tiên dùng cột average_rating nếu có; nếu không thì tính từ ratings)
    # ===== 2 biểu đồ CHUNG 1 DÒNG =====
    col1, col2 = st.columns(2)

    # --- Cột trái: Top sách theo rating trung bình ---
    with col1:
        st.markdown("**Top sách theo rating trung bình**")

        b = books.copy()
        if "average_rating" in b.columns:
            b["avg_rating"] = pd.to_numeric(b["average_rating"], errors="coerce")
        else:
            avg_by_book = (
                ratings.groupby("book_id")["rating"]
                .mean()
                .rename("avg_rating")
                .reset_index()
            )
            b = b.merge(avg_by_book, on="book_id", how="left")

        b["avg_rating"] = b["avg_rating"].fillna(0)
        b = b.sort_values("avg_rating", ascending=False).head(10)

        title_col = "title" if "title" in b.columns else "book_id"
        plot_bar(
            b[title_col].astype(str).tolist(),
            b["avg_rating"].round(2).tolist(),
            "Top 10 books by average rating",
            xlabel="Book",
            ylabel="Avg rating",
        )

    # --- Cột phải: Tần suất theo ngôn ngữ ---
    with col2:
        if "language_code" in books.columns:
            st.markdown("**Tần suất theo ngôn ngữ**")
            lang_counts = (
                books["language_code"]
                .fillna("unknown")
                .value_counts()
                .head(10)
            )

            plot_bar(
                lang_counts.index.tolist(),
                lang_counts.values.tolist(),
                "Language distribution",
                xlabel="Language",
                ylabel="Count",
            )


    with st.expander("📌 Xem nhanh schema (cột) của data", expanded=False):
        st.write("**books_clean.csv columns:**", list(books.columns))
        st.write("**ratings_clean.csv columns:**", list(ratings.columns))

with tab2:
    st.subheader("Gợi ý cho người dùng (Item-Item CF)")
    st.caption("Chọn user_id trong tập train. User mới không có lịch sử rating → model không gợi ý được.")

    train_df, _test_df = train_test_split_by_user(ratings, test_size=test_size, min_per_user=5)
    R, user_index, item_index, users, items = build_user_item_matrix(train_df)
    S = compute_item_similarity(R)

    # ---- INPUT + BUTTON (TRÊN) ----
    uid = st.selectbox("user_id", options=[int(u) for u in users[:5000]], index=0)
    run = st.button("✨ Recommend", use_container_width=True)

    # ---- KẾT QUẢ (DƯỚI) ----
    if run:
        recs = recommend_for_user(
            int(uid),
            R, S, user_index, items,
            k=int(top_k),
            k_neighbors=int(k_neighbors)
        )

        if not recs:
            st.warning("Không có gợi ý (user mới hoặc không đủ dữ liệu).")
        else:
            rec_df = pd.DataFrame(recs, columns=["book_id", "pred_score"])

            join_cols = [
                c for c in
                ["book_id", "title", "authors", "original_publication_year", "language_code", "n_ratings"]
                if c in books.columns
            ]
            if join_cols:
                rec_df = rec_df.merge(books[join_cols], on="book_id", how="left")

            st.markdown("### ✅ Danh sách gợi ý")
            st.dataframe(
                rec_df.drop(columns=["pred_score"], errors="ignore"),
                use_container_width=True,
                hide_index=True
            )

            st.markdown("### 🧾 Lịch sử rating của user (train)")
            hist = train_df[train_df["user_id"] == int(uid)].copy()
            if "title" in books.columns:
                hist = hist.merge(books[["book_id", "title"]], on="book_id", how="left")

            hist = hist.sort_values("rating", ascending=False).head(30)
            st.dataframe(hist, use_container_width=True, hide_index=True)

with tab3:

    train_df, test_df = train_test_split_by_user(ratings, test_size=test_size, min_per_user=5)

    cbtn1, cbtn2 = st.columns([1, 3])
    run_eval = st.button("📏 Run evaluation", use_container_width=True)

    if run_eval:
        metrics = evaluate_model(
            train_df,
            test_df,
            k=int(top_k),
            relevant_threshold=float(relevant_threshold),
            k_neighbors=int(k_neighbors),
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("RMSE", f"{metrics['rmse']:.4f}" if not math.isnan(metrics["rmse"]) else "N/A")
        c2.metric("MAE", f"{metrics['mae']:.4f}" if not math.isnan(metrics["mae"]) else "N/A")
        c3.metric(f"Precision@{top_k}", f"{metrics['p_at_k']:.4f}" if not math.isnan(metrics["p_at_k"]) else "N/A")
        c4.metric(f"Recall@{top_k}", f"{metrics['r_at_k']:.4f}" if not math.isnan(metrics["r_at_k"]) else "N/A")

