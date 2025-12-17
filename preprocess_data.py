# preprocess_data.py
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_DIR = "data"
BOOKS_CSV = os.path.join(DATA_DIR, "books.csv")
RATINGS_CSV = os.path.join(DATA_DIR, "ratings.csv")
TAGS_CSV = os.path.join(DATA_DIR, "tags.csv")
BOOK_TAGS_CSV = os.path.join(DATA_DIR, "book_tags.csv")

OUT_BOOKS = os.path.join(DATA_DIR, "books_clean.csv")
OUT_RATINGS = os.path.join(DATA_DIR, "ratings_clean.csv")


def normalize_text(s):
    if pd.isna(s):
        return ""
    return " ".join(str(s).lower().strip().split())


def build_tags_text(books_df, tags, book_tags):
    b = books_df.copy()

    # ALWAYS ensure the column exists
    if "tags_text" not in b.columns:
        b["tags_text"] = ""
    b["tags_text"] = b["tags_text"].fillna("")

    # If missing optional files -> keep empty tags_text
    if tags is None or book_tags is None:
        return b

    bt = book_tags.copy()
    tg = tags.copy()

    # Normalize possible column name variants
    if "book_id" not in bt.columns and "goodreads_book_id" in bt.columns:
        bt = bt.rename(columns={"goodreads_book_id": "book_id"})
    if "tag_name" not in tg.columns and "tag" in tg.columns:
        tg = tg.rename(columns={"tag": "tag_name"})

    # Schema guard
    if not {"book_id", "tag_id"}.issubset(bt.columns) or not {"tag_id", "tag_name"}.issubset(tg.columns):
        return b

    merged = bt.merge(tg[["tag_id", "tag_name"]], on="tag_id", how="left")
    merged["tag_name"] = merged["tag_name"].fillna("").map(normalize_text)

    # Top 10 tags / book (ưu tiên theo count nếu có)
    if "count" in merged.columns:
        merged = merged.sort_values(["book_id", "count"], ascending=[True, False])

    merged = (
        merged.groupby("book_id")["tag_name"]
        .apply(lambda x: " ".join(list(x.head(10))))
        .reset_index()
        .rename(columns={"tag_name": "tags_text_new"})
    )

    b = b.merge(merged, on="book_id", how="left")
    b["tags_text_new"] = b["tags_text_new"].fillna("")
    b["tags_text"] = (b["tags_text"].astype(str) + " " + b["tags_text_new"].astype(str)).str.strip()
    b = b.drop(columns=["tags_text_new"])

    return b



def main():
    print("📥 Loading raw data...")
    books = pd.read_csv(BOOKS_CSV)
    ratings = pd.read_csv(RATINGS_CSV)
    # --- FIX: đồng bộ kiểu book_id để merge khớp ---
    books["book_id"] = pd.to_numeric(books["book_id"], errors="coerce")
    ratings["book_id"] = pd.to_numeric(ratings["book_id"], errors="coerce")

    books = books.dropna(subset=["book_id"]).copy()
    ratings = ratings.dropna(subset=["book_id"]).copy()

    books["book_id"] = books["book_id"].astype(int)
    ratings["book_id"] = ratings["book_id"].astype(int)


    tags = pd.read_csv(TAGS_CSV) if os.path.exists(TAGS_CSV) else None
    book_tags = pd.read_csv(BOOK_TAGS_CSV) if os.path.exists(BOOK_TAGS_CSV) else None

    # ===== CLEAN BOOKS =====
    books = books[[
        "book_id", "title", "authors",
        "original_publication_year",
        "language_code", "average_rating", "ratings_count"
    ]]

    books["title"] = books["title"].fillna("")
    books["authors"] = books["authors"].fillna("")
    books["language_code"] = books["language_code"].fillna("unknown")

    books["original_publication_year"] = (
        pd.to_numeric(books["original_publication_year"], errors="coerce")
        .fillna(2000)
        .astype(int)
        .clip(1400, 2026)
    )

    books["title_norm"] = books["title"].map(normalize_text)
    books["authors_norm"] = books["authors"].map(normalize_text)

    books = build_tags_text(books, tags, book_tags)

    books = books.drop_duplicates(
        subset=["title_norm", "authors_norm"]
    ).reset_index(drop=True)

    # ===== CLEAN RATINGS =====
    ratings = ratings[["user_id", "book_id", "rating"]]
    ratings["rating"] = (
        pd.to_numeric(ratings["rating"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    ratings = ratings[(ratings["rating"] >= 1) & (ratings["rating"] <= 5)]
    ratings = ratings.drop_duplicates(["user_id", "book_id"])

    # ===== FILTER ≥ 2000 ITEMS =====
    pop = ratings.groupby("book_id").size().rename("n_ratings")
    books = books.merge(pop, on="book_id", how="left")
    books["n_ratings"] = books["n_ratings"].fillna(0).astype(int)
    
    print("Books with n_ratings>0:", (books["n_ratings"] > 0).sum())
    print("Max n_ratings:", books["n_ratings"].max())

    books = books.sort_values("n_ratings", ascending=False).head(2500)

    keep_ids = set(books["book_id"])
    ratings = ratings[ratings["book_id"].isin(keep_ids)]

    # ===== ITEM TEXT =====
    books["item_text"] = (
        books["title_norm"]
        + " " + books["authors_norm"]
        + " " + books["tags_text"].map(normalize_text)
    )

    print(f"✅ Final books: {len(books)}")
    print(f"✅ Final ratings: {len(ratings)}")

    books.to_csv(OUT_BOOKS, index=False)
    ratings.to_csv(OUT_RATINGS, index=False)
    print("💾 Saved books_clean.csv & ratings_clean.csv")


if __name__ == "__main__":
    main()
