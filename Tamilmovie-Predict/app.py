import streamlit as st
import pandas as pd

# 🎨 Dark Theme CSS
st.markdown("""
    <style>
    body {
        background-color: #0f1117;
        color: white;
    }
    .stDataFrame {
        background-color: #1c1c1c;
    }
    h1, h2 {
        color: #ff4b4b;
    }
    </style>
""", unsafe_allow_html=True)

# 🎬 App Title
st.title("🎬 Tamil Movie Genre Finder")

# Load the dataset from local CSV
try:
    df = pd.read_csv("tamil_movie_dataset.csv", header=None, skiprows=1,
                     names=["Title", "Year", "Genre", "Director", "Lead_Actor", "IMDB_Rating"])

    # 🔍 Input Field to Type Genre
    genre_input = st.text_input("Enter a genre (e.g., Action, Comedy, Drama):")

    if genre_input:
        # Filter movies matching the input genre (case-insensitive)
        filtered_df = df[df["Genre"].str.contains(genre_input, case=False, na=False)]

        if not filtered_df.empty:
            st.success(f"Showing {len(filtered_df)} '{genre_input.title()}' Tamil movies:")
            st.dataframe(filtered_df, use_container_width=True)
        else:
            st.warning(f"No Tamil movies found with genre: {genre_input}")

    else:
        st.info("Please type a genre to search.")

except FileNotFoundError:
    st.error("❌ File 'tamil_movie_dataset.csv' not found. Please make sure it's in the same folder.")
