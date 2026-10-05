import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Movie Recommendation System")
st.write("Select a movie and get Top 10 similar movie recommendations.")

# Your dataset file
DATA_FILE = "imdb_top_1000.csv"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)
    df = df.dropna(how="all")
    return df


try:
    movies = load_data()

except FileNotFoundError:
    st.error(
        f"❌ '{DATA_FILE}' was not found.\n\n"
        "Keep imdb_top_1000.csv in the same folder as app.py."
    )
    st.stop()


# Detect movie title column
possible_title_columns = [
    "Series_Title",
    "title",
    "Title",
    "movie_title",
    "movie",
    "Movie",
    "name",
    "Name"
]

title_column = None

for column in possible_title_columns:
    if column in movies.columns:
        title_column = column
        break

if title_column is None:
    st.error(
        "❌ Movie title column was not found.\n\n"
        f"Available columns: {list(movies.columns)}"
    )
    st.stop()


# Detect genre column
possible_genre_columns = [
    "Genre",
    "genres",
    "Genres",
    "genre"
]

genre_column = None

for column in possible_genre_columns:
    if column in movies.columns:
        genre_column = column
        break

if genre_column is None:
    st.error(
        "❌ Genre column was not found.\n\n"
        f"Available columns: {list(movies.columns)}"
    )
    st.stop()


# Clean data
movies[title_column] = movies[title_column].fillna("").astype(str)
movies[genre_column] = movies[genre_column].fillna("").astype(str)

movies = movies[
    movies[title_column].str.strip() != ""
].reset_index(drop=True)


@st.cache_resource
def build_similarity_model(genres):

    vectorizer = CountVectorizer(
        stop_words="english"
    )

    genre_matrix = vectorizer.fit_transform(genres)

    similarity_matrix = cosine_similarity(
        genre_matrix
    )

    return similarity_matrix


similarity = build_similarity_model(
    movies[genre_column]
)


def recommend_movies(movie_name):

    matching = movies[
        movies[title_column] == movie_name
    ]

    if matching.empty:
        return []

    movie_index = matching.index[0]

    similarity_scores = list(
        enumerate(similarity[movie_index])
    )

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    # Skip selected movie
    similarity_scores = similarity_scores[1:11]

    recommendations = []

    for index, score in similarity_scores:

        recommendations.append({
            "title": movies.iloc[index][title_column],
            "genre": movies.iloc[index][genre_column],
            "similarity": round(score * 100, 2)
        })

    return recommendations


st.divider()

st.subheader("🔍 Select a Movie")

movie_list = sorted(
    movies[title_column]
    .drop_duplicates()
    .tolist()
)

selected_movie = st.selectbox(
    "Choose your favourite movie",
    movie_list
)


if st.button(
    "Recommend Movies",
    use_container_width=True
):

    recommendations = recommend_movies(
        selected_movie
    )

    st.divider()

    st.subheader(
        f"🍿 Top 10 Recommendations for {selected_movie}"
    )

    if not recommendations:

        st.warning(
            "No recommendations found."
        )

    else:

        for number, movie in enumerate(
            recommendations,
            start=1
        ):

            with st.container(border=True):

                col1, col2 = st.columns(
                    [1, 6]
                )

                with col1:
                    st.markdown(
                        f"## {number}"
                    )

                with col2:

                    st.markdown(
                        f"### 🎥 {movie['title']}"
                    )

                    st.write(
                        f"**Genre:** {movie['genre']}"
                    )

                    st.write(
                        f"**Similarity Score:** "
                        f"{movie['similarity']}%"
                    )


st.divider()

st.caption(
    "Content-Based Movie Recommendation System "
    "using CountVectorizer and Cosine Similarity."
)