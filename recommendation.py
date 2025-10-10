import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
# --- Page Configuration ---
st.set_page_config(
    page_title="MeloMix Song Recommender",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Load Data and Models ---
# Load clustered dataset
song_data = pd.read_csv("clustered_df.csv")  # Make sure your CSV has 'track_name', 'artists', 'Cluster', numerical features, optional 'image_url'

# Load scaler and clustering model
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("kmeans_model.pkl", "rb") as f:
    kmeans = pickle.load(f)

# --- Numerical features for similarity ---
numerical_features = ["valence", "danceability", "energy", "tempo", 
    "acousticness", "liveness", "speechiness", "instrumentalness"]  # Replace with your actual numeric feature columns

# --- Recommendation Function ---
def get_recommendations(song_title, num_recs=5):
    if song_title not in song_data['track_name'].values:
        return None  # song not found

    # Scale numerical features
    features = song_data[numerical_features]
    scaled_features = scaler.transform(features)

    # If clusters not in CSV, predict them
    # song_data['Cluster'] = kmeans.predict(scaled_features)

    # Get cluster of input song
    song_cluster = song_data[song_data['track_name'] == song_title]['Cluster'].values[0]

    # Filter songs in same cluster
    cluster_songs = song_data[song_data['Cluster'] == song_cluster].reset_index(drop=True)

    # Compute cosine similarity within cluster
    cluster_features = cluster_songs[numerical_features]
    similarity = cosine_similarity(cluster_features, cluster_features)

    # Index of input song
    song_index = cluster_songs[cluster_songs['track_name'] == song_title].index[0]

    # Get top similar songs
    similar_idx = np.argsort(similarity[song_index])[-(num_recs+1):-1][::-1]
    recommendations = cluster_songs.iloc[similar_idx]

    return recommendations

# --- Streamlit UI ---
st.title("🎵 MeloMix Song Recommender")
st.markdown(
    "Discover your next favorite song! Choose a song you love, and we'll find similar tracks for you."
)

# Sidebar
with st.sidebar:
    st.header("About")
    st.write(
        "This app uses a content-based filtering model to recommend songs "
        "based on your input. The model analyzes song features to find tracks "
        "with a similar vibe."
    )
    st.header("How to Use")
    st.markdown(
        "1. **Select a song** from the dropdown menu.\n"
        "2. **Click 'Recommend'** to see your personalized suggestions.\n"
        "3. **Enjoy** the music!"
    )
    st.markdown("---")
    st.info("Powered by your trained model and clustered dataset.")

# User Input
song_list = song_data['track_name'].unique()
selected_song = st.selectbox(
    "Choose a song you like:",
    options=song_list,
    index=0
)

# Recommendation Button
if st.button("Recommend Me Songs!", key='recommend_button'):
    with st.spinner("Finding recommendations for you..."):
        recommendations = get_recommendations(selected_song, num_recs=5)

        if recommendations is not None and not recommendations.empty:
            st.success(f"Here are some recommendations based on '{selected_song}':")

            # Display recommendations in columns
            cols = st.columns(len(recommendations))
            for i, (index, row) in enumerate(recommendations.iterrows()):
                with cols[i]:
                    if 'image_url' in row and pd.notnull(row['image_url']):
                        st.image(row['image_url'], caption=f"{row['track_name']} by {row['artists']}", use_column_width=True)
                    else:
                        st.write(f"{row['track_name']} by {row['artists']}")

        else:
            st.error("Sorry, we couldn't find recommendations for that song. Please try another.")

# Footer
st.markdown("---")
st.write("Created with ❤️ using Streamlit")


