# music-recommendation-systemm
A machine learning-based Music Recommendation System that suggests songs to users based on their preferences and similarity of audio features.
Features:-
Recommends songs based on content similarity (audio features like danceability, energy, tempo, etc.).
Uses cosine similarity / nearest neighbors for recommendation.
Simple and interactive Streamlit web interface.
Trained on Spotify dataset (Kaggle).
Dataset:-

The dataset is taken from Spotify’s audio features dataset available on Kaggle.
It contains details like:

Track Name
Artists
Album
Popularity
Audio Features (danceability, energy, loudness, etc.)
🛠️ Tech Stack
Python 3.9+
Pandas, NumPy – Data handling
Scikit-learn – Machine Learning (similarity-based recommendation)
Streamlit – Web app interface
Pickle – Model storage
Installation:-

Clone the repository:

git clone https://github.com/your-username/music-recommendation-system.git
cd music-recommendation-system


Install dependencies:-

pip install -r requirements.txt


Run the app:

streamlit run app.py
 Usage:-
Enter the name of a song in the search box.
The system recommends similar songs you might like.
Example: Enter Shape of You → Get recommendations of similar tracks.

Future Scope:-
Add collaborative filtering (recommend based on user behavior).
Deploy on Heroku/Render/Streamlit Cloud.
Integrate with Spotify API for real-time recommendations.
