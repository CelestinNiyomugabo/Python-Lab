import pandas as pd
import random
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from kneed import KneeLocator

# -------------------------
# 🎵 MUSIC DATASET
# -------------------------
music_data = [
    {"Title": "Katerina", "Artist": "Bruce Melodie", "Genre": "Afropop", "Album": "Ikinya", "Year": 2019},
    {"Title": "Saa Moya", "Artist": "Bruce Melodie", "Genre": "Afropop", "Album": "Ikinya", "Year": 2021},
    {"Title": "Katapilla", "Artist": "Bruce Melodie", "Genre": "Afropop", "Album": "Mixed", "Year": 2020},
    {"Title": "Slowly", "Artist": "Meddy", "Genre": "RnB", "Album": "Meddy Classics", "Year": 2017},
    {"Title": "My Vow", "Artist": "Meddy", "Genre": "RnB", "Album": "Meddy Classics", "Year": 2021},
    {"Title": "Ntawamusimbura", "Artist": "Meddy", "Genre": "RnB", "Album": "Meddy Classics", "Year": 2016},
    {"Title": "Habibi", "Artist": "The Ben", "Genre": "Afropop", "Album": "Kigali Love", "Year": 2022},
    {"Title": "Ndaje", "Artist": "The Ben", "Genre": "Afropop", "Album": "Kigali Love", "Year": 2017},
    {"Title": "Bad", "Artist": "Ariel Wayz", "Genre": "RnB", "Album": "Self Love", "Year": 2022},
    {"Title": "10 Days", "Artist": "Ariel Wayz", "Genre": "RnB", "Album": "Self Love", "Year": 2023},
    {"Title": "Ready", "Artist": "Bwiza", "Genre": "Afropop", "Album": "Bwiza Season", "Year": 2022},
    {"Title": "Ubudodo", "Artist": "Bwiza", "Genre": "Afropop", "Album": "Bwiza Season", "Year": 2023},
    {"Title": "Anytime", "Artist": "Mike Kayihura", "Genre": "RnB", "Album": "Zuba", "Year": 2021},
    {"Title": "Sabrina", "Artist": "Mike Kayihura", "Genre": "RnB", "Album": "Zuba", "Year": 2022},
    {"Title": "Madiba", "Artist": "Kivumbi King", "Genre": "Hip-hop", "Album": "Igikwe", "Year": 2021},
    {"Title": "Pasta", "Artist": "Kivumbi King", "Genre": "Hip-hop", "Album": "Igikwe", "Year": 2023},
    {"Title": "Amata", "Artist": "Social Mula", "Genre": "Traditional", "Album": "Mula Mix", "Year": 2018},
    {"Title": "Superstar", "Artist": "Social Mula", "Genre": "RnB", "Album": "Mula Mix", "Year": 2019}
]

music_df = pd.DataFrame(music_data)
music_df['TrackID'] = ['T{:04d}'.format(i+1) for i in range(len(music_df))]

# -------------------------
# 👥 SIMULATE LISTENING DATA
# -------------------------
user_ids = [f"U{str(i).zfill(4)}" for i in range(1, 1001)]
track_ids = music_df['TrackID'].tolist()

popularity_weights = np.linspace(1, 3, len(track_ids))**2
track_popularity = dict(zip(track_ids, popularity_weights))

user_profiles = {}
for user in user_ids:
    fav_genre = random.choice(music_df['Genre'].unique())
    fav_artist = random.choice(music_df['Artist'].unique())
    activity = np.random.choice(['low', 'medium', 'high'], p=[0.3, 0.5, 0.2])
    num_listens = {
        'low': random.randint(5, 15),
        'medium': random.randint(20, 40),
        'high': random.randint(50, 100)
    }[activity]
    user_profiles[user] = {
        'favorite_genre': fav_genre,
        'favorite_artist': fav_artist,
        'num_listens': num_listens
    }

listens_data = []
for user, profile in user_profiles.items():
    preferred_tracks = music_df[
        (music_df['Genre'] == profile['favorite_genre']) | 
        (music_df['Artist'] == profile['favorite_artist'])
    ]['TrackID'].tolist()
    other_tracks = list(set(track_ids) - set(preferred_tracks))
    
    for _ in range(profile['num_listens']):
        if random.random() < 0.7 and preferred_tracks:
            track = random.choice(preferred_tracks)
        else:
            weights = np.array([track_popularity[t] for t in other_tracks])
            probs = weights / weights.sum()
            track = np.random.choice(other_tracks, p=probs)

        listens_data.append({
            'UserID': user,
            'TrackID': track,
            'Rating': random.choices([3, 4, 5], weights=[1, 2, 3])[0]
        })

listens_df = pd.DataFrame(listens_data)
print("\n🎧 Sample of Listening Dataset:")
print(listens_df.head())

# -------------------------
# 🔍 AUTO ELBOW DETECTOR
# -------------------------
def find_optimal_k(X, title="Elbow Method", max_k=10):
    distortions = []
    K = list(range(1, max_k + 1))
    for k in K:
        model = KMeans(n_clusters=k, random_state=42)
        model.fit(X)
        distortions.append(model.inertia_)
    kl = KneeLocator(K, distortions, curve='convex', direction='decreasing')
    best_k = kl.elbow
    plt.plot(K, distortions, 'bo-')
    plt.vlines(best_k, ymin=min(distortions), ymax=max(distortions), linestyles='--', colors='red')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title(f"{title} (Elbow at k={best_k})")
    plt.grid(True)
    plt.show()
    return best_k

# -------------------------
# 🎼 SONG CLUSTERING
# -------------------------
song_features = music_df.copy()
song_features['GenreCode'] = LabelEncoder().fit_transform(song_features['Genre'])
song_features['ArtistCode'] = LabelEncoder().fit_transform(song_features['Artist'])

X_song = song_features[['GenreCode', 'ArtistCode', 'Year']]
X_song_scaled = StandardScaler().fit_transform(X_song)

optimal_k_song = find_optimal_k(X_song_scaled, "Song Clustering")
kmeans_song = KMeans(n_clusters=optimal_k_song, random_state=42)
song_features['SongCluster'] = kmeans_song.fit_predict(X_song_scaled)

print("\n📊 Clustered Songs:")
print(song_features[['Title', 'Artist', 'Genre', 'Year', 'SongCluster']].sort_values('SongCluster'))

# -------------------------
# 👥 USER CLUSTERING
# -------------------------
listens_merged = listens_df.merge(music_df, on='TrackID')
user_genre_matrix = listens_merged.pivot_table(
    index='UserID',
    columns='Genre',
    values='Rating',
    aggfunc='count',
    fill_value=0
)

X_user = StandardScaler().fit_transform(user_genre_matrix)

optimal_k_user = find_optimal_k(X_user, "User Clustering")
kmeans_user = KMeans(n_clusters=optimal_k_user, random_state=42)
user_genre_matrix['UserCluster'] = kmeans_user.fit_predict(X_user)

print("\n👥 User Clusters Summary:")
print(user_genre_matrix.groupby('UserCluster').mean())



# Compute cosine similarity between all songs
cosine_sim_matrix = cosine_similarity(X_song_scaled)

# Map TrackID to index
trackid_to_index = dict(zip(song_features['TrackID'], song_features.index))
index_to_trackid = dict(zip(song_features.index, song_features['TrackID']))

# Recommend top N similar songs
def recommend_similar_songs(track_id, top_n=3):
    if track_id not in trackid_to_index:
        print("❌ Track ID not found.")
        return
    
    idx = trackid_to_index[track_id]
    sim_scores = list(enumerate(cosine_sim_matrix[idx]))
    
    # Sort by similarity score (skip the song itself)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_matches = sim_scores[1:top_n+1]  # Skip index 0 (itself)

    print(f"\n🎧 Selected Song: {song_features.loc[idx, 'Title']} by {song_features.loc[idx, 'Artist']}")
    print("🔥 Recommended Songs:\n")
    
    for i, score in top_matches:
        song = song_features.loc[i]
        print(f"🎵 {song['Title']} by {song['Artist']} [{song['Genre']}, {song['Year']}]  (Similarity: {score:.2f})")

recommend_similar_songs("T0015")  # Meddy - Slowly
