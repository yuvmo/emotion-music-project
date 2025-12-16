import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
INPUT_CSV = BASE_DIR / "data" / "tracks_with_language_FINAL.csv"
OUTPUT_CSV = BASE_DIR / "data" / "tracks_cleaned.csv"
DOTENV_PATH = BASE_DIR / ".env"

load_dotenv(DOTENV_PATH)


def load_dataset() -> pd.DataFrame:
    logger.info(f"Loading dataset from {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    logger.info(f"Loaded {len(df)} tracks")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=["spotify_id"], keep="first")
    after = len(df)
    logger.info(f"Removed {before - after} duplicates, {after} tracks remaining")
    return df


def remove_empty_names(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=["name"])
    df = df[df["name"].str.strip() != ""]
    after = len(df)
    logger.info(f"Removed {before - after} tracks without names")
    return df


def extract_year_from_release_date(df: pd.DataFrame) -> pd.DataFrame:
    def parse_year(date_str):
        if pd.isna(date_str):
            return None
        date_str = str(date_str)
        if len(date_str) >= 4:
            try:
                return int(date_str[:4])
            except ValueError:
                return None
        return None
    
    df["year"] = df["release_date"].apply(parse_year)
    missing = df["year"].isna().sum()
    logger.info(f"Extracted year from release_date, {missing} missing values")
    return df


def clean_artist_names(df: pd.DataFrame) -> pd.DataFrame:
    import ast
    
    def clean_artist(x):
        if pd.isna(x):
            return "Unknown"
        x_str = str(x)
        if x_str.startswith("[") and x_str.endswith("]"):
            try:
                artists = ast.literal_eval(x_str)
                if isinstance(artists, list) and len(artists) > 0:
                    return ", ".join([str(a).strip("'\"") for a in artists])
            except:
                pass
        return x_str.strip("[]'\"")
    
    if "artist_clean" in df.columns:
        df["artist"] = df["artist_clean"].apply(clean_artist)
    elif "artists" in df.columns:
        df["artist"] = df["artists"].apply(clean_artist)
    else:
        df["artist"] = "Unknown"
    
    logger.info("Cleaned artist names")
    return df


def improve_language_detection(df: pd.DataFrame) -> pd.DataFrame:
    import re
    
    cyrillic_pattern = re.compile(r'[а-яА-ЯёЁ]')
    latin_pattern = re.compile(r'[a-zA-Z]')
    
    russian_artist_keywords = [
        "серёга", "серега", "кино", "цой", "дdt", "ддт", "грибы", "баста", 
        "тимати", "нюша", "егор крид", "макс корж", "хаски", "моргенштерн",
        "oxxxymiron", "оксимирон", "земфира", "гречка", "ханза", "miyagi",
        "скриптонит", "pharaoh", "фараон", "face", "элджей", "bumble beezy",
        "noize mc", "ленинград", "шнуров", "t-fest", "jah khalib"
    ]
    
    def detect_language(row):
        name = str(row.get("name", ""))
        artist = str(row.get("artist", "")).lower()
        current_lang = row.get("language", "other")
        
        if current_lang == "instrumental":
            return "instrumental"
        
        cyrillic_in_name = len(cyrillic_pattern.findall(name))
        latin_in_name = len(latin_pattern.findall(name))
        
        cyrillic_in_artist = len(cyrillic_pattern.findall(artist))
        
        is_russian_artist = any(kw in artist for kw in russian_artist_keywords)
        
        if cyrillic_in_name > 2:
            return "ru"
        
        if is_russian_artist or cyrillic_in_artist > 2:
            return "ru"
        
        if current_lang == "ru" and cyrillic_in_name == 0 and cyrillic_in_artist == 0:
            if not is_russian_artist:
                return "en"
        
        if latin_in_name > 2 and cyrillic_in_name == 0:
            if current_lang == "other":
                return "en"
        
        if current_lang and current_lang != "other":
            return current_lang
        
        return "other"
    
    df["language"] = df.apply(detect_language, axis=1)
    
    lang_counts = df["language"].value_counts()
    logger.info(f"Language distribution after improvement:\n{lang_counts}")
    return df


def enrich_with_spotify(df: pd.DataFrame, batch_size: int = 50, max_tracks: int = 5000) -> pd.DataFrame:
    try:
        import spotipy
        from spotipy.oauth2 import SpotifyClientCredentials
    except ImportError:
        logger.warning("spotipy not installed, skipping Spotify enrichment")
        return df
    
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    
    if not client_id or not client_secret:
        logger.warning("Spotify credentials not found, skipping enrichment")
        return df
    
    try:
        auth = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
        sp = spotipy.Spotify(auth_manager=auth)
    except Exception as e:
        logger.error(f"Failed to connect to Spotify: {e}")
        return df
    
    missing_date = df[df["release_date"].isna() | (df["release_date"] == "")]
    logger.info(f"Found {len(missing_date)} tracks with missing release_date")
    
    if len(missing_date) == 0:
        logger.info("No tracks need enrichment for dates")
    else:
        tracks_to_process = missing_date.head(max_tracks)
        track_ids = tracks_to_process["spotify_id"].tolist()
        enriched_count = 0
        
        for i in range(0, len(track_ids), batch_size):
            batch = track_ids[i:i+batch_size]
            logger.info(f"Enriching dates: batch {i//batch_size + 1}/{(len(track_ids)-1)//batch_size + 1}")
            
            try:
                tracks_data = sp.tracks(batch)
                
                for track in tracks_data["tracks"]:
                    if track is None:
                        continue
                    
                    track_id = track["id"]
                    release_date = track["album"].get("release_date", "")
                    
                    if release_date:
                        df.loc[df["spotify_id"] == track_id, "release_date"] = release_date
                        enriched_count += 1
                
                time.sleep(0.3)
                
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                time.sleep(2)
        
        logger.info(f"Enriched {enriched_count} tracks with release dates")
    
    return df


def enrich_genres_from_artists(df: pd.DataFrame, batch_size: int = 50, max_artists: int = 1000) -> pd.DataFrame:
    import ast
    
    try:
        import spotipy
        from spotipy.oauth2 import SpotifyClientCredentials
    except ImportError:
        logger.warning("spotipy not installed, skipping genre enrichment")
        return df
    
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    
    if not client_id or not client_secret:
        logger.warning("Spotify credentials not found, skipping genre enrichment")
        return df
    
    try:
        auth = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
        sp = spotipy.Spotify(auth_manager=auth)
    except Exception as e:
        logger.error(f"Failed to connect to Spotify: {e}")
        return df
    
    def parse_genres(x):
        if pd.isna(x) or x == "[]" or x == "":
            return []
        try:
            parsed = ast.literal_eval(str(x))
            if isinstance(parsed, list):
                return parsed
        except:
            pass
        return []
    
    df["genres_parsed"] = df["genres"].apply(parse_genres)
    missing_genres = df[df["genres_parsed"].apply(len) == 0]
    
    if "artist_spotify_id" not in df.columns:
        logger.warning("No artist_spotify_id column, cannot enrich genres")
        return df
    
    artist_ids = missing_genres["artist_spotify_id"].dropna().unique()[:max_artists]
    logger.info(f"Found {len(artist_ids)} unique artists without genres")
    
    if len(artist_ids) == 0:
        return df
    
    artist_genres_map = {}
    
    for i in range(0, len(artist_ids), batch_size):
        batch = list(artist_ids[i:i+batch_size])
        logger.info(f"Fetching artist genres: batch {i//batch_size + 1}/{(len(artist_ids)-1)//batch_size + 1}")
        
        try:
            artists_data = sp.artists(batch)
            
            for artist in artists_data["artists"]:
                if artist:
                    artist_genres_map[artist["id"]] = artist.get("genres", [])
            
            time.sleep(0.3)
            
        except Exception as e:
            logger.error(f"Error fetching artists: {e}")
            time.sleep(2)
    
    enriched_count = 0
    for artist_id, genres in artist_genres_map.items():
        if genres:
            mask = (df["artist_spotify_id"] == artist_id) & (df["genres_parsed"].apply(len) == 0)
            df.loc[mask, "genres"] = str(genres)
            enriched_count += mask.sum()
    
    logger.info(f"Enriched {enriched_count} tracks with genres from artist data")
    
    df = df.drop(columns=["genres_parsed"], errors="ignore")
    
    return df


def filter_quality_tracks(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    
    required_features = ["valence", "energy", "danceability"]
    for col in required_features:
        if col in df.columns:
            df = df[df[col].notna()]
    
    after = len(df)
    logger.info(f"Removed {before - after} tracks without audio features, {after} remaining")
    return df


def save_dataset(df: pd.DataFrame):
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    
    columns_to_keep = [
        "spotify_id", "name", "artist", "year", "release_date",
        "genres", "language", "popularity", "explicit",
        "valence", "energy", "danceability", "acousticness", 
        "tempo", "instrumentalness", "liveness", "loudness",
        "speechiness", "duration_ms", "mode", "key"
    ]
    
    available_columns = [c for c in columns_to_keep if c in df.columns]
    df_out = df[available_columns].copy()
    
    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    logger.info(f"Saved cleaned dataset to {OUTPUT_CSV}")
    logger.info(f"Final dataset: {len(df_out)} tracks, {len(available_columns)} columns")


def print_statistics(df: pd.DataFrame):
    print("\n" + "=" * 50)
    print("СТАТИСТИКА ОЧИЩЕННОГО ДАТАСЕТА")
    print("=" * 50)
    print(f"Всего треков: {len(df)}")
    print(f"\nЯзыки:")
    print(df["language"].value_counts())
    print(f"\nГода (диапазон):")
    if "year" in df.columns and df["year"].notna().any():
        print(f"  Min: {df['year'].min()}, Max: {df['year'].max()}")
    print(f"\nПропущенные значения:")
    print(df.isnull().sum()[df.isnull().sum() > 0])


def main():
    df = load_dataset()
    
    df = remove_duplicates(df)
    df = remove_empty_names(df)
    df = extract_year_from_release_date(df)
    df = clean_artist_names(df)
    df = improve_language_detection(df)
    
    if "--enrich" in sys.argv:
        df = enrich_with_spotify(df)
        df = enrich_genres_from_artists(df)
    
    if "--filter-quality" in sys.argv:
        df = filter_quality_tracks(df)
    
    save_dataset(df)
    print_statistics(df)


if __name__ == "__main__":
    main()

