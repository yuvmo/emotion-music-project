import ast
import logging
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

ARTIST_ALIASES = {
    "monetochka": ["монеточка", "мониточка", "монетка", "monetka", "monetochka"],
    "og buda": ["og buda", "og booda", "огбуда", "оджибуда", "о.г.буда", "ogbuda"],
    "scriptonit": ["скриптонит", "скрипт", "scriptonite", "скриптонайт"],
    "oxxxymiron": ["оксимирон", "окси", "oxxxy", "oxymiron", "оксимир"],
    "face": ["фейс", "фэйс"],
    "morgenshtern": ["моргенштерн", "морген", "morgenstern"],
    "kizaru": ["кизару", "kizary"],
    "pharaoh": ["фараон", "фараох", "faraon"],
    "lsp": ["лсп", "l.s.p"],
    "miyagi": ["мияги", "miyagi"],
    "bumble beezy": ["бамбл бизи", "bumble", "бамблби"],
    "big baby tape": ["биг бейби тейп", "bbt", "bigbabytape"],
    "bushido zho": ["бушидо жо", "bushido"],
    "mayot": ["майот", "mayot"],
    "seemee": ["сими", "simi", "seemee"],
    "104": ["104", "сто четыре"],
    "obladaet": ["обладает", "obladaet"],
    "markul": ["маркул", "markul"],
    "feduk": ["федук", "feduk"],
    "gone.fludd": ["гон флад", "гонфлад", "gone fludd", "gonefludd"],
    "thomas mraz": ["томас мраз", "mraz"],
    "cream soda": ["крем сода", "creamsoda"],
    "три дня дождя": ["три дня дождя", "3 дня дождя"],
    "макс корж": ["макс корж", "корж", "maks korzh"],
    "хлеб": ["хлеб", "hleb"],
    "кис-кис": ["кис-кис", "кискис", "kis kis"],
    "пошлая молли": ["пошлая молли", "molly"],
}


def normalize_artist_name(name: str) -> str:
    """Нормализует имя артиста для нечёткого поиска."""
    if not name:
        return ""
    name = name.lower().strip()
    name = re.sub(r'[\.\-\s\'\"\,]+', '', name)
    
    translit = {
        'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ё': 'e',
        'ж': 'zh', 'з': 'z', 'и': 'i', 'й': 'y', 'к': 'k', 'л': 'l', 'м': 'm',
        'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u',
        'ф': 'f', 'х': 'h', 'ц': 'ts', 'ч': 'ch', 'ш': 'sh', 'щ': 'sch',
        'ъ': '', 'ы': 'y', 'ь': '', 'э': 'e', 'ю': 'yu', 'я': 'ya',
    }
    result = []
    for char in name:
        result.append(translit.get(char, char))
    name = ''.join(result)
    
    replacements = {
        'oo': 'u', 
        'uu': 'u',
        'ii': 'i',
        'ee': 'e',
        'aa': 'a',
        'dj': 'j',
        'dzh': 'j',
    }
    for old, new in replacements.items():
        name = name.replace(old, new)
    return name


def levenshtein_distance(s1: str, s2: str) -> int:
    """Вычисляет расстояние Левенштейна между двумя строками."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def fuzzy_artist_match(query: str, candidate: str, threshold: float = 0.7) -> bool:
    """Проверяет нечёткое совпадение имён артистов."""
    q_norm = normalize_artist_name(query)
    c_norm = normalize_artist_name(candidate)
    
    if not q_norm or not c_norm:
        return False
    
    if q_norm == c_norm:
        return True
    
    if q_norm in c_norm or c_norm in q_norm:
        return True
    
    max_len = max(len(q_norm), len(c_norm))
    if max_len == 0:
        return False
    
    distance = levenshtein_distance(q_norm, c_norm)
    similarity = 1 - (distance / max_len)
    
    if similarity >= threshold:
        return True
    
    return False


def resolve_artist_alias(query: str) -> str:
    """Проверяет алиасы и возвращает каноническое имя артиста."""
    query_lower = query.lower().strip()
    query_norm = normalize_artist_name(query_lower)
    
    for canonical, aliases in ARTIST_ALIASES.items():
        for alias in aliases:
            alias_norm = normalize_artist_name(alias)
            if query_norm == alias_norm or query_lower == alias.lower():
                return canonical
            if levenshtein_distance(query_norm, alias_norm) <= 2:
                return canonical
    
    return query


DATA_PATH = Path(__file__).parent.parent.parent / "data" / "tracks_cleaned.csv"
FALLBACK_PATH = Path(__file__).parent.parent.parent / "data" / "tracks_with_language_FINAL.csv"


@dataclass
class Track:
    spotify_id: str
    name: str
    artist: str
    year: Optional[int] = None
    genres: List[str] = field(default_factory=list)
    language: str = "other"
    valence: float = 0.5
    energy: float = 0.5
    danceability: float = 0.5
    acousticness: float = 0.5
    tempo: float = 120.0
    distance: float = 0.0
    
    @property
    def spotify_url(self) -> str:
        if self.spotify_id and self.spotify_id != "nan":
            return f"https://open.spotify.com/track/{self.spotify_id}"
        return ""
    
    def to_dict(self) -> dict:
        return {
            "spotify_id": self.spotify_id,
            "name": self.name,
            "artist": self.artist,
            "year": self.year,
            "genres": self.genres,
            "language": self.language,
            "spotify_url": self.spotify_url,
            "valence": self.valence,
            "energy": self.energy,
            "danceability": self.danceability,
            "distance": round(self.distance, 4)
        }


class MusicRecommender:
    FEATURE_COLUMNS = ["valence", "energy", "danceability", "acousticness", "tempo"]
    
    FEATURE_WEIGHTS = {
        "valence": 1.5,
        "energy": 1.2,
        "danceability": 1.0,
        "acousticness": 0.8,
        "tempo": 0.5
    }
    
    GENRE_MAPPING = {
        "pop": ["pop", "dance pop", "russian pop", "classic russian pop"],
        "dance": ["dance", "dance pop", "edm", "russian dance pop"],
        "electronic": ["electronic", "edm", "house", "techno", "trance", "russian electronic"],
        "indie": ["indie", "indie pop", "russian indie", "indie rock"],
        "hip_hop": ["hip hop", "russian hip hop", "southern hip hop"],
        "rap": ["rap", "pop rap", "russian hip hop", "trap"],
        "trap": ["trap", "russian trap"],
        "rnb": ["r&b", "urban contemporary"],
        "rock": ["rock", "russian rock", "classic russian rock", "modern rock", "hard rock"],
        "metal": ["metal", "russian metal", "russian folk metal", "russian black metal"],
        "punk": ["punk", "russian punk", "russian post-punk"],
        "alternative": ["alternative", "russian alternative"],
        "classical": ["classical", "russian classical piano", "symphony"],
        "instrumental": ["instrumental", "piano", "ambient"],
        "ambient": ["ambient", "chill"],
        "jazz": ["jazz"],
        "folk": ["folk", "russian folk", "russian folk metal"],
        "latin": ["latin", "reggaeton", "tropical"],
        "soundtrack": ["soundtrack", "score"],
        "blues": ["blues"],
    }
    
    def __init__(self, data_path: Optional[Path] = None):
        self._df: Optional[pd.DataFrame] = None
        self._data_path = data_path or DATA_PATH
        
    def _load_data(self) -> pd.DataFrame:
        if self._df is not None:
            return self._df
        
        path = self._data_path
        if not path.exists():
            path = FALLBACK_PATH
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {DATA_PATH} or {FALLBACK_PATH}")
        
        logger.info(f"Loading dataset: {path}")
        
        self._df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        
        for col in self.FEATURE_COLUMNS:
            if col in self._df.columns:
                self._df[col] = pd.to_numeric(self._df[col], errors="coerce")
        
        self._df["genres_parsed"] = self._df["genres"].apply(self._parse_genres)
        
        if "language" in self._df.columns:
            self._df["language"] = self._df["language"].fillna("other")
        else:
            self._df["language"] = "other"
        
        logger.info(f"Loaded {len(self._df)} tracks")
        return self._df
    
    @staticmethod
    def _parse_genres(x) -> List[str]:
        if pd.isna(x):
            return []
        if isinstance(x, list):
            return x
        try:
            parsed = ast.literal_eval(str(x))
            if isinstance(parsed, list):
                return [str(g).lower() for g in parsed]
        except:
            pass
        if isinstance(x, str):
            return [x.lower()]
        return []
    
    def _expand_genres(self, genre_filters: List[str]) -> List[str]:
        expanded = set()
        for genre in genre_filters:
            genre_lower = genre.lower()
            if genre_lower in self.GENRE_MAPPING:
                expanded.update(self.GENRE_MAPPING[genre_lower])
            else:
                expanded.add(genre_lower)
        return list(expanded)
    
    def _calculate_distance(self, df: pd.DataFrame, target_features: Dict[str, float]) -> pd.Series:
        distance = pd.Series(0.0, index=df.index)
        
        for feature, weight in self.FEATURE_WEIGHTS.items():
            if feature not in df.columns or feature not in target_features:
                continue
            
            target = target_features[feature]
            
            if feature == "tempo":
                diff = (df[feature].fillna(120) - target) / 140
            else:
                diff = df[feature].fillna(0.5) - target
            
            distance += weight * (diff ** 2)
        
        return np.sqrt(distance)
    
    def recommend(
        self,
        features: Dict[str, float],
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5
    ) -> List[Track]:
        df = self._load_data().copy()
        filters = filters or {}
        
        artist = filters.get("artist")
        artist_found = False
        artist_df = None
        
        if artist:
            artist = resolve_artist_alias(artist)
            logger.info(f"Resolved artist: {artist}")
            artist_lower = artist.lower().strip()
            artist_col = "artist_clean" if "artist_clean" in df.columns else "artist"
            norm_col = "artist_clean_norm" if "artist_clean_norm" in df.columns else artist_col
            
            def match_artist_exact(raw):
                if pd.isna(raw):
                    return False
                raw_str = str(raw).lower()
                if raw_str.startswith("["):
                    try:
                        parsed = ast.literal_eval(raw_str)
                        if isinstance(parsed, list) and len(parsed) == 1:
                            return str(parsed[0]).lower() == artist_lower
                    except:
                        pass
                    return raw_str.strip("[]'\"") == artist_lower
                return raw_str == artist_lower
            
            def match_artist_partial(raw):
                if pd.isna(raw):
                    return False
                raw_str = str(raw).lower()
                if raw_str.startswith("["):
                    try:
                        parsed = ast.literal_eval(raw_str)
                        if isinstance(parsed, list):
                            for a in parsed:
                                if artist_lower in str(a).lower() or str(a).lower() == artist_lower:
                                    return True
                    except:
                        pass
                return artist_lower in raw_str
            
            def match_artist_fuzzy(raw):
                if pd.isna(raw):
                    return False
                raw_str = str(raw).lower()
                if raw_str.startswith("["):
                    try:
                        parsed = ast.literal_eval(raw_str)
                        if isinstance(parsed, list):
                            for a in parsed:
                                if fuzzy_artist_match(artist_lower, str(a).lower()):
                                    return True
                    except:
                        pass
                return fuzzy_artist_match(artist_lower, raw_str)
            
            exact_match = df[df[artist_col].apply(match_artist_exact)]
            if len(exact_match) > 0:
                artist_df = exact_match
                artist_found = True
                logger.info(f"Found {len(exact_match)} tracks by artist '{artist}' (exact)")
            else:
                partial_match = df[df[artist_col].apply(match_artist_partial)]
                if len(partial_match) > 0:
                    artist_df = partial_match
                    artist_found = True
                    logger.info(f"Found {len(partial_match)} tracks by artist '{artist}' (partial)")
                else:
                    fuzzy_match = df[df[norm_col].apply(match_artist_fuzzy)]
                    if len(fuzzy_match) > 0:
                        artist_df = fuzzy_match
                        artist_found = True
                        logger.info(f"Found {len(fuzzy_match)} tracks by artist '{artist}' (fuzzy)")
                    else:
                        logger.warning(f"No tracks found for artist '{artist}', searching in all tracks")
            
            if artist_found and len(artist_df) >= top_k:
                logger.info(f"Artist mode: skipping other filters, using only artist tracks")
                df = artist_df
            elif artist_found:
                df = artist_df
        
        if not artist_found:
            language = filters.get("language")
            if language and language != "other" and language != "any":
                filtered = df[df["language"] == language]
                if len(filtered) >= top_k:
                    df = filtered
            
            genres = filters.get("genres")
            if genres:
                expanded_genres = self._expand_genres(genres)
                
                def has_genre(track_genres):
                    if not track_genres:
                        return False
                    track_set = set(g.lower() for g in track_genres)
                    return bool(track_set & set(expanded_genres))
                
                filtered = df[df["genres_parsed"].apply(has_genre)]
                if len(filtered) >= top_k:
                    df = filtered
            
            year_start = filters.get("year_start")
            year_end = filters.get("year_end")
            
            if year_start and "year" in df.columns:
                filtered = df[df["year"] >= year_start]
                if len(filtered) >= top_k:
                    df = filtered
            if year_end and "year" in df.columns:
                filtered = df[df["year"] <= year_end]
                if len(filtered) >= top_k:
                    df = filtered
        
        if len(df) < top_k and not artist_found:
            logger.warning(f"Too few tracks ({len(df)}) after filtering, resetting filters")
            df = self._load_data().copy()
            
            language = filters.get("language")
            if language and language != "other":
                filtered = df[df["language"] == language]
                if len(filtered) >= top_k:
                    df = filtered
        
        df["distance"] = self._calculate_distance(df, features)
        
        df["name_lower"] = df["name"].str.lower().str.strip()
        df = df.drop_duplicates(subset=["name_lower"], keep="first")
        
        df = df.nsmallest(top_k * 2, "distance")
        
        tracks = []
        seen_names = set()
        
        for _, row in df.iterrows():
            name = str(row.get("name", "Unknown"))
            name_key = name.lower().strip()
            
            if name_key in seen_names:
                continue
            seen_names.add(name_key)
            
            track_artist = ""
            for col in ["artist_clean_norm", "artist_clean", "artist", "artists", "artist_list"]:
                if col in row.index and pd.notna(row[col]):
                    raw = str(row[col])
                    if raw.startswith("[") and raw.endswith("]"):
                        try:
                            parsed = ast.literal_eval(raw)
                            if isinstance(parsed, list) and parsed:
                                track_artist = parsed[0] if len(parsed) == 1 else ", ".join(parsed)
                        except:
                            track_artist = raw.strip("[]'\"")
                    else:
                        track_artist = raw
                    if track_artist and track_artist.lower() not in ["nan", "none", "unknown", "[]", ""]:
                        break
            if not track_artist or track_artist.lower() in ["nan", "none", "unknown", "[]", ""]:
                track_artist = "Unknown Artist"
            
            track = Track(
                spotify_id=str(row.get("spotify_id", "")),
                name=name,
                artist=track_artist,
                year=int(row["year"]) if pd.notna(row.get("year")) else None,
                genres=row.get("genres_parsed", []),
                language=str(row.get("language", "other")),
                valence=float(row.get("valence", 0.5)),
                energy=float(row.get("energy", 0.5)),
                danceability=float(row.get("danceability", 0.5)),
                acousticness=float(row.get("acousticness", 0.5)),
                tempo=float(row.get("tempo", 120)),
                distance=float(row.get("distance", 0))
            )
            tracks.append(track)
            
            if len(tracks) >= top_k:
                break
        
        return tracks


_recommender_instance: Optional[MusicRecommender] = None


def get_music_recommender() -> MusicRecommender:
    global _recommender_instance
    if _recommender_instance is None:
        _recommender_instance = MusicRecommender()
    return _recommender_instance
