import logging
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from src.spotify_client import get_spotify_client
from src.llm.gigachat import get_gigachat_service

logger = logging.getLogger(__name__)

CYRILLIC_PATTERN = re.compile(r'[а-яА-ЯёЁ]')
LATIN_PATTERN = re.compile(r'[a-zA-Z]')


@dataclass
class ValidatedTrack:
    spotify_id: str
    name: str
    artist: str
    year: Optional[int]
    language: str
    genres: List[str]
    spotify_url: str
    is_verified: bool = False
    verification_source: str = "dataset"
    
    def to_dict(self) -> dict:
        return {
            "spotify_id": self.spotify_id,
            "name": self.name,
            "artist": self.artist,
            "year": self.year,
            "language": self.language,
            "genres": self.genres,
            "spotify_url": self.spotify_url,
            "is_verified": self.is_verified,
        }


class TrackValidator:
    def __init__(self):
        self.spotify = get_spotify_client()
        self._gigachat = None
    
    @property
    def gigachat(self):
        if self._gigachat is None:
            self._gigachat = get_gigachat_service()
        return self._gigachat
    
    def detect_language_from_text(self, text: str) -> str:
        if not text:
            return "other"
        
        cyrillic_count = len(CYRILLIC_PATTERN.findall(text))
        latin_count = len(LATIN_PATTERN.findall(text))
        total = cyrillic_count + latin_count
        
        if total == 0:
            return "other"
        
        if cyrillic_count / total > 0.5:
            return "ru"
        elif latin_count / total > 0.8:
            return "en"
        
        return "other"
    
    def verify_with_spotify(self, track_id: str) -> Optional[Dict[str, Any]]:
        if not self.spotify.is_available():
            return None
        
        try:
            info = self.spotify.get_track_info(track_id)
            if info:
                artist_ids = []
                search_result = self.spotify.search_tracks(
                    f"{info['artist']} {info['name']}", 
                    limit=1
                )
                if search_result:
                    artist_ids = search_result[0].get("artist_ids", [])
                
                genres = []
                if artist_ids:
                    genres_map = self.spotify.get_artists_genres_batch(artist_ids[:1])
                    for g in genres_map.values():
                        genres.extend(g)
                
                return {
                    "name": info["name"],
                    "artist": info["artist"],
                    "release_date": info.get("release_date", ""),
                    "spotify_url": info["spotify_url"],
                    "genres": genres,
                    "verified": True
                }
        except Exception as e:
            logger.warning(f"Spotify verification failed for {track_id}: {e}")
        
        return None
    
    def analyze_with_gigachat(self, tracks: List[Dict]) -> List[Dict]:
        if not tracks:
            return tracks
        
        tracks_info = []
        for i, t in enumerate(tracks[:5]):
            tracks_info.append(f"{i+1}. {t.get('artist', 'Unknown')} - {t.get('name', 'Unknown')}")
        
        prompt = f"""Проанализируй эти треки и определи для каждого:
1. Язык трека (ru/en/instrumental/other) - по названию и артисту
2. Правильно ли записан артист

Треки:
{chr(10).join(tracks_info)}

Ответь СТРОГО в формате JSON (без markdown):
{{"tracks": [
  {{"index": 1, "language": "ru", "artist_correct": true, "suggested_artist": null}},
  ...
]}}"""
        
        try:
            response = self.gigachat.chat(prompt)
            
            import json
            clean_response = response.strip()
            if clean_response.startswith("```"):
                clean_response = clean_response.split("```")[1]
                if clean_response.startswith("json"):
                    clean_response = clean_response[4:]
            
            data = json.loads(clean_response)
            
            for item in data.get("tracks", []):
                idx = item.get("index", 0) - 1
                if 0 <= idx < len(tracks):
                    if item.get("language"):
                        tracks[idx]["language_suggested"] = item["language"]
                    if item.get("suggested_artist"):
                        tracks[idx]["artist_suggested"] = item["suggested_artist"]
            
            return tracks
            
        except Exception as e:
            logger.warning(f"GigaChat analysis failed: {e}")
            return tracks
    
    def validate_tracks(
        self, 
        tracks: List[Dict], 
        verify_spotify: bool = True,
        use_gigachat: bool = False
    ) -> List[ValidatedTrack]:
        validated = []
        
        for track in tracks:
            name = track.get("name", "")
            artist = track.get("artist", "")
            spotify_id = track.get("spotify_id", "")
            year = track.get("year")
            language = track.get("language", "other")
            genres = track.get("genres", [])
            spotify_url = track.get("spotify_url", "")
            
            is_verified = False
            verification_source = "dataset"
            
            if verify_spotify and spotify_id and self.spotify.is_available():
                spotify_info = self.verify_with_spotify(spotify_id)
                if spotify_info:
                    name = spotify_info["name"]
                    artist = spotify_info["artist"]
                    spotify_url = spotify_info["spotify_url"]
                    if spotify_info["genres"]:
                        genres = spotify_info["genres"]
                    if spotify_info.get("release_date"):
                        try:
                            year = int(spotify_info["release_date"][:4])
                        except:
                            pass
                    is_verified = True
                    verification_source = "spotify"
            
            detected_lang = self.detect_language_from_text(f"{name} {artist}")
            if detected_lang != "other":
                language = detected_lang
            elif CYRILLIC_PATTERN.search(name) or CYRILLIC_PATTERN.search(artist):
                language = "ru"
            elif not CYRILLIC_PATTERN.search(name) and not CYRILLIC_PATTERN.search(artist):
                if language not in ["en", "instrumental"]:
                    language = "en"
            
            if not artist or artist.lower() in ["unknown", "nan", "none", ""]:
                artist = "Unknown Artist"
            
            if not spotify_url and spotify_id and spotify_id not in ["nan", "None", ""]:
                spotify_url = f"https://open.spotify.com/track/{spotify_id}"
            
            validated.append(ValidatedTrack(
                spotify_id=spotify_id,
                name=name,
                artist=artist,
                year=year,
                language=language,
                genres=genres if isinstance(genres, list) else [],
                spotify_url=spotify_url,
                is_verified=is_verified,
                verification_source=verification_source
            ))
        
        return validated
    
    def validate_and_enrich(
        self,
        tracks: List[Dict],
        max_spotify_calls: int = 3,
        use_gigachat: bool = False
    ) -> List[ValidatedTrack]:
        validated = []
        
        for i, track in enumerate(tracks):
            verify_spotify = i < max_spotify_calls
            
            result = self.validate_tracks(
                [track], 
                verify_spotify=verify_spotify,
                use_gigachat=False
            )
            
            if result:
                validated.append(result[0])
        
        if use_gigachat and validated:
            track_dicts = [v.to_dict() for v in validated]
            enriched = self.analyze_with_gigachat(track_dicts)
            
            for i, enriched_data in enumerate(enriched):
                if i < len(validated):
                    if enriched_data.get("language_suggested"):
                        validated[i].language = enriched_data["language_suggested"]
                    if enriched_data.get("artist_suggested"):
                        validated[i].artist = enriched_data["artist_suggested"]
        
        return validated


_validator_instance: Optional[TrackValidator] = None


def get_track_validator() -> TrackValidator:
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = TrackValidator()
    return _validator_instance

