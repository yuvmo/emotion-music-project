import os
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

try:
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
    SPOTIPY_AVAILABLE = True
except ImportError:
    SPOTIPY_AVAILABLE = False


class SpotifyClient:
    def __init__(self):
        self._client = None
        self._available = False
    
    def _get_client(self):
        if not SPOTIPY_AVAILABLE:
            return None
        
        if self._client is None:
            client_id = os.getenv("SPOTIFY_CLIENT_ID")
            client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
            
            if not client_id or not client_secret:
                logger.warning("Spotify credentials not found")
                return None
            
            try:
                auth_manager = SpotifyClientCredentials(
                    client_id=client_id,
                    client_secret=client_secret
                )
                self._client = spotipy.Spotify(auth_manager=auth_manager)
                self._available = True
            except Exception as e:
                logger.error(f"Failed to init Spotify client: {e}")
                return None
        
        return self._client
    
    def is_available(self) -> bool:
        return self._get_client() is not None
    
    def search_tracks(
        self,
        query: str,
        limit: int = 10,
        market: str = "RU"
    ) -> List[Dict[str, Any]]:
        client = self._get_client()
        if not client:
            return []
        
        try:
            results = client.search(q=query, type="track", limit=limit, market=market)
        except Exception as e:
            logger.error(f"Spotify search error: {e}")
            return []
        
        tracks = []
        for item in results.get("tracks", {}).get("items", []):
            artist_ids = [a["id"] for a in item["artists"]]
            track = {
                "spotify_id": item["id"],
                "name": item["name"],
                "artist": ", ".join([a["name"] for a in item["artists"]]),
                "artist_ids": artist_ids,
                "album": item["album"]["name"],
                "release_date": item["album"].get("release_date", ""),
                "popularity": item["popularity"],
                "preview_url": item.get("preview_url"),
                "spotify_url": item["external_urls"]["spotify"],
                "image_url": item["album"]["images"][0]["url"] if item["album"]["images"] else None
            }
            tracks.append(track)
        
        return tracks
    
    def search_by_mood(
        self,
        mood: str,
        genre: str = None,
        language: str = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        mood_keywords = {
            "happy": ["happy", "upbeat", "feel good", "party"],
            "sad": ["sad", "melancholy", "heartbreak", "emotional"],
            "angry": ["aggressive", "intense", "rage", "heavy"],
            "calm": ["calm", "relaxing", "peaceful", "ambient"],
            "energetic": ["energetic", "workout", "power", "pump up"],
            "romantic": ["love", "romantic", "ballad", "heart"],
        }
        
        keywords = mood_keywords.get(mood.lower(), [mood])
        query_parts = [keywords[0]]
        
        if genre:
            query_parts.append(f"genre:{genre}")
        
        query = " ".join(query_parts)
        
        market = "RU" if language == "ru" else "US"
        
        return self.search_tracks(query, limit=limit, market=market)
    
    def get_artist_genres(self, artist_id: str) -> List[str]:
        client = self._get_client()
        if not client:
            return []
        
        try:
            artist = client.artist(artist_id)
            return artist.get("genres", [])
        except Exception as e:
            logger.error(f"Error getting artist genres: {e}")
            return []
    
    def get_artists_genres_batch(self, artist_ids: List[str]) -> Dict[str, List[str]]:
        client = self._get_client()
        if not client:
            return {}
        
        result = {}
        batch_size = 50
        
        for i in range(0, len(artist_ids), batch_size):
            batch = artist_ids[i:i+batch_size]
            try:
                artists_data = client.artists(batch)
                for artist in artists_data.get("artists", []):
                    if artist:
                        result[artist["id"]] = artist.get("genres", [])
            except Exception as e:
                logger.error(f"Error getting artists batch: {e}")
        
        return result
    
    def get_track_info(self, track_id: str) -> Optional[Dict[str, Any]]:
        client = self._get_client()
        if not client:
            return None
        
        try:
            track = client.track(track_id)
            return {
                "spotify_id": track["id"],
                "name": track["name"],
                "artist": ", ".join([a["name"] for a in track["artists"]]),
                "album": track["album"]["name"],
                "release_date": track["album"].get("release_date", ""),
                "popularity": track["popularity"],
                "spotify_url": track["external_urls"]["spotify"],
            }
        except Exception as e:
            logger.error(f"Error getting track info: {e}")
            return None
    
    def get_tracks_batch(self, track_ids: List[str]) -> List[Dict[str, Any]]:
        client = self._get_client()
        if not client:
            return []
        
        results = []
        batch_size = 50
        
        for i in range(0, len(track_ids), batch_size):
            batch = track_ids[i:i+batch_size]
            try:
                tracks_data = client.tracks(batch)
                for track in tracks_data.get("tracks", []):
                    if track:
                        results.append({
                            "spotify_id": track["id"],
                            "name": track["name"],
                            "artist": ", ".join([a["name"] for a in track["artists"]]),
                            "release_date": track["album"].get("release_date", ""),
                            "popularity": track["popularity"],
                            "spotify_url": track["external_urls"]["spotify"],
                        })
            except Exception as e:
                logger.error(f"Error getting tracks batch: {e}")
        
        return results


_spotify_client: Optional[SpotifyClient] = None


def get_spotify_client() -> SpotifyClient:
    global _spotify_client
    if _spotify_client is None:
        _spotify_client = SpotifyClient()
    return _spotify_client

