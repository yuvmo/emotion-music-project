import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

TRACKS_DATASET_PATH = DATA_DIR / "tracks_with_language_FINAL.csv"
GENRES_DATASET_PATH = DATA_DIR / "all_genres_from_tracks_dataset.csv"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

GIGACHAT_API_KEY = os.getenv("GIGACHAT_API_KEY")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "GigaChat-2-Max")

SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")
TARGET_SAMPLE_RATE = 16000

DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))

N_RETRY = int(os.getenv("N_RETRY", "3"))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", "2"))
TASK_SEMAPHORE = asyncio.Semaphore(5)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
