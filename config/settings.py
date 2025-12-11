import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

GIGACHAT_URL = os.getenv("GIGACHAT_URL", "https://gigachat.devices.sberbank.ru/api/v1")
GIGACHAT_API_KEY = os.getenv("GIGACHAT_API_KEY", None)
GIGACHAT_CERT = os.getenv("GIGACHAT_CERT", None)
GIGACHAT_KEY = os.getenv("GIGACHAT_KEY", None)
GIGACHAT_DEBUG = os.getenv("GIGACHAT_DEBUG", "false").lower() == "true"
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "GigaChat-2-Max")

N_RETRY = int(os.getenv("N_RETRY", "3"))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", "2"))
TASK_SEMAPHORE = asyncio.Semaphore(5)
