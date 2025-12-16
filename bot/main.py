import os
import sys
import ssl
import asyncio
import logging
import certifi

ssl._create_default_https_context = ssl._create_unverified_context

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.handlers import router

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def preload_models():
    logger.info("Preloading models...")
    
    logger.info("Loading Whisper...")
    import whisper
    whisper.load_model("small")
    
    logger.info("Loading HuBERT emotion model...")
    from transformers import pipeline
    pipeline("audio-classification", model="superb/hubert-large-superb-er")
    
    logger.info("Models loaded!")


async def main():
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    
    if not bot_token:
        logger.error("TELEGRAM_BOT_TOKEN not found in .env")
        sys.exit(1)
    
    preload_models()
    
    bot = Bot(token=bot_token, default=DefaultBotProperties())
    dp = Dispatcher()
    dp.include_router(router)
    
    logger.info("Bot starting...")
    
    try:
        await bot.delete_webhook(drop_pending_updates=True)
        logger.info("Bot started! Waiting for messages...")
        await dp.start_polling(bot)
        
    except Exception as e:
        logger.exception(f"Bot error: {e}")
        
    finally:
        await bot.session.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped")
