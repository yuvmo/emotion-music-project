import os
import csv
import logging
import tempfile
from datetime import datetime
from pathlib import Path

from aiogram import Router, F, Bot
from aiogram.types import Message, CallbackQuery, Voice
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup

from .keyboards import get_main_keyboard, get_tracks_keyboard, get_feedback_keyboard

logger = logging.getLogger(__name__)

router = Router()

FEEDBACK_FILE = Path(__file__).parent.parent / "data" / "feedback.csv"


class UserStates(StatesGroup):
    waiting_voice = State()


def save_feedback(user_id: int, feedback: str, transcript: str, emotion: str, 
                  response_text: str, tracks: list):
    file_exists = FEEDBACK_FILE.exists()
    
    tracks_str = "; ".join([f"{t.get('artist', '')} - {t.get('name', '')}" for t in tracks])
    
    with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        if not file_exists:
            writer.writerow([
                "timestamp", "user_id", "feedback", "transcript", 
                "emotion", "response_text", "tracks"
            ])
        
        writer.writerow([
            datetime.now().isoformat(),
            user_id,
            feedback,
            transcript,
            emotion,
            response_text,
            tracks_str
        ])


@router.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext):
    await state.clear()
    
    welcome_text = """
Привет! Я Music Emotion Bot

Я подберу музыку под твое настроение.

Как это работает:
1. Отправь голосовое сообщение
2. Расскажи, как себя чувствуешь или какую музыку хочешь
3. Я определю твое настроение и подберу треки

Примеры запросов:
- Хочу что-то веселое
- Грустно, поставь русский рэп
- Нужна энергичная музыка для спорта

Просто отправь голосовое!
"""
    
    await message.answer(welcome_text, reply_markup=get_main_keyboard())


@router.message(Command("help"))
async def cmd_help(message: Message):
    help_text = """
Помощь

Команды:
/start — начать сначала
/help — эта справка

Как пользоваться:
1. Отправь голосовое сообщение
2. Опиши свое настроение или пожелания
3. Получи подборку треков

Что я понимаю:
- Эмоции в голосе (веселый, грустный, злой...)
- Жанры (рок, поп, рэп, электроника...)
- Язык (русский, английский, инструментал)
- Настроение (энергичный, спокойный, романтичный...)

Примеры:
- Включи русский рок
- Хочу спокойную музыку
- Грустно сегодня...
- Нужна музыка для вечеринки

Просто говори естественно — я пойму!
"""
    
    await message.answer(help_text)


@router.message(F.text == "Помощь")
async def btn_help(message: Message):
    await cmd_help(message)


@router.message(F.text == "Подобрать музыку")
async def btn_find_music(message: Message, state: FSMContext):
    await state.set_state(UserStates.waiting_voice)
    await message.answer(
        "Отправь голосовое сообщение!\n"
        "Расскажи, какое у тебя настроение или какую музыку хочешь послушать."
    )


@router.message(F.voice)
async def handle_voice(message: Message, state: FSMContext, bot: Bot):
    processing_msg = await message.answer("Обрабатываю голосовое сообщение...")
    
    try:
        voice: Voice = message.voice
        file = await bot.get_file(voice.file_id)
        
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as temp_file:
            temp_path = temp_file.name
        
        await bot.download_file(file.file_path, temp_path)
        
        await processing_msg.edit_text("Голосовое получено! Анализирую эмоции и текст...")
        
        from src.pipeline import get_pipeline
        
        pipeline = get_pipeline()
        result = pipeline.process_audio(audio_path=temp_path, top_k=5, user_id=message.from_user.id)
        
        os.unlink(temp_path)
        
        if not result.success:
            await processing_msg.edit_text(result.error_message)
            return
        
        response_parts = [result.response_text]
        
        if result.audio_emotion:
            response_parts.append(f"\n\nЭмоция в голосе: {result.audio_emotion}")
        
        if result.transcript:
            response_parts.append(f"\nЯ услышал: {result.transcript}")
        
        response_text = "".join(response_parts)
        
        await processing_msg.edit_text(response_text)
        
        if result.tracks:
            tracks_data = [t.to_dict() for t in result.tracks]
            
            await state.update_data(
                last_transcript=result.transcript,
                last_emotion=result.audio_emotion,
                last_response=result.response_text,
                last_tracks=tracks_data
            )
            
            await message.answer(
                "Подборка для тебя:",
                reply_markup=get_tracks_keyboard(tracks_data)
            )
            
            await message.answer(
                "Подборка понравилась?",
                reply_markup=get_feedback_keyboard()
            )
        
    except Exception as e:
        logger.exception("Error processing voice message")
        await processing_msg.edit_text(
            f"Произошла ошибка при обработке. Попробуй еще раз или отправь другое сообщение."
        )
        
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)


@router.message(F.text)
async def handle_text(message: Message, state: FSMContext):
    if message.text.startswith("/"):
        return
    
    await message.answer(
        "Для подбора музыки отправь голосовое сообщение!\n"
        "Я анализирую не только слова, но и эмоции в твоем голосе.",
        reply_markup=get_main_keyboard()
    )


@router.callback_query(F.data == "more_tracks")
async def callback_more_tracks(callback: CallbackQuery, state: FSMContext):
    await callback.answer("Отправь новое голосовое для новой подборки!")
    await callback.message.answer("Отправь еще одно голосовое сообщение, и я подберу новые треки!")


@router.callback_query(F.data == "new_request")
async def callback_new_request(callback: CallbackQuery, state: FSMContext):
    await callback.answer()
    await state.clear()
    await callback.message.answer("Отправь голосовое сообщение!", reply_markup=get_main_keyboard())


@router.callback_query(F.data.startswith("feedback:"))
async def callback_feedback(callback: CallbackQuery, state: FSMContext):
    feedback_type = callback.data.split(":")[1]
    
    data = await state.get_data()
    
    logger.info(f"Feedback received: {feedback_type}, data: {data}")
    
    try:
        save_feedback(
            user_id=callback.from_user.id,
            feedback=feedback_type,
            transcript=data.get("last_transcript", ""),
            emotion=data.get("last_emotion", ""),
            response_text=data.get("last_response", ""),
            tracks=data.get("last_tracks", [])
        )
        logger.info(f"Feedback saved to {FEEDBACK_FILE}")
    except Exception as e:
        logger.error(f"Error saving feedback: {e}")
    
    if feedback_type == "good":
        await callback.answer("Спасибо! Рад, что понравилось!")
        await callback.message.edit_text("Спасибо за отзыв! Рад, что подборка понравилась!")
    else:
        await callback.answer("Учту на будущее!")
        await callback.message.edit_text(
            "Понял, в следующий раз постараюсь лучше!\n\n"
            "Попробуй описать более конкретно, что хочешь:\n"
            "- Укажи жанр (рок, поп, рэп...)\n"
            "- Укажи язык (русский, английский)\n"
            "- Опиши настроение подробнее"
        )
