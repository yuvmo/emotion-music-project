from aiogram.types import (
    InlineKeyboardMarkup, 
    InlineKeyboardButton,
    ReplyKeyboardMarkup,
    KeyboardButton
)


def get_main_keyboard() -> ReplyKeyboardMarkup:
    keyboard = ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="Подобрать музыку")],
            [KeyboardButton(text="Помощь")]
        ],
        resize_keyboard=True,
        one_time_keyboard=False
    )
    return keyboard


def format_track_button(track: dict, max_len: int = 45) -> str:
    artist = track.get("artist", "")
    name = track.get("name", "")
    
    if not artist or artist.lower() in ["unknown", "nan", "none", "unknown artist"]:
        artist = ""
    
    if not name or name.lower() in ["unknown", "nan", "none"]:
        name = "track"
    
    artist = artist.lower()
    name = name.lower()
    
    if artist:
        text = f"{artist} — {name}"
    else:
        text = name
    
    if len(text) > max_len:
        text = text[:max_len-1] + "…"
    
    return text


def get_tracks_keyboard(tracks: list) -> InlineKeyboardMarkup:
    buttons = []
    
    for track in tracks[:5]:
        url = track.get("spotify_url") or track.get("url", "")
        spotify_id = track.get("spotify_id", "")
        
        if not url and spotify_id and spotify_id not in ["nan", "None", ""]:
            url = f"https://open.spotify.com/track/{spotify_id}"
        
        if url and url != "#" and "nan" not in url.lower():
            text = format_track_button(track)
            buttons.append([
                InlineKeyboardButton(text=text, url=url)
            ])
    
    buttons.append([
        InlineKeyboardButton(text="Другие треки", callback_data="more_tracks"),
        InlineKeyboardButton(text="Новый запрос", callback_data="new_request"),
    ])
    
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def get_feedback_keyboard() -> InlineKeyboardMarkup:
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="Отлично", callback_data="feedback:good"),
            InlineKeyboardButton(text="Не то", callback_data="feedback:bad"),
        ]
    ])
    return keyboard
