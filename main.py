import os
import logging
import sqlite3
import datetime
from dotenv import load_dotenv

import openai
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

# Константы
SUMMARY_THRESHOLD = 10  # Если в диалоге больше 10 сообщений, генерировать summary

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения из .env
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

def initialize_db():
    """
    Инициализирует базу данных и создает таблицы:
    - clients: для информации о клиентах,
    - chats: для хранения детальной истории диалогов,
    - knowledge: для хранения постоянного контекста (база знаний).
    Если таблица knowledge пуста, вставляется базовый контекст.
    """
    conn = sqlite3.connect('fomo_ai.db')
    c = conn.cursor()

    # Таблица для хранения информации о клиентах
    c.execute('''
        CREATE TABLE IF NOT EXISTS clients (
            user_id INTEGER PRIMARY KEY,
            name TEXT,
            business_type TEXT,
            pain_points TEXT,
            status TEXT
        )
    ''')

    # Таблица для хранения истории чата
    c.execute('''
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            sender TEXT,
            message TEXT,
            timestamp TEXT
        )
    ''')

    # Таблица для хранения базы знаний (постоянного контекста)
    c.execute('''
        CREATE TABLE IF NOT EXISTS knowledge (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT
        )
    ''')
    conn.commit()

    # Если таблица knowledge пуста, вставляем базовый контекст
    c.execute("SELECT COUNT(*) FROM knowledge")
    if c.fetchone()[0] == 0:
        default_context = (
            "Ты AI-агент FOMO AI Automation – профессиональный продавец для B2B. "
            "Выявляй боли клиентов и задавай наводящие вопросы для квалификации лидов. "
            "Адаптируй стиль общения под собеседника. Если вопрос пользователя не касается автоматизации продаж, "
            "недвижимости, бронирования или других ключевых тем бизнеса, аккуратно направляй диалог обратно к основной теме, "
            "генерируя ответ динамически, без использования заранее заданных шаблонов. "
            "Отвечай короткими, человечными сообщениями."
        )
        c.execute("INSERT INTO knowledge (content) VALUES (?)", (default_context,))
        conn.commit()

    conn.close()

def get_knowledge_context() -> str:
    """
    Извлекает базовый контекст из таблицы knowledge.
    Возвращает строку с контекстом или пустую строку, если данных нет.
    """
    conn = sqlite3.connect('fomo_ai.db')
    c = conn.cursor()
    c.execute("SELECT content FROM knowledge ORDER BY id LIMIT 1")
    row = c.fetchone()
    conn.close()
    return row[0] if row else ""

def log_chat(user_id: int, sender: str, message: str):
    """
    Записывает сообщение в базу данных (таблица chats).
    """
    conn = sqlite3.connect('fomo_ai.db')
    c = conn.cursor()
    c.execute(
        "INSERT INTO chats (user_id, sender, message, timestamp) VALUES (?, ?, ?, ?)",
        (user_id, sender, message, datetime.datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

def summarize_chat_history(history: list) -> str:
    """
    Генерирует summary (ключевые поинты) для заданной истории диалога.
    history: список кортежей (sender, message)
    Возвращает сгенерированный текст summary.
    """
    prompt = "Сформируй краткий конспект ключевых моментов следующего диалога (без лишних деталей):\n\n"
    for sender, message in history:
        prompt += f"{sender}: {message}\n"
    prompt += "\nКраткий конспект:"
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Ты помощник, который суммирует диалоги, выделяя ключевые поинты."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.5,
        )
        summary = response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error summarizing chat: {e}")
        summary = "Ключевые моменты диалога недоступны."
    return summary

def get_chat_context(user_id: int) -> str:
    """
    Получает историю диалога для данного пользователя.
    Если число сообщений превышает SUMMARY_THRESHOLD, возвращает summary (ключевые поинты),
    иначе возвращает полный текст диалога.
    """
    conn = sqlite3.connect('fomo_ai.db')
    c = conn.cursor()
    c.execute("SELECT sender, message FROM chats WHERE user_id = ? ORDER BY id ASC", (user_id,))
    rows = c.fetchall()
    conn.close()
    
    if len(rows) > SUMMARY_THRESHOLD:
        return summarize_chat_history(rows)
    else:
        conversation = ""
        for sender, message in rows:
            conversation += f"{sender}: {message}\n"
        return conversation

def generate_response(user_message: str) -> str:
    """
    Генерирует ответ через OpenAI gpt-4o-mini, используя базовый контекст и историю диалога (в виде summary или подробного текста).
    """
    knowledge_context = get_knowledge_context()
    chat_context = get_chat_context(user_message.user_id) if hasattr(user_message, "user_id") else ""  # Если потребуется, можно передавать user_id отдельно
    # Для генерации запроса формируем объединённый контекст:
    conversation = f"Контекст: {knowledge_context}\n\n{chat_context}\nUser: {user_message}\nAI:"
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ты AI-агент FOMO AI Automation – профессиональный продавец для B2B. "
                        "Выявляй боли клиентов и задавай наводящие вопросы для квалификации лидов. "
                        "Адаптируй стиль общения под собеседника. Если вопрос пользователя не касается автоматизации продаж, "
                        "недвижимости, бронирования или других ключевых тем бизнеса, аккуратно направляй диалог обратно к основной теме, "
                        "генерируя ответ динамически, без использования заранее заданных шаблонов. "
                        "Отвечай короткими, человечными сообщениями."
                    )
                },
                {"role": "user", "content": conversation}
            ],
            max_tokens=300,
            temperature=0.7,
        )
        ai_message = response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        ai_message = "Извините, произошла ошибка при генерации ответа."
    return ai_message

def process_voice(file_path: str) -> str:
    """
    Обрабатывает голосовое сообщение через OpenAI Whisper API и возвращает распознанный текст.
    """
    try:
        with open(file_path, "rb") as audio_file:
            transcript = openai.Audio.transcribe("whisper-1", audio_file)
        return transcript.get("text", "")
    except Exception as e:
        logger.error(f"Error processing voice message: {e}")
        return ""

def process_image(file_path: str) -> str:
    """
    Обрабатывает изображение.
    Возвращает метку, что пользователь отправил изображение.
    """
    return "Пользователь отправил изображение."

def start(update: Update, context: CallbackContext):
    """
    Обработчик команды /start.
    Отправляет приветственное сообщение и логирует его.
    """
    user = update.effective_user
    welcome_text = "Здравствуйте, меня зовут FOMO AI Automation. Чем могу помочь?"
    update.message.reply_text(welcome_text)
    log_chat(user.id, "bot", welcome_text)

def handle_text(update: Update, context: CallbackContext):
    """
    Обработчик текстовых сообщений.
    Сохраняет сообщение и генерирует ответ через gpt-4o-mini с учетом всей истории (в виде summary, если нужно).
    """
    user = update.effective_user
    text = update.message.text.strip()
    log_chat(user.id, "user", text)

    # Получаем контекст диалога (summary или полный текст)
    chat_context = get_chat_context(user.id)
    # Формируем ответ с учетом нового сообщения
    conversation = f"Контекст: {get_knowledge_context()}\n\n{chat_context}\nUser: {text}\nAI:"
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ты AI-агент FOMO AI Automation – профессиональный продавец для B2B. "
                        "Выявляй боли клиентов и задавай наводящие вопросы для квалификации лидов. "
                        "Адаптируй стиль общения под собеседника. Если вопрос пользователя не касается автоматизации продаж, "
                        "недвижимости, бронирования или других ключевых тем бизнеса, аккуратно направляй диалог обратно к основной теме, "
                        "генерируя ответ динамически, без использования заранее заданных шаблонов. "
                        "Отвечай короткими, человечными сообщениями."
                    )
                },
                {"role": "user", "content": conversation}
            ],
            max_tokens=300,
            temperature=0.7,
        )
        ai_response = response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        ai_response = "Извините, произошла ошибка при генерации ответа."
    
    update.message.reply_text(ai_response)
    log_chat(user.id, "bot", ai_response)

def handle_photo(update: Update, context: CallbackContext):
    """
    Обработчик сообщений с фотографиями.
    Сохраняет изображение, регистрирует факт отправки изображения и генерирует динамический ответ.
    """
    user = update.effective_user
    photo_file = update.message.photo[-1].get_file()
    os.makedirs("downloads", exist_ok=True)
    file_path = os.path.join("downloads", f"{photo_file.file_id}.jpg")
    photo_file.download(file_path)

    image_prompt = process_image(file_path)
    log_chat(user.id, "user", image_prompt)

    # Получаем обновленный контекст диалога
    chat_context = get_chat_context(user.id)
    conversation = f"Контекст: {get_knowledge_context()}\n\n{chat_context}\nUser: {image_prompt}\nAI:"
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ты AI-агент FOMO AI Automation – профессиональный продавец для B2B. "
                        "Выявляй боли клиентов и задавай наводящие вопросы для квалификации лидов. "
                        "Адаптируй стиль общения под собеседника. Если вопрос пользователя не касается автоматизации продаж, "
                        "недвижимости, бронирования или других ключевых тем бизнеса, аккуратно направляй диалог обратно к основной теме, "
                        "генерируя ответ динамически, без использования заранее заданных шаблонов. "
                        "Отвечай короткими, человечными сообщениями."
                    )
                },
                {"role": "user", "content": conversation}
            ],
            max_tokens=300,
            temperature=0.7,
        )
        ai_response = response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        ai_response = "Извините, произошла ошибка при генерации ответа."
    
    update.message.reply_text(ai_response)
    log_chat(user.id, "bot", ai_response)

def handle_voice(update: Update, context: CallbackContext):
    """
    Обработчик голосовых сообщений.
    Распознаёт речь и сразу генерирует ответ через gpt-4o-mini без отправки транскрипции.
    """
    user = update.effective_user
    voice = update.message.voice
    file = voice.get_file()
    os.makedirs("downloads", exist_ok=True)
    file_path = os.path.join("downloads", f"{voice.file_id}.ogg")
    file.download(file_path)

    transcribed_text = process_voice(file_path)
    if transcribed_text:
        log_chat(user.id, "user", transcribed_text)
        chat_context = get_chat_context(user.id)
        conversation = f"Контекст: {get_knowledge_context()}\n\n{chat_context}\nUser: {transcribed_text}\nAI:"
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Ты AI-агент FOMO AI Automation – профессиональный продавец для B2B. "
                            "Выявляй боли клиентов и задавай наводящие вопросы для квалификации лидов. "
                            "Адаптируй стиль общения под собеседника. Если вопрос пользователя не касается автоматизации продаж, "
                            "недвижимости, бронирования или других ключевых тем бизнеса, аккуратно направляй диалог обратно к основной теме, "
                            "генерируя ответ динамически, без использования заранее заданных шаблонов. "
                            "Отвечай короткими, человечными сообщениями."
                        )
                    },
                    {"role": "user", "content": conversation}
                ],
                max_tokens=300,
                temperature=0.7,
            )
            ai_response = response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            ai_response = "Извините, произошла ошибка при генерации ответа."
        
        update.message.reply_text(ai_response)
        log_chat(user.id, "bot", ai_response)
    else:
        error_reply = "Извините, не удалось распознать голосовое сообщение."
        update.message.reply_text(error_reply)
        log_chat(user.id, "bot", error_reply)

def main():
    # Инициализируем базу данных и таблицы
    initialize_db()

    # Создаем updater и dispatcher для Telegram-бота
    updater = Updater(token=TELEGRAM_BOT_TOKEN, use_context=True)
    dp = updater.dispatcher

    # Регистрируем обработчики команд и сообщений
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(Filters.photo, handle_photo))
    dp.add_handler(MessageHandler(Filters.voice, handle_voice))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_text))

    # Запускаем бота
    updater.start_polling()
    logger.info("Бот запущен. Ожидание сообщений...")
    updater.idle()

if __name__ == '__main__':
    main()
