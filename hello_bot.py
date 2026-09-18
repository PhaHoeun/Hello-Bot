import os
import re
import uuid
import asyncio
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# Optional dotenv loader for local development
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.isfile(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))

import io
import wave

# Optional google-genai
try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

# Optional gTTS (Google Voice / Text-to-Speech)
try:
    from gtts import gTTS
except ImportError:
    gTTS = None

# =========================================================
# CONFIGURATION & API KEYS
# =========================================================
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Supported models: gemini-3.8-flash, gemini-3.7-flash, gemini-3.6-flash, gemini-3.5-flash
raw_model = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash").strip()
MODEL_NAME = raw_model if raw_model else "gemini-3.8-flash"

# Google AI Studio human voice configuration (Aoede, Puck, Charon, Fenrir, Kore)
DEFAULT_VOICE_GENDER = os.environ.get("VOICE_GENDER", "male").lower()

VOICE_MAP = {
    "female": {
        "gemini_voice": "Aoede",  # Google AI Studio female voice
        "label": "ស្រី (Google AI Studio: Aoede)",
    },
    "male": {
        "gemini_voice": "Puck",   # Google AI Studio male voice
        "label": "ប្រុស (Google AI Studio: Puck)",
    },
}


def pcm_to_wav_bytes(audio_bytes: bytes, mime_type: str = "") -> bytes:
    """Convert raw 16-bit PCM from Google AI Studio into a valid WAV format for Telegram."""
    if not audio_bytes:
        return audio_bytes
    # If it already contains RIFF/WAV header, return directly
    if len(audio_bytes) > 4 and audio_bytes[:4] == b"RIFF":
        return audio_bytes

    sample_rate = 24000
    if mime_type and "rate=" in mime_type:
        match = re.search(r"rate=(\d+)", mime_type)
        if match:
            try:
                sample_rate = int(match.group(1))
            except ValueError:
                pass

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_out:
        wav_out.setnchannels(1)       # Mono
        wav_out.setsampwidth(2)       # 16-bit PCM = 2 bytes per sample
        wav_out.setframerate(sample_rate)
        wav_out.writeframes(audio_bytes)
    return buf.getvalue()



def clean_text_for_tts(text: str) -> str:
    """Strip markdown, dialogue prefixes, quotes, emojis, and symbols so TTS sounds human."""
    # Remove speaker prefixes like "Bot:", "Friend:", "AI:"
    text = re.sub(r"^(Bot|AI|Friend|Me|ឆ្លើយតប|សំឡេង)\s*[:：\-]\s*", "", text, flags=re.IGNORECASE)
    # Remove markdown formatting characters
    text = re.sub(r"[*_~`#>]", "", text)
    # Remove markdown links [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    # Remove bullet points
    text = re.sub(r"^\s*[-*•]\s*", "", text, flags=re.MULTILINE)
    # Strip wrapping quotes
    text = re.sub(r'^["“\'‘](.*)["”\'’]$', r"\1", text.strip())
    # Remove emojis (which cause robotic pronunciation of symbol descriptions)
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001F900-\U0001F9FF"
        "\U0001FA70-\U0001FAFF"
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def generate_google_tts_audio(text: str, lang: str = "km") -> bytes | None:
    """Generate natural voice audio bytes using Google Voice (Khmer 'km' or English 'en')."""
    if not gTTS or not text:
        return None
    try:
        clean = clean_text_for_tts(text)
        if not clean:
            clean = text
        tts = gTTS(text=clean, lang=lang, slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf.read()
    except Exception as e:
        print(f"Google Voice synthesis error: {e}", flush=True)
        return None


ai_client = None
if GEMINI_API_KEY and genai is not None:
    try:
        ai_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"Warning: Failed to initialize Gemini Client: {e}", flush=True)


# =========================================================
# RENDER HEALTH CHECK SERVER
# =========================================================
class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status":"ok","message":"Telegram AI Bot is running smoothly on Render!"}')

    def do_HEAD(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, format, *args):
        # Silence routine health check logs to keep Render console clean
        pass


def run_web():
    port = int(os.environ.get("PORT", 10000))
    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    print(f"✅ Health server listening on 0.0.0.0:{port} for Render keep-alive", flush=True)
    server.serve_forever()


# =========================================================
# GEMINI AI RESPONSE GENERATION
# =========================================================
def get_ai_response(prompt: str) -> str | None:
    if not ai_client:
        return "⚠️ Gemini AI client is not configured. Please verify GEMINI_API_KEY."

    # Tiered models to try in order: latest 3.8 down to 3.7, 3.6, 3.5
    models_to_try = [
        MODEL_NAME,
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
    ]
    # De-duplicate while preserving order
    seen = set()
    models_to_try = [m for m in models_to_try if not (m in seen or seen.add(m))]

    # Disable automatic function calling to avoid SDK warning on generate_content
    gen_config = None
    if types is not None:
        try:
            gen_config = types.GenerateContentConfig(
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
            )
        except Exception:
            gen_config = None

    for model in models_to_try:
        for attempt in range(2):
            try:
                kwargs = {"model": model, "contents": prompt}
                if gen_config:
                    kwargs["config"] = gen_config

                response = ai_client.models.generate_content(**kwargs)
                if response and response.text:
                    return response.text.strip()
                return None
            except Exception as err:
                error_text = str(err)
                if "404" in error_text:
                    # Model not supported or not found, fall back to next model immediately
                    print(f"Model {model} returned 404. Falling back to next model...", flush=True)
                    break
                if "503" in error_text:
                    # High traffic / temporary server unavailability, retry with brief backoff
                    wait_time = (attempt + 1) * 1.5
                    print(f"Gemini 503 ({model}) - server busy. Retrying in {wait_time}s...", flush=True)
                    time.sleep(wait_time)
                    continue
                if "429" in error_text:
                    print(f"Gemini 429 ({model}) rate limit reached. Trying next model...", flush=True)
                    break
                print(f"Gemini API error with {model}: {err}", flush=True)
                break
    return None


def get_gemini_audio_response(prompt: str, voice_name: str = "Aoede") -> tuple[bytes | None, str | None]:
    """Generate human speech directly using Google AI Studio native voices (Aoede, Puck, Kore, Fenrir, Charon)."""
    if not ai_client or types is None:
        return None, None

    # Google AI Studio models supporting native audio modality
    models_to_try = [
        "gemini-2.0-flash",
        "gemini-2.5-flash",
        "gemini-2.0-flash-lite",
        MODEL_NAME,
    ]
    seen = set()
    models_to_try = [m for m in models_to_try if not (m in seen or seen.add(m))]

    for model in models_to_try:
        try:
            config = types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name)
                    )
                ),
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            )
            response = ai_client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            if response and response.candidates:
                candidate = response.candidates[0]
                audio_bytes = None
                text_content = ""
                mime_type = ""
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if getattr(part, "inline_data", None) and part.inline_data.data:
                            audio_bytes = part.inline_data.data
                            mime_type = getattr(part.inline_data, "mime_type", "") or ""
                        if getattr(part, "text", None):
                            text_content += part.text
                if audio_bytes:
                    wav_data = pcm_to_wav_bytes(audio_bytes, mime_type)
                    return wav_data, text_content.strip()
        except Exception as err:
            print(f"Gemini native audio with {model} unavailable: {err}", flush=True)
            continue
    return None, None


# =========================================================
# TELEGRAM COMMAND HANDLERS
# =========================================================
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler for /start command."""
    welcome_text = (
        "👋 **សួស្តី! Hello!**\n\n"
        "🤖 ខ្ញុំជា **AI Telegram Bot** ដំណើរការដោយ **Google Gemini**!\n"
        "🎙️ ខ្ញុំឆ្លើយតបជា **សារសំឡេង Google Voice** ទាំងភាសាខ្មែរ 🇰🇭 និងអង់គ្លេស 🇺🇸។\n\n"
        "✨ *សាកល្បងផ្ញើសារសួរសំណួរអ្វីមួយមកកាន់ខ្ញុំឥឡូវនេះ!*\n"
        "📌 វាយ `/help` ដើម្បីមើលព័ត៌មានបន្ថែម។"
    )
    await update.message.reply_text(welcome_text, parse_mode="Markdown")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler for /help command."""
    help_text = (
        "ℹ️ **ជំនួយ និងរបៀបប្រើប្រាស់ / Bot Help**:\n\n"
        "1. ផ្ញើសារជាអក្សរធម្មតា ខ្ញុំនឹងឆ្លើយតបជាសារសំឡេងមនុស្សពិតៗ (Google Voice Note) មកវិញភ្លាមៗ។\n"
        "2. 🇰🇭 ភាសាខ្មែរ: សំឡេង Google Khmer Voice ធម្មជាតិទន់ភ្លន់។\n"
        "3. 🇺🇸 ភាសាអង់គ្លេស: សំឡេង Google English Voice។\n\n"
        "⚙️ **Commands**:\n"
        "• `/start` - ចាប់ផ្តើម និងស្វាគមន៍\n"
        "• `/voice` - ព័ត៌មានពីសំឡេង Google Voice\n"
        "• `/help` - មើលរបៀបប្រើប្រាស់"
    )
    await update.message.reply_text(help_text, parse_mode="Markdown")


async def voice_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler for /voice command."""
    await update.message.reply_text(
        "🎙️ **Google Voice Configuration**:\n\n"
        "• 🇰🇭 **ភាសាខ្មែរ**: សំឡេងផ្លូវការ Google Khmer Voice\n"
        "• 🇺🇸 **English**: Google English Voice\n\n"
        "✨ រាល់ពេលអ្នកផ្ញើសារមក Bot នឹងឆ្លើយតបជាសំឡេង Voice Note ដោយស្វ័យប្រវត្តិ!",
        parse_mode="Markdown",
    )


# =========================================================
# TELEGRAM MESSAGE HANDLER
# =========================================================
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    chat_id = update.effective_chat.id
    user_text = update.message.text.strip()
    print(f"Incoming message from {chat_id}: {user_text}", flush=True)

    # Indicate activity to the user in Telegram
    await context.bot.send_chat_action(chat_id=chat_id, action="record_voice")

    try:
        has_khmer = bool(re.search(r"[\u1780-\u17FF]", user_text))

        if has_khmer:
            prompt = f"""
អ្នកជាមិត្តភក្តិជិតស្និទ្ធម្នាក់កំពុងផ្ញើសារសំឡេង (Voice Note) តាម Telegram ទៅកាន់មិត្ត។
ច្បាប់សំខាន់បំផុត ដើម្បីកុំឱ្យសំឡេងស្តាប់ទៅដូចជា Robot ឬ AI:
1. ត្រូវប្រើភាសានិយាយប្រចាំថ្ងៃ (Spoken Khmer) ធម្មជាតិៗ រួសរាយ រាក់ទាក់ កក់ក្តៅ និងមានជីវិតជីវ៉ា។
2. ហាមដាច់ខាតប្រើភាសាផ្លូវការ រដ្ឋបាល ឬភាសាសៀវភៅ ដូចជា "តើខ្ញុំអាចជួយអ្វីបាន"។
3. អាចប្រើពាក្យធម្មជាតិដូចជា "បាទបង/ចាសបង", "អរគុណច្រើន!", "ហ្នឹងហើយ", "អូ!", "មែនតើ" តាមការគួរសម។
4. ឆ្លើយខ្លីៗល្មមស្តាប់ត្រឹម 1 ទៅ 2 ឃ្លា (ប្រហែល 5 ទៅ 10 វិនាទី)។
5. ហាមប្រើសញ្ញាផ្កាយ (*), hashtag (#), emoji ឬ markdown ព្រោះសារនេះនឹងត្រូវអានផ្ទាល់ជាសំឡេង។
សាររបស់មិត្ត: {user_text}
"""
        else:
            prompt = f"""
You are recording a quick, authentic voice note on Telegram for a close friend.
Crucial rules to sound completely human (NOT like an AI assistant or robot):
1. Speak in casual, conversational everyday speech. Sound warm, relaxed, and spontaneous.
2. Absolutely DO NOT sound like a robotic customer service bot (never say "How may I assist you today?" or "As an AI...").
3. Use natural contractions and spoken rhythm ("Hey!", "I'm", "don't", "yeah", "honestly", "gotcha").
4. Keep it concise: 1 or 2 spoken sentences (about 5 to 10 seconds).
5. Absolutely NO emojis, asterisks (*), hashtags (#), bullet points, or markdown formatting, as this text is read aloud directly.
Friend's message: {user_text}
"""

        # 1. Get smart response from Gemini AI
        reply_text = await asyncio.to_thread(get_ai_response, prompt)
        if not reply_text:
            await update.message.reply_text("សូមអភ័យទោស ខ្ញុំមិនអាចឆ្លើយតបបានទេនៅពេលនេះ។ សូមព្យាយាមម្តងទៀត!")
            return

        print(f"Bot response: {reply_text}", flush=True)

        # 2. Synthesize Google Voice Note (Khmer 'km' or English 'en')
        voice_sent = False
        lang = "km" if has_khmer else "en"
        audio_bytes = await asyncio.to_thread(generate_google_tts_audio, reply_text, lang)

        if audio_bytes:
            voice_file = f"voice_google_{uuid.uuid4().hex[:8]}_{update.message.message_id}.mp3"
            try:
                with open(voice_file, "wb") as f:
                    f.write(audio_bytes)

                if os.path.exists(voice_file) and os.path.getsize(voice_file) > 0:
                    with open(voice_file, "rb") as audio:
                        try:
                            await context.bot.send_voice(
                                chat_id=chat_id,
                                voice=audio,
                                caption=reply_text,
                                reply_to_message_id=update.message.message_id,
                            )
                            voice_sent = True
                        except Exception as send_voice_err:
                            print(f"send_voice fallback to send_audio: {send_voice_err}", flush=True)
                            audio.seek(0)
                            await context.bot.send_audio(
                                chat_id=chat_id,
                                audio=audio,
                                title="Google Voice Note",
                                caption=reply_text,
                                reply_to_message_id=update.message.message_id,
                            )
                            voice_sent = True

                    if voice_sent:
                        print(f"✅ Sent Google Voice Note successfully ({lang})!", flush=True)
            except Exception as voice_err:
                print(f"Error delivering Google voice note: {voice_err}", flush=True)
            finally:
                if os.path.exists(voice_file):
                    try:
                        os.remove(voice_file)
                    except OSError:
                        pass

        # 3. If voice generation failed, fallback directly to text message
        if not voice_sent:
            await update.message.reply_text(
                reply_text,
                reply_to_message_id=update.message.message_id,
            )

    except Exception as e:
        print(f"Error in handle_message: {e}", flush=True)
        try:
            await update.message.reply_text("⚠️ មានបញ្ហាក្នុងការដំណើរការសាររបស់អ្នក។ សូមព្យាយាមម្តងទៀត!")
        except Exception:
            pass


# =========================================================
# GLOBAL ERROR HANDLER
# =========================================================
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    err_str = str(context.error)
    if "Conflict" in err_str or "terminated by other getUpdates request" in err_str:
        print(
            "\n⚠️ [TELEGRAM CONFLICT]: Another bot instance is already running with this TELEGRAM_TOKEN!\n"
            "👉 If your bot is deployed on Render, please stop the local python process.\n"
            "👉 Telegram only permits ONE active polling instance per bot token at any time.\n",
            flush=True,
        )
    else:
        print(f"⚠️ Telegram bot error: {context.error}", flush=True)


# =========================================================
# START BOT APPLICATION
# =========================================================
def main():
    if not TELEGRAM_TOKEN:
        print("❌ TELEGRAM_TOKEN is missing! Set it in .env or Render Dashboard Environment Variables.", flush=True)
        return

    if not GEMINI_API_KEY:
        print("❌ GEMINI_API_KEY is missing! Set it in .env or Render Dashboard Environment Variables.", flush=True)
        return

    # Start Render keep-alive HTTP server on background thread
    web_thread = threading.Thread(target=run_web, daemon=True)
    web_thread.start()

    print("========================================", flush=True)
    print(f"🤖 Starting Hello-Bot with Gemini ({MODEL_NAME})...", flush=True)
    print("========================================", flush=True)

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    # Register handlers
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("voice", voice_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(error_handler)

    print("🚀 Bot is now online and polling Telegram...", flush=True)
    app.run_polling(drop_pending_updates=True)



if __name__ == "__main__":
    main()
