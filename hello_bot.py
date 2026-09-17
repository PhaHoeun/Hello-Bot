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

# Optional google-genai
try:
    from google import genai
except ImportError:
    genai = None

# Optional edge-tts
try:
    import edge_tts
except ImportError:
    edge_tts = None

# =========================================================
# CONFIGURATION & API KEYS
# =========================================================
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Supported models: gemini-2.5-flash, gemini-2.0-flash, gemini-1.5-flash
# Fallback to gemini-2.5-flash if invalid/empty
raw_model = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
MODEL_NAME = "gemini-3.8-flash" if "3.8" in raw_model else raw_model

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

    models_to_try = [MODEL_NAME, "gemini-3.8-flash"]
    # De-duplicate while preserving order
    seen = set()
    models_to_try = [m for m in models_to_try if not (m in seen or seen.add(m))]

    for model in models_to_try:
        for attempt in range(2):
            try:
                response = ai_client.models.generate_content(
                    model=model,
                    contents=prompt,
                )
                if response and response.text:
                    return response.text.strip()
                return None
            except Exception as err:
                error_text = str(err)
                if "404" in error_text:
                    # Model not found, break out to try next fallback model
                    print(f"Model {model} returned 404. Falling back to next model...", flush=True)
                    break
                if "503" in error_text:
                    wait_time = (attempt + 1) * 2
                    print(f"Gemini 503 ({model}). Retrying in {wait_time}s...", flush=True)
                    time.sleep(wait_time)
                    continue
                if "429" in error_text:
                    print("Gemini 429: API rate quota reached.", flush=True)
                    return "⚠️ API rate limit reached. Please try again in a few seconds."
                print(f"Gemini API error with {model}: {err}", flush=True)
                return None
    return None


# =========================================================
# TELEGRAM COMMAND HANDLERS
# =========================================================
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler for /start command."""
    welcome_text = (
        "👋 **សួស្តី! Hello!**\n\n"
        "🤖 ខ្ញុំជា **AI Telegram Bot** ដំណើរការដោយ **Google Gemini**!\n"
        "🎙️ ខ្ញុំអាចឆ្លើយជាសំឡេង និងជាអក្សរទាំងជាភាសាខ្មែរ និងភាសាអង់គ្លេស។\n\n"
        "✨ *សាកល្បងសួរសំណួរអ្វីមួយមកកាន់ខ្ញុំឥឡូវនេះ!*\n"
        "📌 វាយ `/help` ដើម្បីមើលព័ត៌មានបន្ថែម។"
    )
    await update.message.reply_text(welcome_text, parse_mode="Markdown")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler for /help command."""
    help_text = (
        "ℹ️ **ជំនួយ និងរបៀបប្រើប្រាស់ / Bot Help**:\n\n"
        "1. គ្រាន់តែផ្ញើសារធម្មតា ខ្ញុំនឹងឆ្លើយតបមកវិញភ្លាមៗ។\n"
        "2. ភាសាខ្មែរ (Khmer): ឆ្លើយតបជាសំឡេងភាសាខ្មែរធម្មជាតិ។\n"
        "3. English: Reply with natural voice and text.\n\n"
        "⚙️ **Commands**:\n"
        "• `/start` - ចាប់ផ្តើម និងស្វាគមន៍\n"
        "• `/help` - មើលរបៀបប្រើប្រាស់"
    )
    await update.message.reply_text(help_text, parse_mode="Markdown")


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
អ្នកជាមិត្តភក្តិជិតស្និទ្ធម្នាក់ ដែលនិយាយភាសាខ្មែរបែបធម្មជាតិ។
ឆ្លើយតបដូចជាមិត្តភក្តិដែលកំពុងនិយាយជាមួយគ្នា កុំប្រើភាសាផ្លូវការ កុំឱ្យស្តាប់ទៅដូចជា robot កុំសរសេរវែង។
ឆ្លើយត្រឹម 1 ទៅ 2 ឃ្លា ប្រហែល 5 ទៅ 10 វិនាទីសម្រាប់និយាយ។
បើសមរម្យ អាចប្រើពាក្យលេងសើចដូចជា "ហាហា" ឬ "ហិហិ" បាន។
សាររបស់មិត្ត: {user_text}
"""
            voice_name = "km-KH-PisethNeural"
        else:
            prompt = f"""
You are a close, witty, and friendly friend.
Reply naturally in casual conversational English.
Do not sound like a robot. Do not sound like a textbook.
Keep the answer short and natural.
Reply with only 1 or 2 sentences, around 5 to 10 seconds of speech.
User message: {user_text}
"""
            voice_name = "en-US-AndrewNeural"

        # Generate response using Gemini in a background worker thread
        reply_text = await asyncio.to_thread(get_ai_response, prompt)
        if not reply_text:
            await update.message.reply_text("សូមអភ័យទោស ខ្ញុំមិនអាចឆ្លើយតបបានទេនៅពេលនេះ។ សូមព្យាយាមម្តងទៀត!")
            return

        print(f"Bot response: {reply_text}", flush=True)

        # Attempt to synthesize speech with edge-tts
        voice_sent = False
        voice_file = f"voice_{uuid.uuid4().hex[:8]}_{update.message.message_id}.ogg"

        if edge_tts is not None:
            try:
                communicate = edge_tts.Communicate(reply_text, voice=voice_name)
                await communicate.save(voice_file)

                if os.path.exists(voice_file) and os.path.getsize(voice_file) > 0:
                    with open(voice_file, "rb") as audio:
                        caption = reply_text if len(reply_text) <= 1024 else reply_text[:1020] + "..."
                        await context.bot.send_voice(
                            chat_id=chat_id,
                            voice=audio,
                            caption=caption,
                            reply_to_message_id=update.message.message_id,
                        )
                    voice_sent = True
                    print("Voice sent successfully!", flush=True)
            except Exception as tts_err:
                print(f"TTS conversion failed: {tts_err}", flush=True)
            finally:
                if os.path.exists(voice_file):
                    try:
                        os.remove(voice_file)
                    except OSError:
                        pass

        # If voice synthesis failed or edge_tts is unavailable, fallback to text message
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
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(error_handler)

    print("🚀 Bot is now online and polling Telegram...", flush=True)
    app.run_polling(drop_pending_updates=True)



if __name__ == "__main__":
    main()
