import os
import re
import asyncio
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from google import genai
import edge_tts
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    MessageHandler,
    filters,
)

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

# =========================================================
# CONFIGURATION & API KEYS
# =========================================================
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not TELEGRAM_TOKEN:
    raise ValueError("TELEGRAM_TOKEN is missing! Please set it in .env or environment variables.")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing! Please set it in .env or environment variables.")

ai_client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-3.5-flash-lite"

# =========================================================
# RENDER HEALTH CHECK SERVER
# =========================================================
class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")

    def do_HEAD(self):
        self.send_response(200)
        self.end_headers()


def run_web():
    port = int(os.environ.get("PORT", 10000))
    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    print(f"Health server running on port {port}", flush=True)
    server.serve_forever()


# =========================================================
# GEMINI AI RESPONSE
# =========================================================
def get_ai_response(prompt):
    for attempt in range(3):
        try:
            response = ai_client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
            )
            if response and response.text:
                return response.text.strip()
            return None
        except Exception as err:
            error_text = str(err)
            if "503" in error_text:
                if attempt < 2:
                    wait_time = (attempt + 1) * 3
                    print(f"Gemini 503. Waiting {wait_time}s...", flush=True)
                    time.sleep(wait_time)
                    continue
                print("Gemini 503: server unavailable after retries.", flush=True)
                return None
            if "429" in error_text:
                print("Gemini 429: API quota exceeded.", flush=True)
                return None
            print(f"Gemini error: {err}", flush=True)
            return None
    return None


# =========================================================
# TELEGRAM MESSAGE HANDLER
# =========================================================
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    chat_id = update.effective_chat.id
    user_text = update.message.text.strip()
    print(f"សារទទួលបាន: {user_text}", flush=True)

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
You are a close, witty and friendly friend.
Reply naturally in casual conversational English.
Do not sound like a robot. Do not sound like a textbook.
Keep the answer very short.
Reply with only 1 or 2 sentences, around 5 to 10 seconds of speech.
User message: {user_text}
"""
            voice_name = "en-US-AndrewNeural"

        reply_text = await asyncio.to_thread(get_ai_response, prompt)
        if not reply_text:
            print("No AI response. Skipping voice generation.", flush=True)
            return

        print(f"Bot ឆ្លើយ: {reply_text}", flush=True)

        voice_file = f"voice_{update.message.message_id}.ogg"
        communicate = edge_tts.Communicate(reply_text, voice=voice_name)
        await communicate.save(voice_file)

        with open(voice_file, "rb") as audio:
            await context.bot.send_voice(
                chat_id=chat_id,
                voice=audio,
                reply_to_message_id=update.message.message_id,
            )
        print("Voice sent successfully!", flush=True)

        if os.path.exists(voice_file):
            os.remove(voice_file)

    except Exception as e:
        print(f"Error in handle_message: {e}", flush=True)


# =========================================================
# START BOT
# =========================================================
if __name__ == "__main__":
    web_thread = threading.Thread(target=run_web, daemon=True)
    web_thread.start()

    print("================================", flush=True)
    print("Bot is starting with Gemini...", flush=True)
    print("================================", flush=True)

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Bot is running polling now...", flush=True)
    app.run_polling(drop_pending_updates=True)
