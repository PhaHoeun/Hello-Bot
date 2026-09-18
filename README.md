# 🤖 AI Telegram Bot (Google Gemini + Voice + Render)

An intelligent Telegram AI Bot built with **Python**, powered by **Google Gemini AI**, with real-time human-sounding voice note generation (**Neural Voice AI**) supporting both **Khmer** (ភាសាខ្មែរ) and **English**, pre-configured for 24/7 deployment on the **Render Dashboard**.

---

## 🌟 Key Features

- 🧠 **Google Gemini AI**: Utilizes Google's fast, multimodal Gemini models (`gemini-2.5-flash` / `gemini-2.0-flash`).
- 🎙️ **Natural Human Voice Notes**: Generates gentle, natural-sounding voice notes:
  - 🇰🇭 **Khmer voice**: `km-KH-SreymomNeural` (tuned with warm, natural cadence)
  - 🇺🇸 **English voice**: `en-US-JennyNeural`
  - Voice bubbles are sent purely as voice notes (no text caption).
- ⚡ **Typing / Voice Indicators**: Displays "recording voice" or "typing" status in Telegram while generating responses.
- 💬 **Command Handlers**: Includes `/start` and `/help` for clean user onboarding.
- 🚀 **Render Keep-Alive Server**: Built-in HTTP health server running on port `PORT` (default `10000`) for seamless hosting as a Render Web Service.

---

## 📋 Prerequisites

Before deploying, collect the following 2 keys:

### 1. Telegram Bot Token
1. Open Telegram and search for [@BotFather](https://t.me/BotFather).
2. Send `/newbot`.
3. Choose a display name (e.g., `My Gemini Bot`) and a username ending in `bot` (e.g., `MyGeminiAi_bot`).
4. Copy the **HTTP API token** provided (e.g., `7123456789:ABCdef...`).

### 2. Google Gemini API Key
1. Go to [Google AI Studio](https://aistudio.google.com/).
2. Sign in with your Google account.
3. Click **Get API key** -> **Create API key**.
4. Copy your API key.

---

## 🚀 Deploying to Render Dashboard

You can host this bot on Render's **Free Web Service** tier in just a few clicks.

### Step 1: Push Code to GitHub
Ensure all your project files are pushed to your GitHub repository (e.g., `https://github.com/PhaHoeun/Hello-Bot`).

```bash
git add .
git commit -m "Configure AI Telegram Bot with Gemini and Render"
git push origin main
```

### Step 2: Create Web Service on Render
1. Go to the [Render Dashboard](https://dashboard.render.com).
2. Click **New +** (top right) and select **Web Service**.
3. Under **Connect a repository**, choose your repository (`Hello-Bot`).
4. Fill in the service configuration:
   - **Name**: `hello-telegram-bot` (or any name you like)
   - **Region**: Choose the closest region (e.g., `Singapore` or `Oregon`)
   - **Branch**: `main`
   - **Root Directory**: *(leave blank)*
   - **Runtime**: `Python 3`
   - **Build Command**: 
     ```bash
     pip install -r requirements.txt
     ```
   - **Start Command**: 
     ```bash
     python hello_bot.py
     ```
   - **Instance Type**: Select **Free** ($0/month).

### Step 3: Add Environment Variables
Scroll down to the **Environment Variables** section on Render and click **Add Environment Variable**:

| Key | Value | Description |
|---|---|---|
| `TELEGRAM_TOKEN` | *your_telegram_bot_token* | From @BotFather |
| `GEMINI_API_KEY` | *your_gemini_api_key* | From Google AI Studio |
| `GEMINI_MODEL` | `gemini-2.5-flash` | (Optional, default is `gemini-2.5-flash`) |
| `PORT` | `10000` | (Render sets this automatically) |

### Step 4: Deploy & Verify
1. Click **Deploy Web Service**.
2. Watch the deployment logs in the Render console:
   - Dependencies will be installed from `requirements.txt`.
   - You should see:
     ```text
     ✅ Health server listening on 0.0.0.0:10000 for Render keep-alive
     🤖 Starting Hello-Bot with Gemini (gemini-2.5-flash)...
     🚀 Bot is now online and polling Telegram...
     ```
3. Open Telegram, find your bot, and send `/start` or ask a question!

---

## 💡 Keeping the Bot Awake 24/7 (Free Tier Tip)

Render's free web services spin down after 15 minutes of inactivity if no incoming HTTP traffic is detected.

Because `hello_bot.py` has an integrated HTTP health server responding `200 OK` to `GET /`, you can use a free pinging service:
1. Copy your Render web service URL (e.g. `https://hello-telegram-bot.onrender.com`).
2. Sign up at [cron-job.org](https://cron-job.org) or [UptimeRobot](https://uptimerobot.com).
3. Set up a monitor or cron job to ping your Render URL every **10 minutes**.
4. Your bot will remain awake 24/7!

---

## 💻 Running Locally

To run and test the bot on your computer:

```bash
# 1. Clone repository
git clone https://github.com/PhaHoeun/Hello-Bot.git
cd Hello-Bot

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create your .env file
cp .env.example .env
# Edit .env and enter your TELEGRAM_TOKEN and GEMINI_API_KEY

# 5. Run the bot
python hello_bot.py
```

---

## 📁 Project Structure

```text
Hello-Bot/
├── hello_bot.py          # Main Telegram bot + Gemini + TTS + Health server
├── requirements.txt      # Python dependencies
├── render.yaml           # Render Blueprint configuration
├── .env.example          # Sample environment variables template
├── .gitignore            # Git ignore rules (protects .env and temp audio)
└── README.md             # Documentation and deployment guide
```
