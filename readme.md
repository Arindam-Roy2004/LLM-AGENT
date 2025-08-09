# AI Assistant Chat (Streamlit + LangGraph)

An AI assistant built with Streamlit that can:
- 🌤️ Get current weather via WeatherAPI
- 📈 Fetch previous closing stock prices via yfinance
- 🔍 Search the web via Tavily
- 💬 Chat for general queries using Google Gemini

Main entry point: [app.py](app.py)

Tools implemented in code:
- [`get_weather`](app.py) – current weather by city
- [`get_stock_price`](app.py) – previous closing price for a ticker
- [`search_tool`](app.py) – web search via Tavily
- Workflow: [`initialize_workflow`](app.py)

## Tech Stack
- Streamlit UI
- LangChain + LangGraph orchestration
- Google Gemini via `langchain-google-genai`
- Tavily Search
- WeatherAPI
- yfinance

## Project Structure
- [app.py](app.py) – Streamlit app and LangGraph workflow
- [requirements.txt](requirements.txt) – Python dependencies
- [requirement.txt](requirement.txt) – duplicate dependency file (see note below)

Note: You have both `requirements.txt` and `requirement.txt`. Keep only one (recommend: `requirements.txt`) to avoid deployment confusion.

## Prerequisites
- Python 3.10+
- API keys (enter in the app sidebar at runtime):
  - Google API Key (Gemini)
  - Weather API Key (weatherapi.com)
  - Tavily API Key (tavily.com)

## Setup
1) Clone and create a virtual environment
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

2) Install dependencies
```bash
pip install -r requirements.txt
```

3) Run the app
```bash
streamlit run app.py
```

## Using the App
- Open the sidebar:
  - Paste your Google API Key (required)
  - Optionally add Weather API Key and Tavily API Key
  - Toggle “Debug Mode” for detailed troubleshooting
- Use “Test Weather API” / “Test Search API” to verify keys
- Start chatting in the input box

## Deployment (Streamlit Community Cloud)
1) Push this repo to GitHub.
2) Go to https://share.streamlit.io and connect your GitHub repo.
3) Select:
   - App file: `app.py`
   - Python version: 3.10+
   - Dependencies: `requirements.txt`
4) Secrets/keys: You can keep using the sidebar inputs. If you prefer defaults, set them in Streamlit “Secrets”:
   - GOOGLE_API_KEY
   - WEATHERAPI_API_KEY
   - TAVILY_API_KEY
5) Deploy. You’ll get a public URL like:
   - https://<your-username>-<your-repo>-<random>.streamlit.app

## Troubleshooting
- If deployment fails due to deps, ensure you only have one dependency file. Keep [requirements.txt](requirements.txt) and remove [requirement.txt](requirement.txt).
- Weather API 401/403: verify the key and plan on weatherapi.com.
- Tavily errors: ensure the key is active and valid.
- Stock data issues: verify the ticker symbol and market availability.

---
