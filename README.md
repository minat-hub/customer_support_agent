# Customer Support AI Assistant

An interactive Streamlit app that lets you answers technical questions. It builds a local vector store with Chroma and answers using an LLM (Groq-hosted or local Ollama).

## Features
- CSV data preview and ingestion (expects `question` and `answer` columns)
- Text chunking with `langchain_text_splitters`
- Embeddings via `sentence-transformers/all-MiniLM-L6-v2`
- Vector store using `Chroma`
- LLM options:
  - Groq (cloud): `openai/gpt-oss-120b`
  - Ollama (local): e.g., `llama3.2`

## Prerequisites
- Windows with PowerShell (commands below use PowerShell syntax)
- Python 3.11 recommended
- (Optional) [Ollama](https://ollama.com) installed and running if you want to use local LLMs
- Groq account + API key if you want to use Groq: https://console.groq.com

## Quick Start

### 1) Clone/download this project
Place the folder anywhere, e.g., `C:\Users\\<you>\\Downloads\\customer_support_agent`.

### 2) Create and activate a virtual environment
```powershell
# From the project root
python -m venv supportenv

# Activate it
. .\supportenv\Scripts\Activate.ps1
```

### 3) Install dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### 4) Set up environment variables
Create a `.env` file in the project root (this repo already ignores it via `.gitignore`).

```env
# .env
GROQ_API_KEY=your_groq_api_key_here
```

To obtain a Groq API key: https://console.groq.com → API Keys → Create API Key.

### 5) Run the app
```powershell
streamlit run app.py
```
Streamlit will print a local URL (e.g., http://localhost:8501). Open it in your browser.

## macOS and Linux Setup

The steps are similar, but commands use bash/zsh and `python3`.

### 1) Create and activate a virtual environment
```bash
# From the project root
python3 -m venv supportenv

# Activate it (macOS/Linux)
source supportenv/bin/activate
```

### 2) Install dependencies
```bash
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3) Set up environment variables
Create a `.env` file in the project root:

```bash
printf "GROQ_API_KEY=your_groq_api_key_here\n" > .env
```

Alternatively, export the variable in your shell session (not persisted):

```bash
export GROQ_API_KEY=your_groq_api_key_here
```

### 4) Run the app
```bash
streamlit run app.py
```
Open the printed local URL (e.g., http://localhost:8501).

### 5) (Optional) Use Ollama locally
- macOS (Homebrew):
  ```bash
  brew install ollama
  ollama serve
  ollama pull llama3.2
  ```

- Linux:
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  # Start the server (keep this running in a separate terminal)
  ollama serve
  # On systemd-based distros you can alternatively do:
  # sudo systemctl enable --now ollama
  ollama pull llama3.2
  ```

Keep the `ollama serve` process running while you use the app.

To deactivate the virtual environment when done:
```bash
deactivate
```

## Switching LLMs
The app supports both Groq (cloud) and Ollama (local). In `app.py` you can toggle which LLM to use.

Current default in code:
```python
from langchain_groq import ChatGroq
# ...
llm = ChatGroq(model="openai/gpt-oss-120b")  # Uses GROQ_API_KEY from .env
```

To use Ollama instead, ensure Ollama is running and a model is installed (e.g., `llama3.2`):
```powershell
ollama list
# or install a model
ollama pull llama3.2
```
Then in `app.py`, switch to:
```python
from langchain_ollama import OllamaLLM
# ...
llm = OllamaLLM(model="llama3.2")
```

## CSV Format
Your CSV should include at least these columns:
- `question`
- `answer`

An example file is provided: `sample_customer_data.csv`.

## Troubleshooting
- ModuleNotFoundError for old LangChain imports:
  - This repo uses the new modular packages (`langchain_community`, `langchain_huggingface`, `langchain_text_splitters`). Avoid legacy `from langchain.vectorstores ...` imports.
- Embeddings error about meta tensors:
  - We pin the embeddings to CPU with `model_kwargs={"device": "cpu"}`. Ensure your environment matches.
- Groq auth errors:
  - Confirm `.env` contains `GROQ_API_KEY` and that `load_dotenv()` is called near the top of `app.py`.
- Ollama 404 (model not found):
  - Run `ollama list` to see installed models. Pull the model you configured (e.g., `ollama pull llama3.2`).

## Deploying
- Streamlit Community Cloud: Push this repo to GitHub and deploy via https://streamlit.io/cloud (note: Ollama won't be available there; prefer Groq).
- Docker/Railway/Render/AWS: Containerize with a Dockerfile and run on your provider. Ensure you set `GROQ_API_KEY` in the service environment.

## Project Structure
```
.
├── app.py                    # Streamlit application
├── requirements.txt          # Python dependencies
├── sample_customer_data.csv  # Example dataset
├── .env                      # Local secrets (not committed)
└── .gitignore                # Ignores .env and venv
```

