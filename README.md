# Custom Text RAG Chatbot

A Streamlit-based Retrieval-Augmented Generation (RAG) chatbot that answers questions based on user-provided text using 100% free and open-source models.

## Features

- **Text Input**: Paste any large block of text (paragraph, article, etc.)
- **RAG Pipeline**: Uses RecursiveCharacterTextSplitter, Hugging Face embeddings, and FAISS vector database
- **Free Models**: sentence-transformers/all-MiniLM-L6-v2 for embeddings, microsoft/phi-2 for text generation
- **Chat Interface**: Ask questions and get answers grounded in the provided text

## Prerequisites

- Python 3.8 or higher
- Hugging Face account (free)

## Installation and Setup

### 1. Clone or Download the Project

Ensure you have the following files in your project directory:
- `app.py`
- `requirements.txt`

### 2. Create and Activate Virtual Environment

Open a terminal/command prompt in the project directory and run:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
# source venv/bin/activate
```

### 3. Install Dependencies

With the virtual environment activated:

```bash
pip install -r requirements.txt
```

### 4. Get Hugging Face Access Token

1. Go to [https://huggingface.co/](https://huggingface.co/) and create a free account
2. Click on your profile picture → Settings → Access Tokens
3. Click "New token"
4. Give it a name (e.g., "RAG Chatbot")
5. Select "Read" permissions (free tier)
6. Click "Generate token"
7. **Copy the token immediately** (you won't be able to see it again)

### 5. Set Environment Variable for Hugging Face Token

**Important**: You must set the `HF_TOKEN` environment variable before running the app.

#### Option A: Set in Terminal (Recommended)

In the same terminal where you activated the virtual environment:

```bash
# Replace YOUR_ACTUAL_TOKEN_HERE with your copied token
set HF_TOKEN=YOUR_ACTUAL_TOKEN_HERE
```

**Note**: On Windows, use `set`. On macOS/Linux, use `export HF_TOKEN=YOUR_ACTUAL_TOKEN_HERE`

#### Option B: Create a .env file (Alternative)

1. Create a file named `.env` in the project directory
2. Add this line to the `.env` file:
   ```
   HF_TOKEN=YOUR_ACTUAL_TOKEN_HERE
   ```
3. Install python-dotenv: `pip install python-dotenv`
4. Modify `app.py` to load the .env file (add at the top after imports):
   ```python
   from dotenv import load_dotenv
   load_dotenv()
   ```

### 6. Run the Application

With the virtual environment activated and HF_TOKEN set:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## How to Use

1. **Input Text**: In the sidebar, paste your text in the large text area
2. **Process Text**: Click the "🚀 Process Text" button to split and embed the text
3. **Ask Questions**: In the main area, type your question and click "🔍 Ask"
4. **Get Answers**: The AI will provide answers based only on your provided text

## Troubleshooting

### "Please enter some text to process"
- Make sure you've pasted text in the sidebar text area before clicking "Process Text"

### API Errors or Rate Limits
- Hugging Face free tier has rate limits. Wait a few minutes and try again
- Ensure your HF_TOKEN is correctly set (no extra spaces, quotes, etc.)

### Import Errors
- Make sure all packages from `requirements.txt` are installed
- Try reinstalling: `pip install -r requirements.txt --force-reinstall`

### Model Loading Issues
- Check your internet connection
- The first run may take longer as models download

### Token Issues
- Verify your Hugging Face token is valid and has "Read" permissions
- Make sure HF_TOKEN is set in the same terminal session you're running the app from

## Technical Details

- **Text Splitting**: RecursiveCharacterTextSplitter (chunk_size=500, overlap=50)
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Vector Store**: FAISS (in-memory)
- **LLM**: microsoft/phi-2 via Hugging Face Inference API
- **Framework**: Streamlit + LangChain

## License

This project uses only free and open-source components.