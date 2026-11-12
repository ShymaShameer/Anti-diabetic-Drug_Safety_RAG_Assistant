import os
import nltk

# 1. Force a secure local NLTK directory before any LlamaIndex imports happen
safe_nltk_dir = os.path.expanduser("~/nltk_data")
os.makedirs(safe_nltk_dir, exist_ok=True)
nltk.data.path.insert(0, safe_nltk_dir)

# 2. Download stopwords to the safe directory if missing
try:
    nltk.data.find("corpora/stopwords", paths=[safe_nltk_dir])
except LookupError:
    nltk.download("stopwords", download_dir=safe_nltk_dir)

# 3. Now it's safe to import your engine/chat loop
from src.engine import main_chat_loop


def main() -> None:
    print("--- 🤖 Main Application Starting ---")
    # Start the main chat session
    main_chat_loop()


if __name__ == "__main__":
    main()
    