"""Tech Explainer: ask a technical question, get a streamed explanation
from a local Ollama model."""

import argparse
import os
import sys

# DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
from dotenv import load_dotenv
from openai import APIConnectionError, NotFoundError, OpenAI, OpenAIError

SYSTEM_PROMPT = """You are a patient, expert technical tutor.
Explain the user's technical question clearly: say what it does, then why it
works that way, and point out any gotchas. Use short examples where helpful."""


# Prepare client creation with ollama
def make_client() -> OpenAI:
    # Ollama exposes an OpenAI-compatible API under /v1; the key is required but ignored
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    return OpenAI(base_url=base_url, api_key="ollama")



# Prepare client creation with gemini
# def make_client() -> OpenAI:
#     google_api_key = os.getenv("GEMINI_API_KEY")
#     base_url = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
#     return OpenAI(base_url=base_url, api_key=google_api_key)


def stream_explanation(question: str, model: str = DEFAULT_MODEL):
    """Yield the explanation piece by piece as Ollama/gemini generates it."""
    stream = make_client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Explain a technical question using a local Ollama/gemini model.")
    parser.add_argument("question", nargs="?", help="The question. Omit to read from stdin.")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL,
                        help="Ollama/gemini model to use (default: llama3.2).")
    args = parser.parse_args()

    question = args.question or sys.stdin.read().strip()
    if not question:
        parser.error("Please provide a question.")

    try:
        for piece in stream_explanation(question, args.model):
            print(piece, end="", flush=True)
        print()
    except APIConnectionError:
        sys.exit("Could not reach Ollama/Gemini. Is it running? Start it with: ollama serve")
    except NotFoundError:
        sys.exit(f"Model '{args.model}' not found. Pull it with: ollama pull {args.model}")
    except OpenAIError as e:
        sys.exit(f"Ollama/Gemini error: {e}")
    except KeyboardInterrupt:
        sys.exit(130)
