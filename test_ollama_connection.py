
import sys
from ollama import chat

MODEL_NAME = "llama3.2:1b"


def main():
    print(f"Testing Ollama with model: {MODEL_NAME}")

    try:
        response = chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an assistant helping with "
                        "exploratory data analysis."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Explain how to compare average account "
                        "balance between customers with and "
                        "without housing loans. Do not invent "
                        "numerical results."
                    ),
                },
            ],
        )

        print("\nOllama response:\n")
        print(response.message.content)
        print("\nConnection test passed.")

    except Exception as exc:
        print(f"\nConnection test failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
