import argparse
import sys

__version__ = "0.1.0"

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--version", action="store_true", help="show version")
    args = parser.parse_args()

    if args.version:
        # print(f"args.version: {args.version}")
        print(__version__)
        sys.exit(0)

    print("Hello from rag-ollama-local!")
