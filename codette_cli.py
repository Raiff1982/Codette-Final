
import argparse
from codette.codette_core import AICore

def main():
    parser = argparse.ArgumentParser(description='Codette CLI')
    parser.add_argument('prompt', type=str, help='User prompt for Codette to process')
    args = parser.parse_args()

    codette = AICore()
    result = codette.process_input(args.prompt)
    print("\n=== Codette CLI Output ===\n")
    print(result)

if __name__ == '__main__':
    main()
