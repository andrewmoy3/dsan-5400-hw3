import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inverted index")
    parser.add_argument("-f", required=True, help="Path to XML collection")
    parser.add_argument("-t", required=True, help="Tokenizer to use")
    parser.add_argument("-s", action="store_true", help="Apply stemming")
    args = parser.parse_args()
