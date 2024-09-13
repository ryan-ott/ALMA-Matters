"""
Main script to run entire pipeline (from fine-tuning, compression to evaluation)
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Run the entire pipeline")
    parser.add_argument("--fine-tune", action="store_true", help="Fine-tune the model")
    parser.add_argument("--compress", action="store_true", help="Compress the model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    args = parser.parse_args()
    
    if args.fine_tune:
        # fine_tune_model()
        pass
    if args.compress:
        # compress_model()
        pass
    if args.evaluate:
        # evaluate_model()
        pass


if __name__ == "__main__":
    main()