import pandas as pd
import argparse

def main(data_path):
    print(f"Reading data from: {data_path}")
    df = pd.read_csv(data_path)
    print("Data preview:")
    print(df.head())
    print(f"Dataset shape: {df.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()
    main(args.data_path)
