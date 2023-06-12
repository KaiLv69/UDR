import argparse
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_files", nargs="+")
    parser.add_argument("--output_file")
    args = parser.parse_args()
    print(args.input_files)
    print(len(args.input_files))
    merged = []
    for input_file in args.input_files:
        with open(input_file, 'r') as f:
            data = json.load(f)
        merged.extend(data)

    with open(args.output_file, "w") as f:
        json.dump(merged, f)
