import sys
import os
import pandas as pd

def main(csv_path, folder_path):
    df = pd.read_csv(csv_path)
    lengths = df['length'].tolist()

    missing_lengths = []
    for length in lengths:
        filename = f"iter_500_L_{length}.fasta"
        filepath = os.path.join(folder_path, filename)
        if not os.path.isfile(filepath):
            missing_lengths.append(length)

    print("Number of missing files:", len(missing_lengths))
    print("Missing lengths:", missing_lengths)
    if missing_lengths:
        print("\nMissing lengths with counts and total_tokens:")
        for length in missing_lengths:
            count = df.loc[df['length'] == length, 'count'].values[0]
            total_tokens = length * count
            print(f"Length: {length}, Count: {count}, Total tokens: {total_tokens}")

    if missing_lengths:
        missing_df = df[df['length'].isin(missing_lengths)][['length', 'count']]
        base, ext = os.path.splitext(csv_path)
        new_csv_path = f"{base}_missing{ext}"
        missing_df.to_csv(new_csv_path, index=False)
        print(f"Missing lengths saved to: {new_csv_path}")
    else:
        print("No missing files. No CSV created.")
    


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <csv_path> <folder_path>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
