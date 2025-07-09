import argparse
import os
import pandas as pd
from Bio import SeqIO

def main(args):
    fasta_files = [
        os.path.join(args.sequence_folder, f)
        for f in os.listdir(args.sequence_folder)
        if f.endswith('.fasta')
    ]

    records = []
    for fasta_file in fasta_files:
        for record in SeqIO.parse(fasta_file, "fasta"):
            processed_sequence = str(record.seq).replace("<null_1>", "").replace(".", "").replace("-", "")
            records.append({
                "id": record.id,
                "description": record.description,
                "sequence": processed_sequence
            })

    csv_path = os.path.join(args.sequence_folder, "all_sequences.csv")
    df = pd.DataFrame(records)
    df = df[["id", "sequence"]]
    df.to_csv(csv_path, index=False)

    fasta_path = os.path.join(args.sequence_folder, "all_sequences.fasta")
    with open(fasta_path, "w") as fasta_out:
        for record in records:
            fasta_out.write(f">{record['id']}\n{record['sequence']}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect sequences script")
    parser.add_argument('--sequence_folder', type=str, required=True, help='Path to the sequence folder')
    args = parser.parse_args()

    main(args)
