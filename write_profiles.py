import os
import re

from langdetect import (
    ngram_table,
    write_ngrams
)


def make_profiles(datafolder: str, n: int, limit: int):
    """Make profiles of ngrams found in path."""

    MODEL_DIR = "./models"  # Top level directory to store models

    # Directory within models to store profiles, based on input
    out_dir = os.path.join(MODEL_DIR, f"{n}-{limit}")

    # Create directories in case they don't exist yet
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # Get all files at the path
    read_file_names = [file_path for file_path in os.listdir(datafolder)]

    for file_name in read_file_names:

        # Get encoding from filename, use utf-8 as standard
        encoding = "utf-8"
        if re.search(r"^.*-Latin1$", file_name):
            encoding = "latin1"

        # Get full path from adding filename to directory
        full_path = os.path.join(datafolder, file_name)

        # Open file at this path and read the contents as a string
        with open(full_path, "r", encoding=encoding) as f:
            text = f.read()

        # Use function from langdetect to read ngrams into table
        table = ngram_table(text, n=n, limit=limit)

        clean_out_file_name = re.sub(r"-.*$", "", file_name)
        # Create output file based on out_dir and input file name
        out_file_path = os.path.join(out_dir, clean_out_file_name)

        # Write the ngrams to this file
        write_ngrams(table, out_file_path)


if __name__ == "__main__":
    make_profiles("./datafiles/training", 3, 200)
    make_profiles("./datafiles/training", 2, 200)

