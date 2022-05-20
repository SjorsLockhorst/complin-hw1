import os
import re

import langdetect as ld


def make_profiles(datafolder: str, n: int, limit: int):
    """
    Make language profiles from training samples.

    Load all files containing natural language from a datafolder, then create ngram
    tables with `n` as the ngram size and `limit` as the size of each table.
    Then write these tables out to a folder.

    Parameters
    ----------
    datafolder : str
        Path to folder where training sample files are stored.
    n : int
        Size of ngrams to use.
    limit : int
        Amount of most common ngrams to put in each final ngram table.
    """

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
        table = ld.ngram_table(text, n=n, limit=limit)

        clean_out_file_name = re.sub(r"-.*$", "", file_name)
        # Create output file based on out_dir and input file name
        out_file_path = os.path.join(out_dir, clean_out_file_name)

        # Write the ngrams to this file
        ld.write_ngrams(table, out_file_path)


if __name__ == "__main__":
    # Create 2 sets of models, bigrams and trigrams, both with 200 limit
    make_profiles("./datafiles/training", 2, 200)
    make_profiles("./datafiles/training", 3, 200)
