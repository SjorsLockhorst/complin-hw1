import re
from collections import Counter
from typing import List,  Sequence, Any, Dict

Table = Dict[str, int]

def prepare(text: str) -> List[str]:
    """
    Tokinzes a text into words.
    """
    subbed = re.sub(r'[!?",.()<>]', r' ', text)
    return subbed.split()

def ngrams(seq: Sequence[Any], n: int = 3) -> List[Any]:
    """
    Creates ngrams from a Sequence.
    """
    ngrams = []  # Store all ngrams here
    i = 0

    while i <= len(seq) - n:  # Loop untill last ngram
        ngram = seq[i:i + n]  # Slice out an ngram
        ngrams.append(ngram)  # Add it to the list
        i += 1
    return ngrams

def ngram_table(text: str, n: int = 3, limit: int = 0) -> dict:
    """Frequency counts of ngrams in text."""

    tokens = prepare(text)  # Clean and tokenize the text
    surrounded = [
        f"<{token}>" for token in tokens  # Surround tokens with <token>
    ]
    all_ngrams = []  # Store all found ngrams
    for token in surrounded:
        all_ngrams += ngrams(token, n)  # Get ngrams from each surrounded token

    table = Counter(all_ngrams)  # Create table to count ngrams

    if limit > 0:  # The amount limit most common ngrams
        return dict(table.most_common(limit))
    return dict(table)


def read_ngrams(filename: str) -> Table:
    """Read ngram frequency table from a file."""
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()
    table = {}
    for line in lines:
        freq, word = line.strip().split(" ")
        table[word] = int(freq)
    return table



def write_ngrams(table: Table, filename: str):
    """
    Writes ngram frequency table to a file.
    """
    # Sort table by values
    sorted_table = dict(sorted(table.items(), key=lambda x: x[1], reverse=True))

    # Put the lines in the correct format
    lines = [f"{freq} {word}\n" for word, freq in sorted_table.items()]

    # Write lines to a file
    with open(filename, "w", encoding="utf-8") as f:
        f.writelines(lines)

def cosine_similarity(known: Table, unknown: Table) -> float:
    """
    Calculates cosine similarity
    """

    def _magnitude(x: List[int]):
        """Calculate the magnitute of a vector."""
        mag = 0
        for x_i in x:
            mag += x_i**2
        return (mag)**0.5

    # Convert table keys (thus the words) to set
    known_words = set(known)
    unknown_words = set(unknown)

    # Find which words are overlapping
    overlapping = known_words & unknown_words

    # Calculate dot product between overlapping words
    dot_product = 0
    for key in overlapping:
        dot_product += known[key] * unknown[key]

    # Create lists of known values
    known_values = list(known.values())
    unknown_values = list(unknown.values())

    # Calculate final result
    cosine = (
        dot_product / (
            _magnitude(known_values) * _magnitude(unknown_values)
        )
    )
    return cosine
