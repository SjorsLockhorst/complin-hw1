import re
from collections import Counter
from typing import List,  Sequence, Any, Dict

Table = Dict[str, int]


def prepare(text: str) -> List[str]:
    """
    Tokinzes a text into words.

    Parameters
    ----------
    text: str
        The text to tokenize.

    Returns
    -------
    List[str]
        List of words from text.
    """
    subbed = re.sub(r'[!?",.()<>]', r' ', text)
    return subbed.split()


def ngrams(seq: Sequence[Any], n: int = 3) -> List[Any]:
    """
    Creates ngrams.

    Parameters
    ----------
    seq: Sequence[Any]
        Any sequence to create ngrams from.
    n : int, default=3

    Returns
    -------
    List[Any]
        List containing ngrams.
    """
    ngrams = []  # Store all ngrams here
    i = 0

    while i <= len(seq) - n:  # Loop untill last ngram
        ngram = seq[i:i + n]  # Slice out an ngram
        ngrams.append(ngram)  # Add it to the list
        i += 1
    return ngrams


def ngram_table(text: str, n: int = 3, limit: int = 0) -> Table:
    """
    Create an ngram table.

    Tokenizes a text on words, and generates ngrams for the characters of each word.
    Then, returns the `limit` amount of most common ngrams from text.

    Parameters
    ----------
    text : str
        Text to tokenize and generate ngrams from.
    n : int, default=3
        Size of ngrams to use.
    limit : int, default=0
        Limit how many most common ngrams get returned, when 0, all frequencies are
        returned.


    Returns
    -------
    Table
        A dictionary containing str keys, the ngrams, and int values, the frequencies.
    """
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
    """
    Reads a ngram frequency table from a file.

    Parameters
    ----------
    filename : str
        Path to file to read ngrams from.

    Returns
    -------
    Table
        A dictionary containing str keys, the ngrams, and int values, the frequencies.
    """
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

    Parameters
    ----------
    table : Table
        A dictionary containing str keys, the ngrams, and int values, the frequencies.
    filename : str
        File path to write ngram table to.
    """

    sorted_table = dict(
        sorted(table.items(), key=lambda x: x[1], reverse=True))

    # Put the lines in the correct format
    lines = [f"{freq} {word}\n" for word, freq in sorted_table.items()]

    # Write lines to a file
    with open(filename, "w", encoding="utf-8") as f:
        f.writelines(lines)


def cosine_similarity(known: Table, unknown: Table) -> float:
    """
    Calculates cosine similarity between ngram tables.

    Parameters
    ----------
    known : Table
        Trained ngram table.
    unknown : Table
        Newly observed ngram table.

    Returns
    -------
    float
        The cosine similarity, some float between 0 and 1.
    """

    def _magnitude(x: List[int]) -> float:
        """
        Calculate the magnitute of a vector.

        Parameters
        ----------
        x : List[int]

        Returns
        -------
        float
            Magnitude of vector x.
        """
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
