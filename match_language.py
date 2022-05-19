import os
import sys
from typing import List, Tuple, Dict

import langdetect as ld

Score = Tuple[str, float]

class LangMatcher:
    """
    Class that can read a file and find a matching language.

    Attributes
    ----------
    modeldir : str
        Directory where models to base prediction on are stored.
    language_map : Dict[str, ld.Table]
        Dictionary that maps each language to it's trained ngram table.
    n : int
        The size of each ngram as used in selected models.
    limit : int
        The limit of each ngram table as used in selected models.
    """

    def __init__(self, modeldir: str):
        self.modeldir = modeldir
        self.language_map: Dict[str, ld.Table] = {}

        files = os.listdir(modeldir)

        # Get folder name by extracting `deepest` directory in path
        model_folder = os.path.basename(os.path.normpath(self.modeldir))

        # Extract used n and limit from folder name
        n, limit = model_folder.split("-")

        # Cast from str to int
        self.n = int(n)
        self.limit = int(limit)

        # Load each ngram in models folder and store it in language map
        for file in files:
            path = os.path.join(modeldir, file)
            table = ld.read_ngrams(path)
            self.language_map[file] = table


    def score(self, text: str, k_best: int = 1) -> List[Score]:
        """
        Get scores of text compared with loaded models.

        Calculates the score of the current text in comparison to all loaded
        language models. Return the k best scores.

        Parameters
        ----------
        text : str
            The text to score.
        k_best : int, default=1
            How many of the best scores to return.

        Returns
        -------
        List[Score]
            The k_best scores for this text compared to language models.
        """

        # Load in unkown text
        unknown = ld.ngram_table(text, self.n, self.limit)

        # List to store scores
        scores: List[Score] = []

        # Loop over known texts
        for language_name, known in self.language_map.items():

            # Calculate cosine similarity between known and unknown text
            score = ld.cosine_similarity(known, unknown)

            # Add name and score to results
            scores.append((language_name, score))

        # Sort scores by in descending order
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

        # Return k_best scores
        return sorted_scores[:k_best]

    def recognize(self, filename: str, encoding: str = "utf-8") -> Score:
        """
        Open a file and recognize which language it is most likely to be.

        Parameters
        ----------
        filename : str
            File path to load language from.
        encoding : str, default='utf-8'
            Encoding to use when opening the file at `filename`.

        Returns
        -------
        Score
            The best score for text in the file when compared to all language models.
        """
        with open(filename, "r", encoding=encoding) as f:
            return self.score(f.read())[0]


if __name__ == "__main__":

    # When not enough arguments, use default files
    if len(sys.argv) < 3:
        model_dir = "models/3-200"
        test_files = ["datafiles/test/europarl-90/ep-00-02-02.fi"]
        print(f"No paths provided so using default {model_dir} and {test_files[0]}")
    else:
        model_dir = sys.argv[1]
        test_files = sys.argv[2:]

    # Use arguments to load test file(s) and recognize their languages
    lm = LangMatcher(model_dir)
    for test_file in test_files:
        lang, score = lm.recognize(test_file)
        print(f"{test_file} recognized as language {lang} with score {score}")
