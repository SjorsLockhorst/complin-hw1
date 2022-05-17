import os
import sys
from typing import List, Tuple, Union

import langdetect as ld

Score = Tuple[str, float]

class LangMatcher:

    def __init__(self, modeldir: str):
        """Initialise instance of this class from ngram tables in modeldir"""
        self.modeldir = modeldir
        self.language_map = {}

        files = os.listdir(modeldir)
        model_folder = os.path.basename(os.path.normpath(self.modeldir))

        n, limit = model_folder.split("-")
        self.n = int(n)
        self.limit = int(limit)

        for file in files:
            path = os.path.join(modeldir, file)
            table = ld.read_ngrams(path)
            self.language_map[file] = table


    def score(self, text: str, k_best: int = 1) -> Union[Score, List[Score]]:
        """Calculate the score of a text compared to models."""

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

        if k_best == 1:
            # Return only 1 score if that is requested
            return sorted_scores[0]
        else:

            # Return k_best scores
            return sorted_scores[:k_best]

    def recognize(self, filename: str, encoding: str = "utf-8"):
        """Open a file, read its contents and recognize which language it is."""
        with open(filename, "r", encoding=encoding) as f:
            return self.score(f.read())


if __name__ == "__main__":
    if len(sys.argv) < 3:
        model_dir = "models/3-200"
        test_files = ["datafiles/test/europarl-90/ep-00-02-03.nl"]
        print(f"No paths provided so using default {model_dir} and {test_files[0]}")
    else:
        model_dir = sys.argv[1]
        test_files = sys.argv[2:-1]

    lm = LangMatcher(model_dir)

    for test_file in test_files:
        lang, score = lm.recognize(test_file)
        print(f"{test_file} recognized as language {lang} with score {score}")

