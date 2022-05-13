import os
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
            return self.score(f.read())[0]
