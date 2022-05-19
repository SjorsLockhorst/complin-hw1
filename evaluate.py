import os
import itertools
from typing import Tuple


import match_language as ml

def eval(
    model_path: str,
    test_path: str,
    verbose: bool = True
) -> Tuple[float, float]:
    """
    Evaluates how well a model is preforming against some test samples.

    Parameters
    ----------
    model_path : str
        Path to folder where models to test are stored.
    test_path : str
        Path to folder where training samles are stored.

    Returns
    -------
    Tuple[float, float]
        Amount of errors, amount of correct guesses.
    """
    # Dict that maps codes to full language names
    language_code_map = {
        "da": "Danish",
        "de": "German",
        "el": "Greek",
        "en": "English",
        "es": "Spanish",
        "fi": "Finnish",
        "fr": "French",
        "it": "Italian",
        "nl": "Dutch",
        "pt": "Portuguese",
        "sv": "Swedish"
    }

    # Create instance of language matcher
    lang_matcher = ml.LangMatcher(model_path)

    errors = 0
    for filename in os.listdir(test_path):

        # Find extention from test file
        ext = filename.split(".")[-1]

        # Find full name using language_code_map, this is the correct language
        expected_lang = language_code_map[ext]

        # Get full path by joining directory with filename
        test_file_path = os.path.join(test_path, filename)

        # Get only predicted langugage from LangMatcher, score is not relevant.
        predicted_lang, _ = lang_matcher.recognize(test_file_path)

        # Create a message to inform the outcome of the prediction
        message = f"{filename} {predicted_lang}"
        if expected_lang != predicted_lang:
            message += f" ERROR {expected_lang}"
            errors += 1

        if verbose:
            print(message)

    # Return the amount of errors, and the amount of correct predictions
    return errors, len(os.listdir(test_path)) - errors

def eval_all(verbose: bool = True):
    MODEL_DIR = "./models/"
    TEST_DIR = "./datafiles/test/"

    test_dirs = ["europarl-10", "europarl-30", "europarl-90"]
    model_dirs = os.listdir(MODEL_DIR)

    # Get full test paths and model paths
    test_paths = [os.path.join(TEST_DIR, test_dir) for test_dir in test_dirs]
    model_paths = [os.path.join(MODEL_DIR, model_dir) for model_dir in model_dirs]

    # Cartesion product to obtain all possible combinations of model and test dirs
    all_combinations = list(itertools.product(model_paths, test_paths))

    data = {}

    # Try each of these combinations
    for model_path, test_path in all_combinations:

        # Get n and limit from model directory name
        n_str, lim_str = os.path.basename(os.path.normpath(model_path)).split("-")
        n = int(n_str)
        lim = int(lim_str)

        # Get sentence length from test file name
        sent_length = int(os.path.basename(os.path.normpath(test_path)).split("-")[-1])

        # Generate a message about the performance of the model overall
        if n == 2:
            ngram_type = "Bigram"
        elif n == 3:
            ngram_type = "Trigram"
        else:
            ngram_type = f"{n}gram"

        error, correct = eval(model_path, test_path, verbose=False)
        message = f"\n {ngram_type} models with limit {lim} for {sent_length}-word "\
            f"sentences: {correct} correct, {error} incorrect. \n"

        percentage_correct = correct / (error + correct) * 100
        row = [n, lim, sent_length, percentage_correct]
        if ngram_type not in data:
            data[ngram_type] = [row]
        else:
            data[ngram_type].append(row)
        if verbose:
            print(message)
    return data



if __name__ == "__main__":
    eval_all()
