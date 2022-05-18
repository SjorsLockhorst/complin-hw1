import os
import itertools

import match_language as ml

def eval(model_path: str, test_path: str):
    """
    Evaluates how well a model is preforming
    """
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
    lang_matcher = ml.LangMatcher(model_path)

    errors = 0

    for filename in os.listdir(test_path):
        ext = filename.split(".")[1]
        expected_lang = language_code_map[ext]
        test_file_path = os.path.join(test_path, filename)
        predicted_lang, _ = lang_matcher.recognize(test_file_path)

        message = f"{filename} {predicted_lang}"
        if expected_lang != predicted_lang:
            message += f" ERROR {expected_lang}"
            errors += 1

    return errors, len(os.listdir(test_path)) - errors



if __name__ == "__main__":
    MODEL_DIR = "./models/"
    TEST_DIR = "./datafiles/test/"

    test_dirs = ["europarl-10", "europarl-30", "europarl-90"]
    model_dirs = os.listdir(MODEL_DIR)

    test_paths = [os.path.join(TEST_DIR, test_dir) for test_dir in test_dirs]
    model_paths = [os.path.join(MODEL_DIR, model_dir) for model_dir in model_dirs]

    all_combinations = list(itertools.product(model_paths, test_paths))

    for model_path, test_path in all_combinations:

        n_str, lim_str = model_path.split("/")[-1].split("-")
        n = int(n_str)
        lim = int(lim_str)
        sent_length = test_path.split("/")[-1].split("-")[-1]
        if n == 2:
            ngram_type = "Bigram"
        elif n == 3:
            ngram_type = "Trigram"
        else:
            ngram_type = f"{n}gram"

        error, correct = eval(model_path, test_path)
        print(f"{ngram_type} models with limit {lim} for {sent_length}-word sentences: {correct} correct, {error} incorrect.")
