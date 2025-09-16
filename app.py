from hangman_v3 import *
from flask import Flask, request, jsonify


def get_best_guess(current_word_state: str, guessed_letters: list) -> str:
    """
    Determine the next best guess for the Hangman game.

    This function combines candidate filtering, entropy scoring, and Bayesian
    n-gram models to select the most promising next guess.

    Args:
        current_word_state (str): The current state of the word with underscores
                                representing unknown letters (e.g., "_ p p _ e").
        guessed_letters (list): List of letters that have already been guessed.

    Returns:
        str: The next guessed letter.
    """
    guess = None
    guessed = set(guessed_letters)
    print(f"GUESSED: {guessed}")
    pattern = current_word_state.replace(" ", "")
    print(f"PATTERN: {pattern}")
    wrong_guesses = set(guessed) - set(pattern) - set('_')
    print(f"WRONG GUESSES: {wrong_guesses}")
    candidates = filter_candidates(words, pattern, wrong_guesses)
    if candidates:
        guess = entropy_guess(candidates, guessed, weight_freq=0.5)
    else:
        guess = bayesian_guess(pattern, guessed, unigram_probs, bigram, trigram,
                               bigram_next_sum, bigram_prev_sum, trigram_lr_sum)
        if guess is None:
            guess = fallback_unigram(unigram_probs, guessed)

    return guess


# --- Flask API ---
app = Flask(__name__)


@app.route('/play', methods=['POST'])
def play():
    """
    API endpoint to provide the next guess in the word game.

    Expects a JSON payload with:
        - current_word_state (str): Current state of the word (e.g., "_ p p _ e").
        - guessed_letters (list): Letters that have already been guessed.
        - guesses_remaining (int): Remaining number of incorrect guesses allowed.

    Returns:
        JSON: {"nextGuess": <letter>}
    """
    # Parse JSON data from request
    data = request.get_json()
    current_word_state = data.get("currentWordState", "")
    guessed_letters = data.get("guessedLetters", [])
    guesses_remaining = data.get("guessesRemaining", 6)

    # Call guessing logic
    next_guess = get_best_guess(current_word_state, guessed_letters)

    # Return JSON response
    return jsonify({"nextGuess": next_guess})


if __name__ == '__main__':
    # Run Flask app in debug mode (auto-reload and detailed errors)

    corpus_list = ["corpus/word_corpus.txt", "corpus/airline_corpus.txt"]
    words = []
    for file_name in corpus_list:
        words += read_words(file_name)
        print(f"{len(words)} words in the corpus")

    if not words:
        raise RuntimeError("No words available?!")

    # Build n-gram stats for Bayesian strategy
    unigram_probs, bigram, trigram, bigram_next_sum, bigram_prev_sum, trigram_lr_sum = build_ngram_stats(words)
    app.run(debug=True)
