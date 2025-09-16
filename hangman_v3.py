import math
import time
import json
import random
import pandas as pd
from collections import Counter, defaultdict
from create_corpus import read_words


# # # # # # # # # # # # # # # # # # Candidate Filtering # # # # # # # # # # # # # # # # # #

def filter_candidates(word_list: list, pattern: str, wrong_guesses: list) -> list:
    """
    Filter the word list to include only candidates matching the current pattern
    and excluding any letters already guessed incorrectly.

    Args:
        word_list (list): List of all possible words.
        pattern (str): Current pattern of the word (e.g., "_ p p _ e").
        wrong_guesses (list): List of letters guessed incorrectly.

    Returns:
        list: Words that are valid candidates for next guess.
    """

    def matches(word: str):
        if len(word) != len(pattern):
            return False
        for i, p in enumerate(pattern):
            if p != "_" and word[i] != p:  # known letters should match for valid candidates
                return False
            if p == "_" and word[i] in wrong_guesses:  # wrongly guessed letters shouldn't be present in candidates
                return False
        return True

    return [w for w in word_list if matches(w)]


# # # # # # # # # # # # # # # # # #  Build n-gram statistics  # # # # # # # # # # # # # # # # # #

def build_ngram_stats(words: list) -> tuple:
    """
    Build unigram, bigram, and trigram frequency statistics from a word list.

    Args:
        words (list): List of words.

    Returns:
        tuple: (unigram_probs, bigram, trigram, bigram_next_sum, bigram_prev_sum, trigram_lr_sum)
    """
    unigram = Counter()
    bigram = Counter()
    trigram = Counter()

    for word in words:
        for i, c in enumerate(word):
            unigram[c] += 1
            if i > 0:
                bigram[(word[i - 1], c)] += 1
            if i > 1:
                trigram[(word[i - 2], word[i - 1], c)] += 1

    total = sum(unigram.values()) if unigram else 1
    unigram_probs = {c: unigram[c] / total for c in unigram}

    # Precompute sums needed for conditional probabilities and smoothing
    bigram_next_sum = defaultdict(int)  # sum over next letters given previous
    bigram_prev_sum = defaultdict(int)  # sum over previous letters given next
    for (a, b), v in bigram.items():
        bigram_next_sum[a] += v
        bigram_prev_sum[b] += v

    # Trigram sum for smoothing
    trigram_lr_sum = defaultdict(int)
    for (a, b, c), v in trigram.items():
        trigram_lr_sum[(a, c)] += v

    return unigram_probs, bigram, trigram, bigram_next_sum, bigram_prev_sum, trigram_lr_sum


# # # # # # # # # # # # # # # # # # #  Entropy + frequency strategy  # # # # # # # # # # # # # # # # # #

def single_letter_freq_score(candidates: list, guessed: list):
    """
        Compute frequency score for each letter in candidate words not yet guessed.

        Args:
            candidates (list): List of candidate words.
            guessed (list): Letters already guessed.

        Returns:
            Counter: Frequency of each letter across candidates.
        """
    counter = Counter()
    for word in candidates:
        for letter in set(word):
            if letter not in guessed:
                counter[letter] += 1
    return counter


def entropy_score(candidates: list, guessed: list):
    """
    Compute entropy (information gain) for each unguessed letter.

    Args:
        candidates (list): List of candidate words.
        guessed (list): Letters already guessed.

    Returns:
        dict: Entropy score for each unguessed letter.
    """

    scores = {}
    alphabet = set("abcdefghijklmnopqrstuvwxyz") - guessed
    for letter in alphabet:
        partitions = defaultdict(int)
        for word in candidates:
            # partition by mask of positions where `letter` appears
            mask = tuple(i for i, c in enumerate(word) if c == letter)
            partitions[mask] += 1
        total = len(candidates)
        ent = 0.0
        for part_count in partitions.values():
            p = part_count / total
            ent -= p * math.log2(p)
        scores[letter] = ent
    return scores


def entropy_guess(candidates, guessed, weight_freq=0.5):
    """
    Combine entropy and letter frequency to choose next guess.

    Args:
        candidates (list): Candidate words.
        guessed (list): Already guessed letters.
        weight_freq (float): Weight for frequency contribution.

    Returns:
        str: Letter with the highest combined score.
    """
    if not candidates:
        return None
    ent_scores = entropy_score(candidates, guessed)
    freq_scores = single_letter_freq_score(candidates, guessed)

    combined = {}
    for letter in ent_scores:
        combined[letter] = ent_scores[letter] + weight_freq * freq_scores.get(letter, 0)

    # choose the letter with the highest combined score
    return max(combined, key=combined.get)


# # # # # # # # # # # # # # # # # # Bayesian strategy using uni/bi/tri-grams  # # # # # # # # # # # # # # # # # #

def bayesian_guess(pattern, guessed, unigram_probs, bigram, trigram, bigram_next_sum,
                   bigram_prev_sum, trigram_lr_sum,
                   alpha=1.0):
    """
    Bayesian guess using n-gram statistics with Laplace smoothing.

    Args:
        pattern (list): Current word pattern.
        guessed (set): Letters already guessed.
        unigram_probs, bigram, trigram: N-gram statistics.
        bigram_next_sum, bigram_prev_sum, trigram_lr_sum: Precomputed sums.
        alpha (float): Laplace smoothing parameter.

    Returns:
        str: Letter with highest posterior score.
    """
    V = 26  # alphabet size for smoothing
    letters = set("abcdefghijklmnopqrstuvwxyz") - guessed
    if not letters:
        return None

    scores = defaultdict(float)

    for pos, p in enumerate(pattern):
        if p != "_":
            continue
        left = pattern[pos - 1] if pos > 0 else None
        right = pattern[pos + 1] if pos < len(pattern) - 1 else None

        for X in letters:
            prior = unigram_probs.get(X, 1e-12)
            likelihood = 1.0

            # P(left | X) ~ count(left,X) / sum_a count(a,X)
            if left and left != "_":
                numer = bigram.get((left, X), 0) + alpha
                denom = bigram_next_sum.get(left, 0) + alpha * V
                likelihood *= numer / denom

            # P(right | X) ~ count(X,right) / sum_b count(X,b)
            if right and right != "_":
                numer = bigram.get((X, right), 0) + alpha
                denom = bigram_next_sum.get(X, 0) + alpha * V
                likelihood *= numer / denom

            # trigram factor (how likely is X to appear between left and right)
            if left and right and left != "_" and right != "_":
                numer = trigram.get((left, X, right), 0) + alpha
                denom = trigram_lr_sum.get((left, right), 0) + alpha * V
                likelihood *= numer / denom

            score = prior * likelihood
            scores[X] += score

    if not scores:
        return None
    return max(scores, key=scores.get)


# # # # # # # # # # # # # # # # # # Unigram fallback  # # # # # # # # # # # # # # # # # #

def fallback_unigram(unigram_probs, guessed: list):
    """
    Fallback guess based on unigram frequencies.

    Args:
        unigram_probs (dict): Unigram probabilities.
        guessed (list): Letters already guessed.

    Returns:
        str: Most frequent unguessed letter.
    """
    candidates = [(p, i) for i, p in unigram_probs.items() if i not in guessed]
    if candidates:
        return max(candidates)[1]
    # last resort: any unguessed letter
    remaining = [i for i in "abcdefghijklmnopqrstuvwxyz" if i not in guessed]
    return random.choice(remaining) if remaining else None


# # # # # # # # # # # # # # # # # # Brute Force # # # # # # # # # # # # # # # # # #

def random_guess(guessed: list):
    """
    Random guess from unguessed letters.

    Args:
        guessed (list): Letters already guessed.

    Returns:
        str: Random unguessed letter.
    """
    remaining = [i for i in "abcdefghijklmnopqrstuvwxyz" if i not in guessed]
    return random.choice(remaining) if remaining else None


# # # # # # # # # # # # # # # # # # Game Loop with Strategy Config # # # # # # # # # # # # # # # # # #

def play_hangman(secret_word: str, corpus_words: list, max_attempts=6, strategy="entropy", weight_freq=0.5):
    """
        Play a single game of Hangman with different strategies.

        Args:
            secret_word (str): Word to guess.
            corpus_words (list): List of valid words for candidate filtering.
            max_attempts (int): Maximum incorrect guesses allowed.
            strategy (str): Strategy to use ("entropy", "bayesian", "unigram", "random", "combo").
            weight_freq (float): Weight for frequency in entropy-based strategies.

        Returns:
            tuple: ("Win"/"Loss", remaining_attempts)
        """
    secret_word = secret_word.lower()
    word_list = corpus_words

    guessed = set()
    wrong_guesses = set()
    pattern = ["_" for _ in secret_word]
    attempts_left = max_attempts

    while attempts_left > 0 and "_" in pattern:
        candidates = filter_candidates(word_list, pattern, wrong_guesses)
        guess = None
        if strategy == "entropy":
            if candidates:
                guess = entropy_guess(candidates, guessed, weight_freq=weight_freq)
            else:
                guess = fallback_unigram(unigram_probs, guessed)

        elif strategy == "bayesian":
            # bayesian uses pattern+ngram stats directly (works even when candidates exist)
            guess = bayesian_guess(pattern, guessed, unigram_probs, bigram, trigram, bigram_next_sum, bigram_prev_sum,
                                   trigram_lr_sum)
            if guess is None:
                guess = fallback_unigram(unigram_probs, guessed)

        elif strategy == "random":
            guess = random_guess(guessed)

        elif strategy == "unigram":
            guess = fallback_unigram(unigram_probs, guessed)

        elif strategy == "combo":
            if candidates:
                guess = entropy_guess(candidates, guessed, weight_freq=weight_freq)
            else:
                guess = bayesian_guess(pattern, guessed, unigram_probs, bigram, trigram,
                                       bigram_next_sum, bigram_prev_sum, trigram_lr_sum)
                if guess is None:
                    guess = fallback_unigram(unigram_probs, guessed)

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        if not guess:
            # should not normally happen, but handle defensively
            remaining = [l for l in "abcdefghijklmnopqrstuvwxyz" if l not in guessed]
            if not remaining:
                break
            guess = random.choice(remaining)

        guessed.add(guess)

        if guess in secret_word:
            for i, c in enumerate(secret_word):
                if c == guess:
                    pattern[i] = guess
            # feedback (can be commented out in batch mode)
             #print(f"Correct! Guessed '{guess}' → {''.join(pattern)}")
        else:
            wrong_guesses.add(guess)
            attempts_left -= 1
            # print(f"Wrong! Guessed '{guess}' → {''.join(pattern)} | Attempts left: {attempts_left}")

    if "_" not in pattern:
        print(f"Bot wins! Word was '{secret_word}'.")
        return "Win", attempts_left
    else:
        print(f"Bot loses! Word was '{secret_word}'.")
        return "Loss", 0


# # # # # # # # # # # # # # # # # # #  Data Analysis # # # # # # # # # # # # # # # # # # # # # # #

def data_analysis(json_filename: str, strat_names: list):
    """
    Analyze Hangman simulation statistics stored in JSON files.

    Args:
        json_filename (str): Path to JSON file containing results.
        strat_names (list): List of strategy names to analyze.
    """
    with open(json_filename, "r") as f:
        stats = json.load(f)
    flat_data = {}
    for l1_key, l2_dict in stats.items():
        for l2_key, l3_dict in l2_dict.items():
            if l2_key not in flat_data.keys():
                flat_data[l2_key] = {}
            flat_data[l2_key].update({l1_key + k: v for k, v in l3_dict.items()})

    # Convert to DataFrame
    df = pd.DataFrame.from_dict(flat_data, orient="index")
    for strat in strat_names:
        win_rate = (df[strat + "status"] == "Win").sum() / len(df)

        mask = df[strat + "status"] == "Win"
        mean_attempts = pd.to_numeric(df.loc[mask, strat + "attempts_left"], errors="coerce").mean()

        # total_time = pd.to_numeric(df[strat + 'time'], errors="coerce").sum()
        # total_tries = 6 * len(df) - pd.to_numeric(df[strat + 'attempts_left'], errors="coerce").sum()
        # avg_time = total_time / total_tries

        print(f"{strat.upper()} -- Win rate: {win_rate:.2f} | "
              f"Attempts left if win: {mean_attempts:.2f} ")
        # f"Avg time for a guess: {avg_time:.2f}")


if __name__ == "__main__":
    secret_count = 100
    strategy_names = ["combo", "entropy", "bayesian", "unigram", "random"]
    test_file_names = ["corpus/test/secrets.txt", "corpus/test/blindsecrets.txt"]
    corpus_dict = {"english": ["corpus/word_corpus.txt"],
                   "airline": ["corpus/airline_corpus.txt"],
                   "both": ["corpus/word_corpus.txt", "corpus/airline_corpus.txt"]}

    STRATEGIES = {k: {} for k in strategy_names}
    for corpus, corpus_list in corpus_dict.items():
        words = []

        for file_name in corpus_list:
            words += read_words(file_name)
            print(len(words))
        if not words:
            raise RuntimeError("No words available?!")
        # Build n-gram stats for Bayesian strategy

        unigram_probs, bigram, trigram, bigram_next_sum, bigram_prev_sum, trigram_lr_sum = build_ngram_stats(words)

        # Run Hangman simulations on each secret word list
        for test_file_name in test_file_names:
            secret_list = read_words(test_file_name)
            for secret in secret_list[:secret_count]:
                for strategy in STRATEGIES.keys():
                    start = time.time()  # record start time
                    status, attempts_left = play_hangman(secret, corpus_words=words, strategy=strategy)
                    end = time.time()
                    STRATEGIES[strategy][secret] = {"status": status, "attempts_left": attempts_left,
                                                    "time": end - start}

            # Save results
            with open(f"corpus/test/stats_on_"
                      f"{test_file_name.replace('.txt', '').replace('corpus/test/', '')}"
                      f"_using_{corpus}v2.json",
                      "w") as json_file:
                json.dump(STRATEGIES, json_file, indent=4)

    # Analyze results
    for corpus, corpus_list in corpus_dict.items():
        print("\n" + corpus + "-" * 30)
        for test_file_name in test_file_names:
            print(f'{"<>" * 10} {corpus} on {test_file_name.replace("txt.txt", "")}  {"<>" * 10}')
            data_analysis(f"corpus/test/stats_on_"
                          f"{test_file_name.replace('.txt', '').replace('corpus/test/', '')}"
                          f"_using_{corpus}v2.json",
                          strat_names=strategy_names)
