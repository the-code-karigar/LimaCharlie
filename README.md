
# Lima Charlie: Loud & Clear Strategies for Hangman 

An Aviation-Themed Hangman Solver that uses statistical models, entropy-based heuristics, and Bayesian inference to guess letters intelligently.  

The project includes:
- A Flask REST API (`/play`) to serve predictions.
- Multiple guessing strategies.
- Tools to scrape and build corpora (general English + domain-specific).
- A data analysis module for evaluating solver performance.

---

## Table of Contents
1. [Overview](#overview)  
2. [Approach & Strategies](#approach--strategies)
3. [Corpus Creation](#Corpus--reation)
4. [Project Structure](#project-structure)  
5. [Installation & Setup](#installation--setup)  
   - [Clone the Repository](#1-clone-the-repository)  
   - [Setup Virtual Environment](#2-create--activate-virtual-environment)  
   - [Install Dependencies](#3-install-dependencies)  
   - [Download NLTK Corpus](#4-download-nltk-corpus)  
6. [Running the API](#running-the-api)  
7. [Testing the Client](#testing-the-client)  
8. [Data Analysis](#data-analysis)  
9. [Libraries Used](#libraries-used)  
10. [Future Improvements](#future-improvements)  
11. [License](#license)  
12. [Author](#author)  

---

## Overview

This project simulates and automates Hangman gameplay.  
The solver uses probabilistic models trained on corpora to intelligently select the next guess.  

It supports:
- Running simulations against custom test sets.  
- Exposing results as a REST API for external clients.  
- Comparing strategies via data analytics.  

Use cases include:
- Automated Hangman gameplay.  
- Evaluating different strategies.  
- Integrating a Hangman bot into games or educational tools.  

---

## Approach & Strategies

The solver supports multiple strategies with varying levels of sophistication:

### 1. Entropy Strategy
- Uses information gain to choose the letter that reduces uncertainty the most.  
- Incorporates letter frequency weighting to balance informativeness and likelihood.  
- Effective when candidate words are available.

### 2. Bayesian N-gram Strategy
- Builds unigram, bigram, and trigram statistics from the chosen corpus.  
- Applies Laplace smoothing for unseen cases.  
- Computes probabilities of a letter fitting given context (previous and next letters, as per availability).  
- Effective for structured domains such as airline vocabulary.

### 3. Unigram Strategy
- Picks the most frequent unused letter from the corpus.  
- Serves as a simple fallback.

### 4. Random Strategy
- Selects letters randomly.  
- Used as a baseline for comparison.

### 5. Combo Strategy
- Hybrid method:  
  - Uses Entropy when valid candidate words exist.  
  - Falls back to Bayesian (or unigram if needed) when no candidates remain.

---

## Corpus Creation

A strong solver requires a **representative word corpus**.  
This project builds two main corpora:

1. **General English Corpus**  
   - Extracted from the **NLTK words corpus**.  
   - Provides coverage for everyday vocabulary.

2. **Aviation Corpus**  
   - Collected by scraping **Wikipedia articles** and other sources related to aviation.  
   - Captures domain-specific words such as aircraft models, airline names, and aviation terminology.  
   - Process:  
     - Fetch raw HTML using `requests`.  
     - Parse article text with `BeautifulSoup`.  
     - Tokenize into words, filter for alphabetic terms, normalize case.  
     - Deduplicate and store in plain text.  

### File Outputs
- `corpus/word_corpus.txt` → General English words.  
- `corpus/airline_corpus.txt` → Aviation words.  
- `corpus/test/` → Contains test word lists (`secrets.txt`, etc.) and solver performance logs.  

The combined corpus can be used for broader coverage:  
```python
corpus_list = ["corpus/word_corpus.txt", "corpus/airline_corpus.txt"]
```

---

## Project Structure

```

├── app.py                # Flask API exposing the solver
├── client.py             # Example client querying the API
├── hangman\_v3.py         # Core Hangman solver & strategies
├── create\_corpus.py      # Scripts for building corpora (NLTK + Wikipedia)
├── corpus/               # Generated corpora & test sets
│   ├── word\_corpus.txt
│   ├── airline\_corpus.txt
│   └── test/
│       ├── secrets.txt
│       ├── blindsecrets.txt
│       ├── gitsecrets.txt
│       └── stats\_\*.json
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation

````

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone <REPOSITORY_URL>
cd <REPOSITORY_NAME>
````

### 2. Create & Activate Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLTK Corpus

The solver uses the English words corpus from NLTK:

```python
import nltk
nltk.download('words')
```

---

## Running the API

Start the Flask server:

```bash
python app.py
```

By default, the server runs at:

```
http://127.0.0.1:5000
```

### Available Endpoints

#### `POST /play`

Request:

```json
{
  "currentWordState": "_ _ e _ a n",
  "guessedLetters": ["e", "a", "n"],
  "guessesRemaining": 4
}
```

Response:

```json
{
  "nextGuess": "t"
}
```

---

## Testing the Client

An example client (`client.py`) is included.
Run it while the server is active:

```bash
python client.py
```

This sends a test game state to the API and prints the solver’s guess.

---

## Data Analysis

The project records statistics about solver performance:

* Win rate (percentage of games solved)
* Average attempts left on successful games
* Runtime per guess

JSON logs are stored under `corpus/test/`.
Use the `data_analysis()` function in `hangman_v3.py` to summarize results.

Example:

```bash
python hangman_v3.py
```

Sample Output:

```
BAYESIAN -- Win rate: 0.82 | Attempts left if win: 2.45
COMBO    -- Win rate: 0.87 | Attempts left if win: 2.67
```

---

## Libraries Used

### Standard Library

* `math`, `time`, `json`, `random`
* `collections` (Counter, defaultdict)
* `re`, `os`

### Third-Party Libraries

* **Flask** (`flask`, `Werkzeug`, `Jinja2`, `itsdangerous`, `click`, `blinker`) → REST API
* **requests** → API client
* **nltk** → Natural language corpus (`words`)
* **beautifulsoup4 / bs4** → Wikipedia scraping
* **pandas** → Data analysis
* **numpy** → Array operations
* **tqdm** → Progress tracking

All dependencies are pinned in [`requirements.txt`](requirements.txt) for reproducibility:

```
beautifulsoup4==4.13.5
blinker==1.9.0
bs4==0.0.2
certifi==2025.8.3
charset-normalizer==3.4.3
click==8.2.1
Flask==3.1.2
idna==3.10
itsdangerous==2.2.0
Jinja2==3.1.6
joblib==1.5.2
MarkupSafe==3.0.2
nltk==3.9.1
numpy==2.2.6
pandas==2.3.2
python-dateutil==2.9.0.post0
pytz==2025.2
regex==2025.9.1
requests==2.32.5
six==1.17.0
soupsieve==2.8
tqdm==4.67.1
typing_extensions==4.15.0
tzdata==2025.2
urllib3==2.5.0
Werkzeug==3.1.3
```

---

## Future Improvements

* Containerization (Docker) for deployment.
* Hosted demo at \[Demo URL Placeholder].
* Reinforcement learning for adaptive guessing.
* Improved candidate filtering using morphological analysis.

---

## Author

* Name: Rishabh Rathore
* GitHub: [https://github.com/the-code-karigar](https://github.com/the-code-karigar)
* Email: [rrathore9003@gmail.com](rrathore9003@gmail.com)
