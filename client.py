import requests

# Endpoint URL for the guess API on local host
url = "http://127.0.0.1:5000/play"

# JSON payload representing the current game state
payload = {
    "currentWordState": "_ a t",  # Current revealed letters of the word
    "guessedLetters": ["a", "m", "t", "e", "l"],  # Letters guessed so far
    "guessesRemaining": 6  # Remaining incorrect guesses allowed
}

# Send POST request to the API with the game state in json format
response = requests.post(url, json=payload)

# Print server response
print(response.text)
