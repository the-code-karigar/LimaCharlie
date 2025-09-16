import re
import os
import nltk
import time
import random
import requests
from bs4 import BeautifulSoup

# Download words corpus if not already downloaded
nltk.download('words')
from nltk.corpus import words as nltk_words


# # # # # # # # # # # # # # # # # # Utility Functions # # # # # # # # # # # # # # # # # #

def write_words(word_list: list, file_name: str):
    """
    Write a list of words to a file, one word per line.

    Args:
        word_list (list): List of words to write.
        file_name (str): Path to the output file.
    """
    with open(file_name, "w", encoding="utf-8") as f:
        for word in word_list:
            f.write(word + "\n")


def read_words(filename: str) -> list:
    """
    Read words from a file into a list. Only alphabetic words are included.

    Args:
        filename (str): Path to the input file.

    Returns:
        list: List of lowercase words read from the file.
    """
    word_list = []
    if os.path.exists(filename):
        with open(filename) as file:
            word_list = [w.strip().lower() for w in file if w.strip().isalpha()]
    return word_list


def load_words_from_nltk(min_len=0, max_len=100) -> list:
    """
    Load words from the NLTK words corpus and filter by length.

    Args:
        min_len (int, optional): Minimum length of words to include. Defaults to 0.
        max_len (int, optional): Maximum length of words to include. Defaults to 100.

    Returns:
        list: Sorted list of lowercase words within the specified length range.
    """
    word_list = sorted(list(set([w.lower() for w in nltk_words.words() if w.isalpha()])))
    return [w for w in word_list if min_len <= len(w) <= max_len]


def replacer(ip_txt: str):
    """
    Replace all non-alphabetic characters in text with spaces.

    Args:
        ip_txt (str): Input string.

    Returns:
        str: Cleaned string containing only letters and spaces.
    """
    return re.sub(r'[^A-Za-z\s]', ' ', ip_txt)


# # # # # # # # # # # # # # # # # # Corpus Creation # # # # # # # # # # # # # # # # # #

def create_airline_corpus(wikipedia_dump_file: str, airline_corpus_file: str):
    """
    Create a cleaned airline corpus file from a Wikipedia dump.

    Args:
        wikipedia_dump_file (str): Path to raw scraped Wikipedia text file.
        airline_corpus_file (str): Path to output corpus file containing unique words.
    """
    with open(wikipedia_dump_file, "r", encoding="utf-8") as f:
        text = replacer(f.read().lower())
    wiki_words = sorted(list(set(re.findall(r'\b[a-zA-Z]+\b', text))))
    write_words(wiki_words, airline_corpus_file)


# # # # # # # # # # # # # # # # # # Wikipedia Scraping # # # # # # # # # # # # # # # # # #

def wiki_scraper_recursive(link: str, wiki_dump_filename: str):
    """
    Recursively scrape Wikipedia pages starting from a given link and save content.

    Args:
        link (str): Starting Wikipedia URL.
        wiki_dump_filename (str): Output file where scraped text will be appended.
    """
    # Global settings
    visited = set()  # To avoid revisiting pages

    # Headers to avoid 403 errors
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/122.0.0.0 Safari/537.36"
    }

    def scrape_wiki(url, depth=0, max_depth=1):
        """Recursively scrape Wikipedia pages up to max_depth."""
        if depth > max_depth or url in visited:
            return

        visited.add(url)

        try:
            # Random sleep to avoid hammering Wikipedia
            sleep_time = random.randint(1, 3)
            time.sleep(sleep_time)

            # Fetch page
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract text and append to text file
            page_text = soup.get_text()
            with open(wiki_dump_filename, "a", encoding="utf-8") as f:
                f.write(f"\n\n--- Page: {url} (Depth {depth}) ---\n\n")
                f.write(replacer(page_text.replace('\n', ' ')))

            # Extract Wikipedia links and recurse
            for link in soup.find_all("a", href=True):
                href = link["href"]
                if href.startswith("/wiki/") and not href.startswith("/wiki/Special:"):
                    next_url = "https://en.wikipedia.org" + href
                    scrape_wiki(next_url, depth + 1, max_depth)
                    print(f"{next_url} done")

        except Exception as e:
            print(f"Failed to scrape {url}: {e}")

    scrape_wiki(link, depth=0, max_depth=3)
    print("Scraping complete. Output saved to", wiki_dump_filename)


def wiki_scraper(link: str, wiki_dump_filename: str):
    """
    Scrape a single Wikipedia page and save its content.

    Args:
        link (str): Wikipedia page URL.
        wiki_dump_filename (str): Output file where scraped text will be appended.
    """
    # Headers to avoid 403 errors
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/122.0.0.0 Safari/537.36"
    }

    try:
        # Random sleep to avoid hammering Wikipedia
        sleep_time = random.randint(1, 3)
        time.sleep(sleep_time)

        # Fetch page
        response = requests.get(link, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract text and append to text file
        page_text = soup.get_text()
        with open(wiki_dump_filename, "a", encoding="utf-8") as f:
            f.write(f"\n\n--- Page: {link}  ---\n\n")
            f.write(replacer(page_text.replace('\n', ' ')))

    except Exception as e:
        print(f"Failed to scrape {link}: {e}")

    print(f"Scraping of {link} complete. Output saved to", wiki_dump_filename)


# # # # # # # # # # # # # # # # # # Secret Word Lists For Testing # # # # # # # # # # # # # # # # # #

def create_secrets(secret_file: str, airline_file: str, word_file: str):
    """
    Create a mixed secret word list from airline-specific and general words.

    Args:
        secret_file (str): Path to output secret file.
        airline_file (str): Path to airline-specific corpus file.
        word_file (str): Path to general word corpus file.
    """
    general_words = set(read_words(word_file))
    airline_words = set(read_words(airline_file)) - general_words
    secrets = list(random.sample(airline_words, 50) + random.sample(general_words, 50))
    write_words(secrets, secret_file)


if __name__ == "__main__":
    # create clean corpus of english language words
    write_words(load_words_from_nltk(), "corpus/word_corpus.txt")

    # scrape Wikipedia
    if False:  # Change this to false if you don't want to scrape
        keywords = ["IndiGo", "Airline", "Airport", "Air_traffic_control", "Aircraft_pilot",
                    "Aviation_accidents_and_incidents", "Aviation", "Airplane",
                    "List_of_airlines_of_India", "Air_India", "IndiGo", "SpiceJet", "Akasa_Air", "Vistara", "Go_First",
                    "Air_India_Express", "AirAsia_India", "Alliance_Air_%28India%29", "Quikjet_Airways",
                    "Blue_Dart_Aviation", "Star_Air_%28India%29", "Flybig", "Zoom_Air", "Deccan_Charters",
                    "List_of_defunct_airlines_of_India", "Air_cargo", "Freight_airline", "Regional_airliner",
                    "Low-cost_carrier", "Full-service_airline", "Airline_hub", "Airline_route_network", "Interlining",
                    "Codeshare_agreement", "Frequent-flyer_program", "Global_distribution_system",
                    "Airline_reservation_system", "Revenue_management_%28airlines%29", "Aircraft_lease",
                    "Aircraft_maintenance", "Maintenance,_repair,_and_overhaul", "Ground_handling",
                    "Ground_support_equipment", "Fuel_management", "Weight_and_balance", "Cabin_service",
                    "Catering_%28airline%29", "Fleet_commonality", "Air_traffic_control",
                    "Air_traffic_control_in_India", "Airports_Authority_of_India", "List_of_airports_in_India",
                    "Indira_Gandhi_International_Airport", "Chhatrapati_Shivaji_Maharaj_International_Airport",
                    "Kempegowda_International_Airport", "Netaji_Subhas_Chandra_Bose_International_Airport",
                    "Chennai_International_Airport", "Hyderabad_Shamsheer_Bhavan_Rajiv_Gandhi_International_Airport",
                    "Cochin_International_Airport", "Dabolim_Airport", "Minor_airports_of_India", "UDAN",
                    "Regional_Connectivity_Scheme", "Ministry_of_Civil_Aviation_%28India%29",
                    "Directorate_General_of_Civil_Aviation_%28India%29", "Bureau_of_Civil_Aviation_Security",
                    "Aircraft_Accident_Investigation_Bureau_%28India%29", "Airports_Economic_Regulatory_Authority",
                    "Airport_security_in_India", "Aviation_safety", "Crew_resource_management",
                    "Threat_and_error_management", "Aviation_Safety_Reporting_System", "Aircraft_incident_and_accident",
                    "Black_box_%28flight_data_recorder%29", "Airworthiness", "Type_certification",
                    "Airworthiness_directive", "Pilot_%28aviation%29", "Flight_attendant", "Air_traffic_controller",
                    "Aircraft_maintenance_engineer", "Flight_dispatcher", "Airline_captain", "First_officer",
                    "Flight_engineer", "Check_pilot", "Cabin_crew_training", "Flight_training_school",
                    "Indian_Aviation_Academy", "Aviation_colleges_in_India", "Aviation_regulation_in_India",
                    "Aircraft_registration_in_India", "Aircraft_manufacturers_of_India",
                    "Hindustan_Aeronautics_Limited", "MRO_industry", "Airline_management", "Airline_marketing",
                    "Airline_finance", "Airline_deregulation_in_India", "Privatisation_of_Air_India",
                    "Airport_privatisation_in_India", "Airport_slots", "No-frills_airline", "Passenger_rights_in_India",
                    "Disabled_passenger_rights_in_air_travel", "List_of_aviation_abbreviations"
                    ]
        for keyword in keywords:
            start_url = f"https://en.wikipedia.org/wiki/{keyword}"
            wiki_scraper(start_url, "corpus/airline_wiki.txt")

    # Build an airline-specific corpus from the scraped text
    create_airline_corpus("corpus/airline_wiki.txt", "corpus/airline_corpus.txt")

    # Generate secret word lists for testing reasons
    create_secrets("corpus/test/secrets.txt", "corpus/airline_corpus.txt", "corpus/word_corpus.txt")
