import json
import os
import re
import nltk
from nltk.tokenize import sent_tokenize
from langdetect import detect, DetectorFactory
import matplotlib.pyplot as plt
import pandas as pd

DetectorFactory.seed = 0

nltk.download("punkt")
nltk.download("punkt_tab")

def empty_text(text):
    if text is None:
        return True

    text = text.strip()

    if text == "":
        return True

    return False


def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"


def get_text(content, title):
    if empty_text(content):
        return title

    return content

def count_sentences(text, language):
    if text is None:
        return 0

    text = text.strip()

    if text == "":
        return 0

    nltk_languages = {
        "cs": "czech",
        "da": "danish",
        "de": "german",
        "el": "greek",
        "en": "english",
        "es": "spanish",
        "et": "estonian",
        "fi": "finnish",
        "fr": "french",
        "it": "italian",
        "nl": "dutch",
        "no": "norwegian",
        "pl": "polish",
        "pt": "portuguese",
        "ru": "russian",
        "sl": "slovene",
        "sv": "swedish",
        "tr": "turkish"
    }

    if language in nltk_languages:
        try:
            sentences = sent_tokenize(text, language=nltk_languages[language])
            sentence_count = 0

            for sentence in sentences:
                if sentence.strip() != "":
                    sentence_count = sentence_count + 1

            return sentence_count

        except:
            pass

    sentences = re.split(r"(?<=[.!?])\s+", text)

    sentence_count = 0

    for sentence in sentences:
        if sentence.strip() != "":
            sentence_count = sentence_count + 1

    return sentence_count

def calculate_percentage(count, total):
    if total == 0:
        return 0

    return round(count / total * 100, 2)

def calculate_statistics(series):
    statistics = series.describe().round(4).to_dict()
    statistics["count"] = int(statistics["count"])

    return statistics

data = []
with open("ai_classification_cleaned.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        data.append(json.loads(line))

os.makedirs("length_correlation", exist_ok=True)
os.makedirs("length_correlation/json", exist_ok=True)
os.makedirs("length_correlation/reports", exist_ok=True)
os.makedirs("length_correlation/graphical_representation", exist_ok=True)

for elem in data:
    content1 = get_text(elem["content1"], elem["title1"])
    content2 = get_text(elem["content2"], elem["title2"])

    language1 = detect_language(content1)
    language2 = detect_language(content2)

    length1 = len(content1)
    length2 = len(content2)

    average_length = (length1 + length2) / 2

    maximum_length = max(length1, length2)
    minimum_length = min(length1, length2)

    elem["length1"] = length1
    elem["length2"] = length2
    
    length_difference = abs(length1 - length2)
    if maximum_length == 0:
        relative_length_difference = 0
    else:
        relative_length_difference = length_difference / maximum_length
    
    if length1 == length2:
        same_length = True
    else:
        same_length = False

    elem["length_difference"] = length_difference
    elem["relative_length_difference"] = relative_length_difference
    elem["same_length"] = same_length

with open("length_correlation/ai_classification_with_length.jsonl", "w", encoding="utf-8") as file:
    for elem in data:
        file.write(json.dumps(elem, ensure_ascii=False) + "\n")

yes_average_lengths = []
no_average_lengths = []

yes_length_differences = []
no_length_differences = []

yes_relative_length_differences = []
no_relative_length_differences = []

same_length_yes = 0
same_length_no = 0

for elem in data:
    length1 = elem["length1"]
    length2 = elem["length2"]
    average_length = (length1 + length2) / 2

    label = elem["classification_openai/gpt-oss-120b_v1"]
    if label == "Yes":
        yes_average_lengths.append(average_length)
        yes_length_differences.append(elem["length_difference"])
        yes_relative_length_differences.append(elem["relative_length_difference"])

        if elem["same_length"] == True:
            same_length_yes = same_length_yes + 1

    elif label == "No":
        no_average_lengths.append(average_length)
        no_average_lengths.append(elem["length_difference"])
        no_relative_length_differences.append(elem["relative_length_difference"])

        if elem["same_length"] == False:
            same_length_no = same_length_no + 1

