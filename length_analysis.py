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
        if title is None:
            return ""

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

def count_words(text):
    if text is None:
        return 0

    text = text.strip()

    if text == "":
        return 0

    words = re.findall(r"[^\W_]+(?:[-'][^\W_]+)*", text, flags=re.UNICODE)

    return len(words)

def calculate_percentage(count, total):
    if total == 0:
        return 0

    return round(count / total * 100, 2)

def calculate_statistics(series):
    statistics = series.describe().round(4).to_dict()
    statistics["count"] = int(statistics["count"])

    return statistics

def add_bar_percentages(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.5, str(round(height, 2)) + "%", ha="center", va="bottom")


def add_bar_values(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, str(round(height, 2)), ha="center", va="bottom")

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

    elem["language1_length_analysis"] = language1
    elem["language2_length_analysis"] = language2

    elem["length1"] = length1
    elem["length2"] = length2
    
    length_difference = abs(length1 - length2)

    if maximum_length == 0:
        relative_length_difference = 0
        length_ratio = 1
    else:
        relative_length_difference = length_difference / maximum_length
        length_ratio = minimum_length / maximum_length
    
    if length1 == length2:
        same_length = True
    else:
        same_length = False

    if relative_length_difference <= 0.05:
        similar_length_5_percent = True
    else:
        similar_length_5_percent = False

    if relative_length_difference <= 0.10:
        similar_length_10_percent = True
    else:
        similar_length_10_percent = False

    if relative_length_difference <= 0.20:
        similar_length_20_percent = True
    else:
        similar_length_20_percent = False

    if length1 == 1000 and length2 == 1000:
        both_lengths_1000 = True
    else:
        both_lengths_1000 = False

    if length1 == 1000 or length2 == 1000:
        at_least_one_length_1000 = True
    else:
        at_least_one_length_1000 = False

    sentence_count1 = count_sentences(content1, language1)
    sentence_count2 = count_sentences(content2, language2)

    word_count1 = count_words(content1)
    word_count2 = count_words(content2)

    average_word_count = (word_count1 + word_count2) / 2
    word_count_difference = abs(word_count1 - word_count2)

    average_sentence_count = (sentence_count1 + sentence_count2) / 2
    maximum_sentence_count = max(sentence_count1, sentence_count2)
    minimum_sentence_count = min(sentence_count1, sentence_count2)
    sentence_count_difference = abs(sentence_count1 - sentence_count2)

    if maximum_sentence_count == 0:
        relative_sentence_count = 0
        sentence_count_ratio = 1
    else:
        relative_sentence_count = sentence_count_difference / maximum_sentence_count
        sentence_count_ratio = minimum_sentence_count / maximum_sentence_count

    if sentence_count1 == sentence_count2:
        same_sentence_count = True
    else:
        same_sentence_count = False

    elem["average_length"] = average_length

    elem["length_difference"] = length_difference
    elem["relative_length_difference"] = (relative_length_difference)
    elem["length_ratio"] = length_ratio

    elem["same_length"] = same_length
    elem["similar_length_5_percent"] = (similar_length_5_percent)
    elem["similar_length_10_percent"] = (similar_length_10_percent)
    elem["similar_length_20_percent"] = (similar_length_20_percent)

    elem["both_lengths_1000"] = both_lengths_1000
    elem["at_least_one_length_1000"] = (at_least_one_length_1000)

    elem["sentence_count1"] = sentence_count1
    elem["sentence_count2"] = sentence_count2

    elem["word_count1"] = word_count1
    elem["word_count2"] = word_count2

    elem["average_word_count"] = average_word_count
    elem["word_count_difference"] = word_count_difference

    elem["average_sentence_count"] = (average_sentence_count)
    elem["sentence_count_difference"] = (sentence_count_difference)
    elem["relative_sentence_difference"] = (relative_sentence_count)

    elem["sentence_count_ratio"] = sentence_count_ratio
    elem["same_sentence_count"] = same_sentence_count

with open("length_correlation/ai_classification_with_length.jsonl", "w", encoding="utf-8") as file:
    for elem in data:
        file.write(json.dumps(elem, ensure_ascii=False) + "\n")

yes_average_lengths = []
no_average_lengths = []

yes_length_differences = []
no_length_differences = []

yes_relative_length_differences = []
no_relative_length_differences = []

yes_sentence_counts = []
no_sentence_counts = []

yes_sentence_differences = []
no_sentence_differences = []

yes_relative_sentence_differences = []
no_relative_sentence_differences = []

yes_article_word_counts = []
no_article_word_counts = []

yes_article_sentence_counts = []
no_article_sentence_counts = []

yes_average_word_counts = []
no_average_word_counts = []

yes_word_count_differences = []
no_word_count_differences = []

same_length_yes = 0
same_length_no = 0

similar_length_5_yes = 0
similar_length_5_no = 0

similar_length_10_yes = 0
similar_length_10_no = 0

similar_length_20_yes = 0
similar_length_20_no = 0

both_lengths_1000_yes = 0
both_lengths_1000_no = 0

at_least_one_length_1000_yes = 0
at_least_one_length_1000_no = 0

same_sentence_count_yes = 0
same_sentence_count_no = 0

yes_total = 0
no_total = 0

for elem in data:
    length1 = elem["length1"]
    length2 = elem["length2"]
    average_length = (length1 + length2) / 2
    average_sentence_count = (elem["sentence_count1"] + elem["sentence_count2"]) / 2
    average_word_count = (elem["word_count1"] + elem["word_count2"]) / 2

    label = elem["classification_openai/gpt-oss-120b_v1"]
    if label == "Yes":
        yes_total = yes_total + 1
        yes_average_lengths.append(average_length)
        yes_length_differences.append(elem["length_difference"])
        yes_relative_length_differences.append(elem["relative_length_difference"])
        yes_sentence_counts.append(average_sentence_count)
        yes_sentence_differences.append(elem["sentence_count_difference"])
        yes_relative_sentence_differences.append(elem["relative_sentence_difference"])
        yes_article_word_counts.append(elem["word_count1"])
        yes_article_word_counts.append(elem["word_count2"])

        yes_article_sentence_counts.append(elem["sentence_count1"])
        yes_article_sentence_counts.append(elem["sentence_count2"])

        yes_average_word_counts.append(average_word_count)
        yes_word_count_differences.append(elem["word_count_difference"])

        if elem["same_length"] == True:
            same_length_yes = same_length_yes + 1
        if elem["similar_length_5_percent"] == True:
            similar_length_5_yes = similar_length_5_yes + 1
        if elem["similar_length_10_percent"] == True:
            similar_length_10_yes = similar_length_10_yes + 1
        if elem["similar_length_20_percent"] == True:
            similar_length_20_yes = similar_length_20_yes + 1
        if elem["both_lengths_1000"] == True:
            both_lengths_1000_yes = both_lengths_1000_yes + 1
        if elem["at_least_one_length_1000"] == True:
            at_least_one_length_1000_yes = at_least_one_length_1000_yes + 1
        if elem["same_sentence_count"] == True:
            same_sentence_count_yes = same_sentence_count_yes + 1

    elif label == "No":
        no_total = no_total + 1
        no_average_lengths.append(average_length)
        no_length_differences.append(elem["length_difference"])
        no_relative_length_differences.append(elem["relative_length_difference"])
        no_sentence_counts.append(average_sentence_count)
        no_sentence_differences.append(elem["sentence_count_difference"])
        no_relative_sentence_differences.append(elem["relative_sentence_difference"])
        no_article_word_counts.append(elem["word_count1"])
        no_article_word_counts.append(elem["word_count2"])

        no_article_sentence_counts.append(elem["sentence_count1"])
        no_article_sentence_counts.append(elem["sentence_count2"])

        no_average_word_counts.append(average_word_count)
        no_word_count_differences.append(elem["word_count_difference"])

        if elem["same_length"] == True:
            same_length_no = same_length_no + 1
        if elem["similar_length_5_percent"] == True:
            similar_length_5_no = similar_length_5_no + 1
        if elem["similar_length_10_percent"] == True:
            similar_length_10_no = similar_length_10_no + 1
        if elem["similar_length_20_percent"] == True:
            similar_length_20_no = similar_length_20_no + 1
        if elem["both_lengths_1000"] == True:
            both_lengths_1000_no = both_lengths_1000_no + 1
        if elem["at_least_one_length_1000"] == True:
            at_least_one_length_1000_no = at_least_one_length_1000_no + 1
        if elem["same_sentence_count"] == True:
            same_sentence_count_no = same_sentence_count_no + 1

yes_average_lengths_series = pd.Series(yes_average_lengths, dtype=float)
no_average_lengths_series = pd.Series(no_average_lengths, dtype=float)

yes_length_differences_series = pd.Series(yes_length_differences, dtype=float)
no_length_differences_series = pd.Series(no_length_differences, dtype=float)

yes_relative_length_differences_series = pd.Series(yes_relative_length_differences, dtype=float)
no_relative_length_differences_series = pd.Series(no_relative_length_differences, dtype=float)

yes_sentence_counts_series = pd.Series(yes_sentence_counts, dtype=float)
no_sentence_counts_series = pd.Series(no_sentence_counts, dtype=float)

yes_sentence_differences_series = pd.Series(yes_sentence_differences, dtype=float)
no_sentence_differences_series = pd.Series(no_sentence_differences, dtype=float)

yes_relative_sentence_differences_series = pd.Series(yes_relative_sentence_differences, dtype=float)
no_relative_sentence_differences_series = pd.Series(no_relative_sentence_differences, dtype=float)

yes_article_word_counts_series = pd.Series(yes_article_word_counts, dtype=float)
no_article_word_counts_series = pd.Series(no_article_word_counts, dtype=float)

yes_article_sentence_counts_series = pd.Series(yes_article_sentence_counts, dtype=float)
no_article_sentence_counts_series = pd.Series(no_article_sentence_counts, dtype=float)

yes_average_word_counts_series = pd.Series(yes_average_word_counts, dtype=float)
no_average_word_counts_series = pd.Series(no_average_word_counts, dtype=float)

yes_word_count_differences_series = pd.Series(yes_word_count_differences, dtype=float)
no_word_count_differences_series = pd.Series(no_word_count_differences, dtype=float)

yes_average_length_statistics = calculate_statistics(yes_average_lengths_series)
no_average_length_statistics = calculate_statistics(no_average_lengths_series)

yes_length_difference_statistics = calculate_statistics(yes_length_differences_series)
no_length_difference_statistics = calculate_statistics(no_length_differences_series)

yes_relative_length_difference_statistics = calculate_statistics(yes_relative_length_differences_series)
no_relative_length_difference_statistics = calculate_statistics(no_relative_length_differences_series)

yes_sentence_count_statistics = calculate_statistics(yes_sentence_counts_series)
no_sentence_count_statistics = calculate_statistics(no_sentence_counts_series)

yes_sentence_difference_statistics = calculate_statistics(yes_sentence_differences_series)
no_sentence_difference_statistics = calculate_statistics(no_sentence_differences_series)

yes_relative_sentence_difference_statistics = calculate_statistics(yes_relative_sentence_differences_series)
no_relative_sentence_difference_statistics = calculate_statistics(no_relative_sentence_differences_series)

yes_article_word_count_statistics = calculate_statistics(yes_article_word_counts_series)
no_article_word_count_statistics = calculate_statistics(no_article_word_counts_series)

yes_article_sentence_count_statistics = calculate_statistics(yes_article_sentence_counts_series)
no_article_sentence_count_statistics = calculate_statistics(no_article_sentence_counts_series)

yes_average_word_count_statistics = calculate_statistics(yes_average_word_counts_series)
no_average_word_count_statistics = calculate_statistics(no_average_word_counts_series)

yes_word_count_difference_statistics = calculate_statistics(yes_word_count_differences_series)
no_word_count_difference_statistics = calculate_statistics(no_word_count_differences_series)


same_length_yes_percentage = calculate_percentage(same_length_yes, yes_total)
same_length_no_percentage = calculate_percentage(same_length_no, no_total)

similar_length_5_yes_percentage = calculate_percentage(similar_length_5_yes, yes_total)
similar_length_5_no_percentage = calculate_percentage(similar_length_5_no, no_total)

similar_length_10_yes_percentage = calculate_percentage(similar_length_10_yes, yes_total)
similar_length_10_no_percentage = calculate_percentage(similar_length_10_no, no_total)

similar_length_20_yes_percentage = calculate_percentage(similar_length_20_yes, yes_total)
similar_length_20_no_percentage = calculate_percentage(similar_length_20_no, no_total)

both_lengths_1000_yes_percentage = calculate_percentage(both_lengths_1000_yes, yes_total)
both_lengths_1000_no_percentage = calculate_percentage(both_lengths_1000_no, no_total)

at_least_one_length_1000_yes_percentage = calculate_percentage(at_least_one_length_1000_yes, yes_total)
at_least_one_length_1000_no_percentage = calculate_percentage(at_least_one_length_1000_no, no_total)

same_sentence_count_yes_percentage = calculate_percentage(same_sentence_count_yes, yes_total)
same_sentence_count_no_percentage = calculate_percentage(same_sentence_count_no, no_total)


same_length_without_1000_yes = same_length_yes - both_lengths_1000_yes
same_length_without_1000_no = same_length_no - both_lengths_1000_no

yes_without_both_1000_total = yes_total - both_lengths_1000_yes
no_without_both_1000_total = no_total - both_lengths_1000_no

total_words_yes = sum(yes_article_word_counts)
total_words_no = sum(no_article_word_counts)

total_sentences_yes = sum(yes_article_sentence_counts)
total_sentences_no = sum(no_article_sentence_counts)

total_article_occurrences_yes = len(yes_article_word_counts)
total_article_occurrences_no = len(no_article_word_counts)

same_length_without_1000_yes_percentage = calculate_percentage(same_length_without_1000_yes, yes_without_both_1000_total)
same_length_without_1000_no_percentage = calculate_percentage(same_length_without_1000_no, no_without_both_1000_total)


correlation_data = []

for elem in data:
    label = elem["classification_openai/gpt-oss-120b_v1"]

    if label == "Yes":
        label_numeric = 1
    else:
        label_numeric = 0

    correlation_data.append({
        "label_numeric": label_numeric,
        "average_length": elem["average_length"],
        "length_difference": elem["length_difference"],
        "relative_length_difference": elem["relative_length_difference"],
        "average_sentence_count": elem["average_sentence_count"],
        "sentence_count_difference": elem["sentence_count_difference"],
        "relative_sentence_difference": elem["relative_sentence_difference"]
    })


correlation_dataframe = pd.DataFrame(correlation_data)

average_length_correlation = correlation_dataframe["average_length"].corr(correlation_dataframe["label_numeric"])
length_difference_correlation = correlation_dataframe["length_difference"].corr(correlation_dataframe["label_numeric"])
relative_length_difference_correlation = correlation_dataframe["relative_length_difference"].corr(correlation_dataframe["label_numeric"])

average_sentence_count_correlation = correlation_dataframe["average_sentence_count"].corr(correlation_dataframe["label_numeric"])
sentence_count_difference_correlation = correlation_dataframe["sentence_count_difference"].corr(correlation_dataframe["label_numeric"])
relative_sentence_difference_correlation = correlation_dataframe["relative_sentence_difference"].corr(correlation_dataframe["label_numeric"])


correlation_without_1000_data = []

for elem in data:
    if elem["both_lengths_1000"] == False:
        label = elem["classification_openai/gpt-oss-120b_v1"]

        if label == "Yes":
            label_numeric = 1
        else:
            label_numeric = 0

        correlation_without_1000_data.append({
            "label_numeric": label_numeric,
            "average_length": elem["average_length"],
            "length_difference": elem["length_difference"],
            "relative_length_difference": elem["relative_length_difference"],
            "average_sentence_count": elem["average_sentence_count"],
            "sentence_count_difference": elem["sentence_count_difference"],
            "relative_sentence_difference": elem["relative_sentence_difference"]
        })


correlation_without_1000_dataframe = pd.DataFrame(correlation_without_1000_data)

average_length_correlation_without_1000 = correlation_without_1000_dataframe["average_length"].corr(correlation_without_1000_dataframe["label_numeric"])
length_difference_correlation_without_1000 = correlation_without_1000_dataframe["length_difference"].corr(correlation_without_1000_dataframe["label_numeric"])
relative_length_difference_correlation_without_1000 = correlation_without_1000_dataframe["relative_length_difference"].corr(correlation_without_1000_dataframe["label_numeric"])

average_sentence_count_correlation_without_1000 = correlation_without_1000_dataframe["average_sentence_count"].corr(correlation_without_1000_dataframe["label_numeric"])
sentence_count_difference_correlation_without_1000 = correlation_without_1000_dataframe["sentence_count_difference"].corr(correlation_without_1000_dataframe["label_numeric"])
relative_sentence_difference_correlation_without_1000 = correlation_without_1000_dataframe["relative_sentence_difference"].corr(correlation_without_1000_dataframe["label_numeric"])


results = {
    "pair_counts": {
        "total": len(data),
        "yes": yes_total,
        "no": no_total
    },
    "average_length": {
        "yes": yes_average_length_statistics,
        "no": no_average_length_statistics
    },
    "length_difference": {
        "yes": yes_length_difference_statistics,
        "no": no_length_difference_statistics
    },
    "relative_length_difference": {
        "yes": yes_relative_length_difference_statistics,
        "no": no_relative_length_difference_statistics
    },
    "average_sentence_count": {
        "yes": yes_sentence_count_statistics,
        "no": no_sentence_count_statistics
    },
    "sentence_count_difference": {
        "yes": yes_sentence_difference_statistics,
        "no": no_sentence_difference_statistics
    },
    "relative_sentence_difference": {
        "yes": yes_relative_sentence_difference_statistics,
        "no": no_relative_sentence_difference_statistics
    },
    "same_length": {
        "yes_count": same_length_yes,
        "yes_percentage": same_length_yes_percentage,
        "no_count": same_length_no,
        "no_percentage": same_length_no_percentage
    },
    "similar_length_5_percent": {
        "yes_count": similar_length_5_yes,
        "yes_percentage": similar_length_5_yes_percentage,
        "no_count": similar_length_5_no,
        "no_percentage": similar_length_5_no_percentage
    },
    "similar_length_10_percent": {
        "yes_count": similar_length_10_yes,
        "yes_percentage": similar_length_10_yes_percentage,
        "no_count": similar_length_10_no,
        "no_percentage": similar_length_10_no_percentage
    },
    "similar_length_20_percent": {
        "yes_count": similar_length_20_yes,
        "yes_percentage": similar_length_20_yes_percentage,
        "no_count": similar_length_20_no,
        "no_percentage": similar_length_20_no_percentage
    },
    "both_lengths_1000": {
        "yes_count": both_lengths_1000_yes,
        "yes_percentage": both_lengths_1000_yes_percentage,
        "no_count": both_lengths_1000_no,
        "no_percentage": both_lengths_1000_no_percentage
    },
    "at_least_one_length_1000": {
        "yes_count": at_least_one_length_1000_yes,
        "yes_percentage": at_least_one_length_1000_yes_percentage,
        "no_count": at_least_one_length_1000_no,
        "no_percentage": at_least_one_length_1000_no_percentage
    },
    "same_length_without_1000_1000": {
        "yes_count": same_length_without_1000_yes,
        "yes_percentage": same_length_without_1000_yes_percentage,
        "no_count": same_length_without_1000_no,
        "no_percentage": same_length_without_1000_no_percentage
    },
    "same_sentence_count": {
        "yes_count": same_sentence_count_yes,
        "yes_percentage": same_sentence_count_yes_percentage,
        "no_count": same_sentence_count_no,
        "no_percentage": same_sentence_count_no_percentage
    },
    "correlations_with_yes_label": {
        "average_length": round(average_length_correlation, 4),
        "length_difference": round(length_difference_correlation, 4),
        "relative_length_difference": round(relative_length_difference_correlation, 4),
        "average_sentence_count": round(average_sentence_count_correlation, 4),
        "sentence_count_difference": round(sentence_count_difference_correlation, 4),
        "relative_sentence_difference": round(relative_sentence_difference_correlation, 4)
    },
    "correlations_without_1000_1000_pairs": {
        "average_length": round(average_length_correlation_without_1000, 4),
        "length_difference": round(length_difference_correlation_without_1000, 4),
        "relative_length_difference": round(relative_length_difference_correlation_without_1000, 4),
        "average_sentence_count": round(average_sentence_count_correlation_without_1000, 4),
        "sentence_count_difference": round(sentence_count_difference_correlation_without_1000, 4),
        "relative_sentence_difference": round(relative_sentence_difference_correlation_without_1000, 4)
    },
    "word_analysis": {
        "yes": {
            "article_occurrences": total_article_occurrences_yes,
            "total_words": total_words_yes,
            "word_count_per_article": yes_article_word_count_statistics,
            "average_word_count_per_pair": yes_average_word_count_statistics,
            "word_count_difference": yes_word_count_difference_statistics
        },
        "no": {
            "article_occurrences": total_article_occurrences_no,
            "total_words": total_words_no,
            "word_count_per_article": no_article_word_count_statistics,
            "average_word_count_per_pair": no_average_word_count_statistics,
            "word_count_difference": no_word_count_difference_statistics
        }
    },
    "sentence_analysis": {
        "yes": {
            "article_occurrences": total_article_occurrences_yes,
            "total_sentences": total_sentences_yes,
            "sentence_count_per_article": yes_article_sentence_count_statistics,
            "sentence_count_difference": yes_sentence_difference_statistics
        },
        "no": {
            "article_occurrences": total_article_occurrences_no,
            "total_sentences": total_sentences_no,
            "sentence_count_per_article": no_article_sentence_count_statistics,
            "sentence_count_difference": no_sentence_difference_statistics
        }
    }
}


with open("length_correlation/json/length_correlation_statistics.json", "w", encoding="utf-8") as file:
    json.dump(results, file, indent=4, ensure_ascii=False)


with open("length_correlation/reports/length_correlation_report.txt", "w", encoding="utf-8") as file:
    file.write("LENGTH CORRELATION REPORT\n")
    file.write("=========================\n\n")

    file.write("PAIR COUNTS\n")
    file.write("-----------\n")
    file.write("Total pairs: " + str(len(data)) + "\n")
    file.write("Yes pairs: " + str(yes_total) + "\n")
    file.write("No pairs: " + str(no_total) + "\n\n")

    file.write("AVERAGE ARTICLE LENGTH\n")
    file.write("----------------------\n")
    file.write("Yes mean: " + str(yes_average_length_statistics["mean"]) + "\n")
    file.write("Yes median: " + str(yes_average_length_statistics["50%"]) + "\n")
    file.write("No mean: " + str(no_average_length_statistics["mean"]) + "\n")
    file.write("No median: " + str(no_average_length_statistics["50%"]) + "\n\n")

    file.write("ABSOLUTE LENGTH DIFFERENCE\n")
    file.write("--------------------------\n")
    file.write("Yes mean: " + str(yes_length_difference_statistics["mean"]) + "\n")
    file.write("Yes median: " + str(yes_length_difference_statistics["50%"]) + "\n")
    file.write("No mean: " + str(no_length_difference_statistics["mean"]) + "\n")
    file.write("No median: " + str(no_length_difference_statistics["50%"]) + "\n\n")

    file.write("RELATIVE LENGTH DIFFERENCE\n")
    file.write("--------------------------\n")
    file.write("Yes mean: " + str(yes_relative_length_difference_statistics["mean"]) + "\n")
    file.write("Yes median: " + str(yes_relative_length_difference_statistics["50%"]) + "\n")
    file.write("No mean: " + str(no_relative_length_difference_statistics["mean"]) + "\n")
    file.write("No median: " + str(no_relative_length_difference_statistics["50%"]) + "\n\n")

    file.write("SAME LENGTH\n")
    file.write("-----------\n")
    file.write("Yes: " + str(same_length_yes) + " (" + str(same_length_yes_percentage) + "%)\n")
    file.write("No: " + str(same_length_no) + " (" + str(same_length_no_percentage) + "%)\n\n")

    file.write("SIMILAR LENGTH - MAXIMUM 5% DIFFERENCE\n")
    file.write("--------------------------------------\n")
    file.write("Yes: " + str(similar_length_5_yes) + " (" + str(similar_length_5_yes_percentage) + "%)\n")
    file.write("No: " + str(similar_length_5_no) + " (" + str(similar_length_5_no_percentage) + "%)\n\n")

    file.write("SIMILAR LENGTH - MAXIMUM 10% DIFFERENCE\n")
    file.write("---------------------------------------\n")
    file.write("Yes: " + str(similar_length_10_yes) + " (" + str(similar_length_10_yes_percentage) + "%)\n")
    file.write("No: " + str(similar_length_10_no) + " (" + str(similar_length_10_no_percentage) + "%)\n\n")

    file.write("SIMILAR LENGTH - MAXIMUM 20% DIFFERENCE\n")
    file.write("---------------------------------------\n")
    file.write("Yes: " + str(similar_length_20_yes) + " (" + str(similar_length_20_yes_percentage) + "%)\n")
    file.write("No: " + str(similar_length_20_no) + " (" + str(similar_length_20_no_percentage) + "%)\n\n")

    file.write("BOTH TEXTS HAVE LENGTH 1000\n")
    file.write("---------------------------\n")
    file.write("Yes: " + str(both_lengths_1000_yes) + " (" + str(both_lengths_1000_yes_percentage) + "%)\n")
    file.write("No: " + str(both_lengths_1000_no) + " (" + str(both_lengths_1000_no_percentage) + "%)\n\n")

    file.write("AT LEAST ONE TEXT HAS LENGTH 1000\n")
    file.write("---------------------------------\n")
    file.write("Yes: " + str(at_least_one_length_1000_yes) + " (" + str(at_least_one_length_1000_yes_percentage) + "%)\n")
    file.write("No: " + str(at_least_one_length_1000_no) + " (" + str(at_least_one_length_1000_no_percentage) + "%)\n\n")

    file.write("SAME LENGTH WITHOUT 1000-1000 PAIRS\n")
    file.write("-----------------------------------\n")
    file.write("Yes: " + str(same_length_without_1000_yes) + " (" + str(same_length_without_1000_yes_percentage) + "% of remaining Yes pairs)\n")
    file.write("No: " + str(same_length_without_1000_no) + " (" + str(same_length_without_1000_no_percentage) + "% of remaining No pairs)\n\n")

    file.write("SENTENCE COUNTS\n")
    file.write("---------------\n")
    file.write("Yes average sentence count mean: " + str(yes_sentence_count_statistics["mean"]) + "\n")
    file.write("No average sentence count mean: " + str(no_sentence_count_statistics["mean"]) + "\n")
    file.write("Yes relative sentence difference mean: " + str(yes_relative_sentence_difference_statistics["mean"]) + "\n")
    file.write("No relative sentence difference mean: " + str(no_relative_sentence_difference_statistics["mean"]) + "\n\n")

    file.write("SAME SENTENCE COUNT\n")
    file.write("-------------------\n")
    file.write("Yes: " + str(same_sentence_count_yes) + " (" + str(same_sentence_count_yes_percentage) + "%)\n")
    file.write("No: " + str(same_sentence_count_no) + " (" + str(same_sentence_count_no_percentage) + "%)\n\n")

    file.write("CORRELATIONS WITH YES LABEL\n")
    file.write("---------------------------\n")
    file.write("Yes = 1 and No = 0\n")
    file.write("Average length: " + str(round(average_length_correlation, 4)) + "\n")
    file.write("Length difference: " + str(round(length_difference_correlation, 4)) + "\n")
    file.write("Relative length difference: " + str(round(relative_length_difference_correlation, 4)) + "\n")
    file.write("Average sentence count: " + str(round(average_sentence_count_correlation, 4)) + "\n")
    file.write("Sentence count difference: " + str(round(sentence_count_difference_correlation, 4)) + "\n")
    file.write("Relative sentence difference: " + str(round(relative_sentence_difference_correlation, 4)) + "\n\n")

    file.write("CORRELATIONS WITHOUT 1000-1000 PAIRS\n")
    file.write("------------------------------------\n")
    file.write("Average length: " + str(round(average_length_correlation_without_1000, 4)) + "\n")
    file.write("Length difference: " + str(round(length_difference_correlation_without_1000, 4)) + "\n")
    file.write("Relative length difference: " + str(round(relative_length_difference_correlation_without_1000, 4)) + "\n")
    file.write("Average sentence count: " + str(round(average_sentence_count_correlation_without_1000, 4)) + "\n")
    file.write("Sentence count difference: " + str(round(sentence_count_difference_correlation_without_1000, 4)) + "\n")
    file.write("Relative sentence difference: " + str(round(relative_sentence_difference_correlation_without_1000, 4)) + "\n")

    file.write("WORD COUNTS\n")
    file.write("-----------\n")
    file.write("Yes article occurrences: " + str(total_article_occurrences_yes) + "\n")
    file.write("No article occurrences: " + str(total_article_occurrences_no) + "\n")
    file.write("Total words in Yes pairs: " + str(total_words_yes) + "\n")
    file.write("Total words in No pairs: " + str(total_words_no) + "\n")
    file.write("Yes words per article mean: " + str(yes_article_word_count_statistics["mean"]) + "\n")
    file.write("Yes words per article median: " + str(yes_article_word_count_statistics["50%"]) + "\n")
    file.write("No words per article mean: " + str(no_article_word_count_statistics["mean"]) + "\n")
    file.write("No words per article median: " + str(no_article_word_count_statistics["50%"]) + "\n")
    file.write("Yes word difference mean: " + str(yes_word_count_difference_statistics["mean"]) + "\n")
    file.write("No word difference mean: " + str(no_word_count_difference_statistics["mean"]) + "\n\n")

    file.write("SENTENCE COUNTS PER ARTICLE\n")
    file.write("---------------------------\n")
    file.write("Total sentences in Yes pairs: " + str(total_sentences_yes) + "\n")
    file.write("Total sentences in No pairs: " + str(total_sentences_no) + "\n")
    file.write("Yes sentences per article mean: " + str(yes_article_sentence_count_statistics["mean"]) + "\n")
    file.write("Yes sentences per article median: " + str(yes_article_sentence_count_statistics["50%"]) + "\n")
    file.write("No sentences per article mean: " + str(no_article_sentence_count_statistics["mean"]) + "\n")
    file.write("No sentences per article median: " + str(no_article_sentence_count_statistics["50%"]) + "\n")
    file.write("Yes sentence difference mean: " + str(yes_sentence_difference_statistics["mean"]) + "\n")
    file.write("No sentence difference mean: " + str(no_sentence_difference_statistics["mean"]) + "\n\n")

with open("length_correlation/reports/sentence_word_correlation_report.txt", "w", encoding="utf-8") as file:
    same_sentence_difference = same_sentence_count_yes_percentage - same_sentence_count_no_percentage
    word_mean_difference = yes_article_word_count_statistics["mean"] - no_article_word_count_statistics["mean"]
    word_pair_difference = no_word_count_difference_statistics["mean"] - yes_word_count_difference_statistics["mean"]
    sentence_pair_difference = no_sentence_difference_statistics["mean"] - yes_sentence_difference_statistics["mean"]

    file.write("Articles from Yes and No pairs have a similar individual structural size.\n")
    file.write("Yes articles contain an average of " + str(yes_article_word_count_statistics["mean"]) + " words, while No articles contain an average of " + str(no_article_word_count_statistics["mean"]) + " words.\n")
    file.write("The difference between the two classes is " + str(round(word_mean_difference, 4)) + " words per article.\n\n")

    file.write("The difference between the articles inside each pair is more informative than the individual article length.\n")
    file.write("The average word-count difference is " + str(yes_word_count_difference_statistics["mean"]) + " words for Yes pairs and " + str(no_word_count_difference_statistics["mean"]) + " words for No pairs.\n")
    file.write("Therefore, No pairs differ by approximately " + str(round(word_pair_difference, 4)) + " more words than Yes pairs.\n\n")

    file.write("The average number of sentences per article is almost identical between the two classes.\n")
    file.write("Yes articles contain an average of " + str(yes_article_sentence_count_statistics["mean"]) + " sentences, while No articles contain an average of " + str(no_article_sentence_count_statistics["mean"]) + " sentences.\n\n")

    file.write("Yes pairs have a slightly smaller sentence-count difference.\n")
    file.write("The average sentence-count difference is " + str(yes_sentence_difference_statistics["mean"]) + " for Yes pairs and " + str(no_sentence_difference_statistics["mean"]) + " for No pairs.\n")
    file.write("No pairs differ by approximately " + str(round(sentence_pair_difference, 4)) + " more sentences than Yes pairs.\n\n")

    file.write("Pairs with exactly the same number of sentences represent " + str(same_sentence_count_yes_percentage) + "% of Yes pairs and " + str(same_sentence_count_no_percentage) + "% of No pairs.\n")
    file.write("The percentage is " + str(round(same_sentence_difference, 2)) + " percentage points higher for Yes pairs.\n\n")

    file.write("Most pairs differ by only a small number of sentences.\n")
    file.write("The distributions of Yes and No pairs overlap considerably, which means that sentence count alone cannot reliably determine whether two articles describe the same event.\n\n")

    file.write("Overall, Yes pairs tend to have more similar word counts and sentence counts than No pairs.\n")
    file.write("However, these structural differences are moderate and should be used as auxiliary features together with lexical, semantic and temporal similarity.\n")    

structural_values = []

for elem in data:
    structural_values.append({
        "id1": elem["id1"],
        "id2": elem["id2"],
        "classification": elem["classification_openai/gpt-oss-120b_v1"],
        "language1": elem["language1_length_analysis"],
        "language2": elem["language2_length_analysis"],
        "word_count1": elem["word_count1"],
        "word_count2": elem["word_count2"],
        "average_word_count": elem["average_word_count"],
        "word_count_difference": elem["word_count_difference"],
        "sentence_count1": elem["sentence_count1"],
        "sentence_count2": elem["sentence_count2"],
        "average_sentence_count": elem["average_sentence_count"],
        "sentence_count_difference": elem["sentence_count_difference"]
    })

structural_dataframe = pd.DataFrame(structural_values)

structural_dataframe.to_csv("length_correlation/structural_values.csv", index=False, encoding="utf-8")

yes_relative_length_weights = []
no_relative_length_weights = []

for value in yes_relative_length_differences:
    yes_relative_length_weights.append(100 / len(yes_relative_length_differences))

for value in no_relative_length_differences:
    no_relative_length_weights.append(100 / len(no_relative_length_differences))

plt.figure(figsize=(10, 6))
plt.hist(yes_relative_length_differences, bins=30, weights=yes_relative_length_weights, alpha=0.7, label="Yes", edgecolor="black")
plt.hist(no_relative_length_differences, bins=30, weights=no_relative_length_weights, alpha=0.5, label="No", edgecolor="black")
plt.title("Percentage distribution of relative length differences")
plt.xlabel("Relative length difference")
plt.ylabel("Percentage of pairs")
plt.legend()
plt.tight_layout()
plt.savefig("length_correlation/graphical_representation/relative_length_difference_distribution.png")
plt.close()


plt.figure(figsize=(8, 6))
plt.boxplot([yes_relative_length_differences, no_relative_length_differences], tick_labels=["Yes", "No"], orientation="vertical")
yes_median = yes_relative_length_difference_statistics["50%"]
no_median = no_relative_length_difference_statistics["50%"]

plt.text(1, yes_median + 0.04, "Median: " + str(round(yes_median * 100, 2)) + "%", ha="center")
plt.text(2, no_median + 0.04, "Median: " + str(round(no_median * 100, 2)) + "%", ha="center")
plt.title("Relative length difference by classification")
plt.xlabel("Classification")
plt.ylabel("Relative length difference")
plt.tight_layout()
plt.savefig("length_correlation/graphical_representation/relative_length_difference_boxplot.png")
plt.close()


categories = ["Same length", "<= 5%", "<= 10%", "<= 20%"]
yes_percentages = [same_length_yes_percentage, similar_length_5_yes_percentage, similar_length_10_yes_percentage, similar_length_20_yes_percentage]
no_percentages = [same_length_no_percentage, similar_length_5_no_percentage, similar_length_10_no_percentage, similar_length_20_no_percentage]

positions = list(range(len(categories)))
bar_width = 0.35

yes_positions = []
no_positions = []

for position in positions:
    yes_positions.append(position - bar_width / 2)
    no_positions.append(position + bar_width / 2)

plt.figure(figsize=(10, 6))
# plt.bar(yes_positions, yes_percentages, width=bar_width, label="Yes", edgecolor="black", alpha=0.7)
# plt.bar(no_positions, no_percentages, width=bar_width, label="No", edgecolor="black", alpha=0.7)
yes_bars = plt.bar(yes_positions, yes_percentages, width=bar_width, label="Yes", edgecolor="black", alpha=0.7)
no_bars = plt.bar(no_positions, no_percentages, width=bar_width, label="No", edgecolor="black", alpha=0.7)

add_bar_percentages(yes_bars)
add_bar_percentages(no_bars)

plt.xticks(positions, categories)
plt.title("Percentage of pairs with similar lengths")
plt.xlabel("Length similarity criterion")
plt.ylabel("Percentage")
plt.ylim(0, max(yes_percentages + no_percentages) + 10)
plt.legend()
plt.tight_layout()
plt.savefig("length_correlation/graphical_representation/similar_length_percentages.png")
plt.close()


categories_1000 = ["Both are 1000", "At least one is 1000"]
yes_percentages_1000 = [both_lengths_1000_yes_percentage, at_least_one_length_1000_yes_percentage]
no_percentages_1000 = [both_lengths_1000_no_percentage, at_least_one_length_1000_no_percentage]

positions = list(range(len(categories_1000)))

yes_positions = []
no_positions = []

for position in positions:
    yes_positions.append(position - bar_width / 2)
    no_positions.append(position + bar_width / 2)

plt.figure(figsize=(9, 6))
yes_bars = plt.bar(yes_positions, yes_percentages_1000, width=bar_width, label="Yes", edgecolor="black", alpha=0.7)
no_bars = plt.bar(no_positions, no_percentages_1000, width=bar_width, label="No", edgecolor="black", alpha=0.7)

add_bar_percentages(yes_bars)
add_bar_percentages(no_bars)

plt.xticks(positions, categories_1000)
plt.title("Texts with a length of 1000 characters")
plt.xlabel("Criterion")
plt.ylabel("Percentage")
plt.ylim(0, max(yes_percentages_1000 + no_percentages_1000) + 10)
plt.legend()
plt.tight_layout()
plt.savefig("length_correlation/graphical_representation/length_1000_percentages.png")
plt.close()

plt.figure(figsize=(8, 6))
plt.boxplot([yes_relative_sentence_differences, no_relative_sentence_differences], tick_labels=["Yes", "No"], orientation="vertical")
yes_sentence_median = yes_relative_sentence_difference_statistics["50%"]
no_sentence_median = no_relative_sentence_difference_statistics["50%"]

plt.text(1, yes_sentence_median + 0.04, "Median: " + str(round(yes_sentence_median * 100, 2)) + "%", ha="center")
plt.text(2, no_sentence_median + 0.04, "Median: " + str(round(no_sentence_median * 100, 2)) + "%", ha="center")

plt.title("Relative sentence count difference by classification")
plt.xlabel("Classification")
plt.ylabel("Relative sentence count difference")
plt.tight_layout()
plt.savefig("length_correlation/graphical_representation/relative_sentence_difference_boxplot.png")
plt.close()

yes_sentence_weights = []
no_sentence_weights = []

for value in yes_relative_sentence_differences:
    yes_sentence_weights.append(100 / len(yes_relative_sentence_differences))

for value in no_relative_sentence_differences:
    no_sentence_weights.append(100 / len(no_relative_sentence_differences))

plt.figure(figsize=(10, 6))
plt.hist(yes_relative_sentence_differences, bins=30, weights=yes_sentence_weights, alpha=0.7, label="Yes", edgecolor="black")
plt.hist(no_relative_sentence_differences, bins=30, weights=no_sentence_weights, alpha=0.5, label="No", edgecolor="black")
plt.title("Percentage distribution of relative sentence count differences")
plt.xlabel("Relative sentence count difference")
plt.ylabel("Percentage of pairs")
plt.legend()
plt.tight_layout()
plt.savefig("length_correlation/graphical_representation/relative_sentence_difference_distribution.png")
plt.close()


maximum_average_length = pd.Series(yes_average_lengths + no_average_lengths).quantile(0.95)

yes_average_lengths_for_graph = []
no_average_lengths_for_graph = []

for value in yes_average_lengths:
    if value <= maximum_average_length:
        yes_average_lengths_for_graph.append(value)

for value in no_average_lengths:
    if value <= maximum_average_length:
        no_average_lengths_for_graph.append(value)

yes_average_weights = []
no_average_weights = []

for value in yes_average_lengths_for_graph:
    yes_average_weights.append(100 / len(yes_average_lengths_for_graph))

for value in no_average_lengths_for_graph:
    no_average_weights.append(100 / len(no_average_lengths_for_graph))

plt.figure(figsize=(10, 6))
plt.hist(yes_average_lengths_for_graph, bins=30, weights=yes_average_weights, alpha=0.7, label="Yes", edgecolor="black")
plt.hist(no_average_lengths_for_graph, bins=30, weights=no_average_weights, alpha=0.5, label="No", edgecolor="black")
plt.title("Percentage distribution of average article lengths")
plt.xlabel("Average length in characters")
plt.ylabel("Percentage of pairs")
plt.legend()
plt.tight_layout()
plt.savefig("length_correlation/graphical_representation/average_length_distribution.png")
plt.close()


# -----------------
exact_zero_yes = 0
exact_zero_no = 0

near_zero_yes = 0
near_zero_no = 0

for elem in data:
    label = elem["classification_openai/gpt-oss-120b_v1"]
    relative_difference = elem["relative_sentence_difference"]

    if label == "Yes":
        if relative_difference == 0:
            exact_zero_yes = exact_zero_yes + 1
        elif relative_difference <= 1 / 30:
            near_zero_yes = near_zero_yes + 1

    elif label == "No":
        if relative_difference == 0:
            exact_zero_no = exact_zero_no + 1
        elif relative_difference <= 1 / 30:
            near_zero_no = near_zero_no + 1

print("Exact zero Yes:", exact_zero_yes)
print("Near zero Yes:", near_zero_yes)

print("Exact zero No:", exact_zero_no)
print("Near zero No:", near_zero_no)

# grafic cu diferenta numarului de propozitii intre perechi
maximum_sentence_difference = pd.Series(yes_sentence_differences + no_sentence_differences).quantile(0.95)

yes_sentence_differences_for_graph = []
no_sentence_differences_for_graph = []

for value in yes_sentence_differences:
    if value <= maximum_sentence_difference:
        yes_sentence_differences_for_graph.append(value)

for value in no_sentence_differences:
    if value <= maximum_sentence_difference:
        no_sentence_differences_for_graph.append(value)

yes_sentence_difference_weights = []
no_sentence_difference_weights = []

for value in yes_sentence_differences_for_graph:
    yes_sentence_difference_weights.append(100 / len(yes_sentence_differences_for_graph))

for value in no_sentence_differences_for_graph:
    no_sentence_difference_weights.append(100 / len(no_sentence_differences_for_graph))

maximum_value = int(max(yes_sentence_differences_for_graph + no_sentence_differences_for_graph))

bins = []

for value in range(maximum_value + 2):
    bins.append(value - 0.5)

plt.figure(figsize=(10, 6))
plt.hist(yes_sentence_differences_for_graph, bins=bins, weights=yes_sentence_difference_weights, alpha=0.7, label="Yes", edgecolor="black")
plt.hist(no_sentence_differences_for_graph, bins=bins, weights=no_sentence_difference_weights, alpha=0.5, label="No", edgecolor="black")
plt.title("Percentage distribution of sentence count differences")
plt.xlabel("Absolute difference in number of sentences")
plt.ylabel("Percentage of pairs")
plt.xticks(range(maximum_value + 1))
plt.legend()
plt.tight_layout()
plt.savefig("length_correlation/graphical_representation/sentence_count_difference_distribution.png")
plt.close()

# --------
sentence_difference_categories = [
    "0", "1", "2", "3", "4", "5",
    "6", "7", "8", "9", "10", ">10"
]

yes_sentence_difference_counts = [0] * 12
no_sentence_difference_counts = [0] * 12

for value in yes_sentence_differences:
    if value <= 10:
        yes_sentence_difference_counts[value] = yes_sentence_difference_counts[value] + 1
    else:
        yes_sentence_difference_counts[11] = yes_sentence_difference_counts[11] + 1

for value in no_sentence_differences:
    if value <= 10:
        no_sentence_difference_counts[value] = no_sentence_difference_counts[value] + 1
    else:
        no_sentence_difference_counts[11] = no_sentence_difference_counts[11] + 1

yes_sentence_difference_percentages = []
no_sentence_difference_percentages = []

for count in yes_sentence_difference_counts:
    yes_sentence_difference_percentages.append(calculate_percentage(count, yes_total))

for count in no_sentence_difference_counts:
    no_sentence_difference_percentages.append(calculate_percentage(count, no_total))

positions = list(range(len(sentence_difference_categories)))
bar_width = 0.35

yes_positions = []
no_positions = []

for position in positions:
    yes_positions.append(position - bar_width / 2)
    no_positions.append(position + bar_width / 2)

plt.figure(figsize=(13, 7))

yes_bars = plt.bar(yes_positions, yes_sentence_difference_percentages, width=bar_width, label="Yes", edgecolor="black", alpha=0.7)
no_bars = plt.bar(no_positions, no_sentence_difference_percentages, width=bar_width, label="No", edgecolor="black", alpha=0.7)

for bar in yes_bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.5, str(round(height, 2)) + "%", ha="center", va="bottom", fontsize=9)

for bar in no_bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 1.3, str(round(height, 2)) + "%", ha="center", va="bottom", fontsize=9)

plt.xticks(positions, sentence_difference_categories)
plt.title("Absolute sentence count difference")
plt.xlabel("Difference in number of sentences")
plt.ylabel("Percentage")
plt.ylim(0, max(yes_sentence_difference_percentages + no_sentence_difference_percentages) + 7)
plt.legend()
plt.tight_layout()
plt.savefig("length_correlation/graphical_representation/sentence_count_difference_categories.png")
plt.close()

# grafic pt diferenta de cuvinte
word_difference_categories = [
    "0-10",
    "11-25",
    "26-50",
    "51-75",
    "76-100",
    ">100"
]

yes_word_difference_counts = [0] * 6
no_word_difference_counts = [0] * 6

for value in yes_word_count_differences:
    if value <= 10:
        yes_word_difference_counts[0] = yes_word_difference_counts[0] + 1
    elif value <= 25:
        yes_word_difference_counts[1] = yes_word_difference_counts[1] + 1
    elif value <= 50:
        yes_word_difference_counts[2] = yes_word_difference_counts[2] + 1
    elif value <= 75:
        yes_word_difference_counts[3] = yes_word_difference_counts[3] + 1
    elif value <= 100:
        yes_word_difference_counts[4] = yes_word_difference_counts[4] + 1
    else:
        yes_word_difference_counts[5] = yes_word_difference_counts[5] + 1

for value in no_word_count_differences:
    if value <= 10:
        no_word_difference_counts[0] = no_word_difference_counts[0] + 1
    elif value <= 25:
        no_word_difference_counts[1] = no_word_difference_counts[1] + 1
    elif value <= 50:
        no_word_difference_counts[2] = no_word_difference_counts[2] + 1
    elif value <= 75:
        no_word_difference_counts[3] = no_word_difference_counts[3] + 1
    elif value <= 100:
        no_word_difference_counts[4] = no_word_difference_counts[4] + 1
    else:
        no_word_difference_counts[5] = no_word_difference_counts[5] + 1

yes_word_difference_percentages = []
no_word_difference_percentages = []

for count in yes_word_difference_counts:
    yes_word_difference_percentages.append(calculate_percentage(count, yes_total))

for count in no_word_difference_counts:
    no_word_difference_percentages.append(calculate_percentage(count, no_total))

positions = list(range(len(word_difference_categories)))
bar_width = 0.35

yes_positions = []
no_positions = []

for position in positions:
    yes_positions.append(position - bar_width / 2)
    no_positions.append(position + bar_width / 2)

plt.figure(figsize=(11, 7))

yes_bars = plt.bar(yes_positions, yes_word_difference_percentages, width=bar_width, label="Yes", edgecolor="black", alpha=0.7)
no_bars = plt.bar(no_positions, no_word_difference_percentages, width=bar_width, label="No", edgecolor="black", alpha=0.7)

for bar in yes_bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.4, str(round(height, 2)) + "%", ha="center", va="bottom", fontsize=9)

for bar in no_bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 1.1, str(round(height, 2)) + "%", ha="center", va="bottom", fontsize=9)

plt.xticks(positions, word_difference_categories)
plt.title("Absolute word count difference")
plt.xlabel("Difference in number of words")
plt.ylabel("Percentage")
plt.ylim(0, max(yes_word_difference_percentages + no_word_difference_percentages) + 7)
plt.legend()
plt.tight_layout()
plt.savefig("length_correlation/graphical_representation/word_count_difference_categories.png")
plt.close()

# ditribuitia cuvintelor per articol
maximum_word_count = pd.Series(yes_article_word_counts + no_article_word_counts).quantile(0.99)

yes_word_counts_for_graph = []
no_word_counts_for_graph = []

for value in yes_article_word_counts:
    if value <= maximum_word_count:
        yes_word_counts_for_graph.append(value)

for value in no_article_word_counts:
    if value <= maximum_word_count:
        no_word_counts_for_graph.append(value)

yes_word_weights = []
no_word_weights = []

for value in yes_word_counts_for_graph:
    yes_word_weights.append(100 / len(yes_word_counts_for_graph))

for value in no_word_counts_for_graph:
    no_word_weights.append(100 / len(no_word_counts_for_graph))

plt.figure(figsize=(10, 6))
plt.hist(yes_word_counts_for_graph, bins=30, weights=yes_word_weights, histtype="step", linewidth=2, label="Yes")
plt.hist(no_word_counts_for_graph, bins=30, weights=no_word_weights, histtype="step", linewidth=2, label="No")
plt.title("Percentage distribution of word counts per article")
plt.xlabel("Number of words per article")
plt.ylabel("Percentage of article occurrences")
plt.legend()
plt.tight_layout()
plt.savefig("length_correlation/graphical_representation/word_count_per_article_distribution.png")
plt.close()