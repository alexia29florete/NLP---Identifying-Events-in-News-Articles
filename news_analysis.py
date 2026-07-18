import json
import os
from langdetect import detect, DetectorFactory
from rapidfuzz.distance import Levenshtein
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd

DetectorFactory.seed = 0
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"
    
def normalize_text(text):
    text = text.lower()
    text = " ".join(text.split())

    return text

def empty_text(text):
    if text is None:
        return True

    text = text.strip()

    if text == "":
        return True

    return False

data = []
with open("ai_classification_cleaned.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        data.append(json.loads(line))

# selectez perechile care au eticheta de similaritate YES
similar_news = []
for elem in data:
    if elem["classification_openai/gpt-oss-120b_v1"] == "Yes":
        similar_news.append(elem)

print("Pairs classified as Yes:", len(similar_news))

analyzed_data = []

trivial_dup = 0
near_dup = 0
diff = 0

possible_literal_translation = 0
similar_cross_language_news = 0
different_cross_language_news = 0

trivial_duplicate_pairs = []
near_duplicate_pairs = []
different_news_same_event_pairs = []

possible_literal_translation_pairs = []
similar_cross_language_news_pairs = []
different_cross_language_news_pairs = []

temporal_distances = []
large_temporal_distance_pairs = []

normalized_edit_distances = []
multilingual_similarities = []

same_language_temporal_distances = []
different_language_temporal_distances = []

different_language_pairs = []
language_pair_counts = {}

maximum_multilingual_similarity = None
minimum_multilingual_similarity = None

maximum_similarity_pair = None
minimum_similarity_pair = None

for index, elem in enumerate(similar_news, start = 1):
    content1 = elem["content1"]
    content2 = elem["content2"]

    # daca articolul nu are continut, folosesc titlul
    if empty_text(content1):
        content1 = elem["title1"]

    if empty_text(content2):
        content2 = elem["title2"]

    date1 = datetime.fromisoformat(elem["date1"])
    date2 = datetime.fromisoformat(elem["date2"])

    temporal_distance = abs(date1 - date2)
    temporal_distance_days = temporal_distance.total_seconds() / 86400
    elem["temporal_distance_days"] = temporal_distance_days
    temporal_distances.append(temporal_distance_days)

    # salvez separat perechile aflate la peste 90 de zile

    if temporal_distance_days > 90:
        large_temporal_distance_pairs.append(elem)

    language1 = detect_language(content1)
    language2 = detect_language(content2)

    elem["language1"] = language1
    elem["language2"] = language2

    # atunci cand articolele sunt scrise in aceeasi limba, pot folosi distanta Levenshtein
    if language1 == language2:
        content1 = normalize_text(content1)
        content2 = normalize_text(content2)
        edit_distance = Levenshtein.distance(content1, content2)
        maximum_length = max(len(content1), len(content2))

        # edit distance normalizat
        if maximum_length == 0:
            normalized_edit_distance = 0
        else:
            normalized_edit_distance = edit_distance / maximum_length
        
        elem["normalized_edit_distance"] = normalized_edit_distance
        elem["multilingual_similarity"] = None

        normalized_edit_distances.append(normalized_edit_distance)
        same_language_temporal_distances.append(temporal_distance_days)

        if normalized_edit_distance <= 0.05:
            elem["similarity_type"] = "trivial_duplicate"
            trivial_dup = trivial_dup + 1
            trivial_duplicate_pairs.append(elem)
        elif normalized_edit_distance <= 0.20:
            elem["similarity_type"] = "near_duplicate"
            near_dup = near_dup + 1
            near_duplicate_pairs.append(elem)
        else:
            elem["similarity_type"] = "different_news_same_event"
            diff = diff + 1
            different_news_same_event_pairs.append(elem)
    
    # limbi diferite -> vad daca s-a tradus dintr-o limba in alta mot a mot
    else:
        # transform textul intr-un vector numeric, numit embedding
        embedding1 = model.encode(content1, convert_to_tensor = True)
        embedding2 = model.encode(content2, convert_to_tensor = True)
        multilingual_similarity = cos_sim(embedding1, embedding2).item()

        if maximum_multilingual_similarity is None or multilingual_similarity > maximum_multilingual_similarity:
            maximum_multilingual_similarity = multilingual_similarity
            maximum_similarity_pair = elem.copy()
            maximum_similarity_pair["language1"] = language1
            maximum_similarity_pair["language2"] = language2
            maximum_similarity_pair["multilingual_similarity"] = multilingual_similarity
            maximum_similarity_pair["temporal_distance_days"] = temporal_distance_days

        if minimum_multilingual_similarity is None or multilingual_similarity < minimum_multilingual_similarity:
            minimum_multilingual_similarity = multilingual_similarity
            minimum_similarity_pair = elem.copy()
            minimum_similarity_pair["language1"] = language1
            minimum_similarity_pair["language2"] = language2
            minimum_similarity_pair["multilingual_similarity"] = multilingual_similarity
            minimum_similarity_pair["temporal_distance_days"] = temporal_distance_days

        elem["normalized_edit_distance"] = None
        elem["multilingual_similarity"] = multilingual_similarity

        multilingual_similarities.append(multilingual_similarity)
        different_language_temporal_distances.append(temporal_distance_days)
        different_language_pairs.append(elem)

        language_pair = language1 + "-" + language2

        if language1 > language2:
            language_pair = language2 + "-" + language1

        if language_pair not in language_pair_counts:
            language_pair_counts[language_pair] = 0

        language_pair_counts[language_pair] = language_pair_counts[language_pair] + 1

        if multilingual_similarity >= 0.90:
            elem["similarity_type"] = "possible_literal_translation"
            possible_literal_translation = possible_literal_translation + 1
            possible_literal_translation_pairs.append(elem)
        elif multilingual_similarity >= 0.75:
            elem["similarity_type"] = "similar_cross_language_news"
            similar_cross_language_news = similar_cross_language_news + 1
            similar_cross_language_news_pairs.append(elem)
        else:
            elem["similarity_type"] = "different_cross_language_news"
            different_cross_language_news = different_cross_language_news + 1
            different_cross_language_news_pairs.append(elem)

    analyzed_data.append(elem)

# creez folderul similarity daca nu exista
os.makedirs("similarity", exist_ok=True)
os.makedirs("similarity/graphical_representation", exist_ok=True)
os.makedirs("similarity/json", exist_ok=True)
os.makedirs("similarity/json/same_language", exist_ok=True)
os.makedirs("similarity/json/different_language", exist_ok=True)
os.makedirs("similarity/json/temporal_distribution", exist_ok=True)
os.makedirs("similarity/reports", exist_ok=True)
os.makedirs("similarity/graphical_representation/value_distributions", exist_ok=True)
os.makedirs("similarity/graphical_representation/different_language", exist_ok=True)
os.makedirs("similarity/json/value_distributions", exist_ok=True)
os.makedirs("similarity/reports/value_distributions", exist_ok=True)

with open("similarity/reports/similarity_summary.txt", "w", encoding="utf-8") as file:
    file.write("Pairs classified as Yes: " + str(len(similar_news)) + "\n")
    file.write("trivial_duplicate: " + str(trivial_dup) + " pairs, " + str(round(trivial_dup / len(similar_news) * 100, 2)) + "%\n")
    file.write("near_duplicate: " + str(near_dup) + " pairs, " + str(round(near_dup / len(similar_news) * 100, 2)) + "%\n")
    file.write("different_news_same_event: " + str(diff) + " pairs, " + str(round(diff / len(similar_news) * 100, 2)) + "%\n")
    file.write("possible_literal_translation: " + str(possible_literal_translation) + " pairs, " + str(round(possible_literal_translation / len(similar_news) * 100, 2)) + "%\n")
    file.write("similar_cross_language_news: " + str(similar_cross_language_news) + " pairs, " + str(round(similar_cross_language_news / len(similar_news) * 100, 2)) + "%\n")
    file.write("different_cross_language_news: " + str(different_cross_language_news) + " pairs, " + str(round(different_cross_language_news / len(similar_news) * 100, 2)) + "%\n")
    
with open("similarity/json/similar_news_analysis.jsonl", "w", encoding="utf-8") as file:
    for elem in analyzed_data:
        file.write(json.dumps(elem, ensure_ascii=False) + "\n")

with open("similarity/json/same_language/trivial_duplicate.jsonl", "w", encoding="utf-8") as file:
    for elem in trivial_duplicate_pairs:
        file.write(json.dumps(elem, ensure_ascii=False) + "\n")

with open("similarity/json/same_language/near_duplicate.jsonl", "w", encoding="utf-8") as file:
    for elem in near_duplicate_pairs:
        file.write(json.dumps(elem, ensure_ascii=False) + "\n")

with open("similarity/json/same_language/different_news_same_event.jsonl", "w", encoding="utf-8") as file:
    for elem in different_news_same_event_pairs:
        file.write(json.dumps(elem, ensure_ascii=False) + "\n")

with open("similarity/json/different_language/possible_literal_translation.jsonl", "w", encoding="utf-8") as file:
    for elem in possible_literal_translation_pairs:
        file.write(json.dumps(elem, ensure_ascii=False) + "\n")

with open("similarity/json/different_language/similar_cross_language_news.jsonl", "w", encoding="utf-8") as file:
    for elem in similar_cross_language_news_pairs:
        file.write(json.dumps(elem, ensure_ascii=False) + "\n")

with open("similarity/json/different_language/different_cross_language_news.jsonl", "w", encoding="utf-8") as file:
    for elem in different_cross_language_news_pairs:
        file.write(json.dumps(elem, ensure_ascii=False) + "\n")

extreme_similarity_pairs = {
    "maximum_similarity_pair": maximum_similarity_pair,
    "minimum_similarity_pair": minimum_similarity_pair
}

with open("similarity/json/different_language/extreme_similarity_pairs.json", "w", encoding="utf-8") as file:
    json.dump(extreme_similarity_pairs, file, indent=4, ensure_ascii=False)

with open("similarity/reports/extreme_similarity_pairs.txt", "w", encoding="utf-8") as file:
    file.write("MAXIMUM MULTILINGUAL SIMILARITY\n")
    file.write("================================\n")
    file.write("Score: " + str(maximum_multilingual_similarity) + "\n")
    file.write("Languages: " + maximum_similarity_pair["language1"] + " - " + maximum_similarity_pair["language2"] + "\n")
    file.write("Regions: " + maximum_similarity_pair["region1"] + " - " + maximum_similarity_pair["region2"] + "\n")
    file.write("Temporal distance: " + str(round(maximum_similarity_pair["temporal_distance_days"], 2)) + " days\n")
    file.write("Title 1: " + str(maximum_similarity_pair["title1"]) + "\n")
    file.write("Title 2: " + str(maximum_similarity_pair["title2"]) + "\n\n")
    file.write("Content 1:\n" + str(maximum_similarity_pair["content1"]) + "\n\n")
    file.write("Content 2:\n" + str(maximum_similarity_pair["content2"]) + "\n\n")

    file.write("MINIMUM MULTILINGUAL SIMILARITY\n")
    file.write("================================\n")
    file.write("Score: " + str(minimum_multilingual_similarity) + "\n")
    file.write("Languages: " + minimum_similarity_pair["language1"] + " - " + minimum_similarity_pair["language2"] + "\n")
    file.write("Regions: " + minimum_similarity_pair["region1"] + " - " + minimum_similarity_pair["region2"] + "\n")
    file.write("Temporal distance: " + str(round(minimum_similarity_pair["temporal_distance_days"], 2)) + " days\n")
    file.write("Title 1: " + str(minimum_similarity_pair["title1"]) + "\n")
    file.write("Title 2: " + str(minimum_similarity_pair["title2"]) + "\n\n")
    file.write("Content 1:\n" + str(minimum_similarity_pair["content1"]) + "\n\n")
    file.write("Content 2:\n" + str(minimum_similarity_pair["content2"]) + "\n")

# reprezentare grafica
category_names = ["trivial_duplicate", "near_duplicate", "different_news_same_event", "possible_literal_translation", "similar_cross_language_news", "different_cross_language_news"]

category_counts = [trivial_dup, near_dup, diff, possible_literal_translation, similar_cross_language_news, different_cross_language_news]

category_percentages = []

for count in category_counts:
    percentage = count / len(similar_news) * 100
    category_percentages.append(percentage)

plt.figure(figsize=(13, 7))
plt.bar(category_names, category_percentages)

plt.title("Distributia categoriilor de similaritate")
plt.xlabel("Categorie")
plt.ylabel("Procent din perechile clasificate")

plt.xticks(rotation=30, ha="right")

for index in range(len(category_percentages)):
    plt.text(
        index,
        category_percentages[index],
        str(round(category_percentages[index], 2)) + "%",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.savefig("similarity/graphical_representation/similarity_percentages.png")
plt.close()

# grafic pentru distanta temporala
plt.figure(figsize=(10, 6))

plt.hist(temporal_distances, bins=30, edgecolor="black")

plt.title("Distributia distantelor temporale pentru stirile similare")
plt.xlabel("Distanta temporala in zile")
plt.ylabel("Numar de perechi")

plt.tight_layout()
plt.savefig("similarity/graphical_representation/temporal_distance_distribution.png")
plt.close()

# grafic pentru distante de maximum 30 de zile
temporal_distances_30_days = []

for distance in temporal_distances:
    if distance <= 30:
        temporal_distances_30_days.append(distance)


plt.figure(figsize=(10, 6))

plt.hist(temporal_distances_30_days, bins=30, edgecolor="black")
plt.title("Distributia distantelor temporale de maximum 30 de zile")
plt.xlabel("Distanta temporala in zile")
plt.ylabel("Numar de perechi")
plt.tight_layout()
plt.savefig("similarity/graphical_representation/temporal_distance_distribution_30_days.png")
plt.close()

# impart distantele temporale pe intervale

same_day = 0
one_to_seven_days = 0
eight_to_thirty_days = 0
thirty_one_to_ninety_days = 0
more_than_ninety_days = 0

for distance in temporal_distances:
    if distance < 1:
        same_day = same_day + 1
    elif distance <= 7:
        one_to_seven_days = one_to_seven_days + 1
    elif distance <= 30:
        eight_to_thirty_days = eight_to_thirty_days + 1
    elif distance <= 90:
        thirty_one_to_ninety_days = thirty_one_to_ninety_days + 1
    else:
        more_than_ninety_days = more_than_ninety_days + 1

temporal_categories = [
    "Aceeasi zi",
    "1-7 zile",
    "8-30 zile",
    "31-90 zile",
    "Peste 90 zile"
]

temporal_counts = [
    same_day,
    one_to_seven_days,
    eight_to_thirty_days,
    thirty_one_to_ninety_days,
    more_than_ninety_days
]

temporal_percentages = []

for count in temporal_counts:
    percentage = count / len(temporal_distances) * 100
    temporal_percentages.append(percentage)

plt.figure(figsize=(10, 6))
plt.bar(temporal_categories, temporal_percentages)

plt.title("Distributia distantelor temporale pentru stirile similare")
plt.xlabel("Interval temporal")
plt.ylabel("Procent din perechi")

for index in range(len(temporal_percentages)):
    plt.text(
        index,
        temporal_percentages[index],
        str(round(temporal_percentages[index], 2)) + "%",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.savefig("similarity/graphical_representation/temporal_distance_intervals.png")
plt.close()

# calculez statisticile pentru distantele temporale

temporal_distances_sorted = sorted(temporal_distances)
total_temporal_distances = len(temporal_distances_sorted)

if total_temporal_distances % 2 == 1:
    middle_index = total_temporal_distances // 2
    median_temporal_distance = temporal_distances_sorted[middle_index]
else:
    middle_index1 = total_temporal_distances // 2 - 1
    middle_index2 = total_temporal_distances // 2

    median_temporal_distance = (temporal_distances_sorted[middle_index1] + temporal_distances_sorted[middle_index2]) / 2

average_temporal_distance = sum(temporal_distances) / len(temporal_distances)

temporal_statistics = {
    "number_of_pairs": len(temporal_distances),
    "average_distance_days": round(average_temporal_distance, 2),
    "median_distance_days": round(median_temporal_distance, 2),
    "minimum_distance_days": round(min(temporal_distances), 2),
    "maximum_distance_days": round(max(temporal_distances), 2),

    "same_day": {
        "count": same_day,
        "percentage": round(same_day / len(temporal_distances) * 100, 2)
    },

    "one_to_seven_days": {
        "count": one_to_seven_days,
        "percentage": round(one_to_seven_days / len(temporal_distances) * 100, 2)
    },

    "eight_to_thirty_days": {
        "count": eight_to_thirty_days,
        "percentage": round(eight_to_thirty_days / len(temporal_distances) * 100, 2)
    },

    "thirty_one_to_ninety_days": {
        "count": thirty_one_to_ninety_days,
        "percentage": round(thirty_one_to_ninety_days / len(temporal_distances) * 100, 2)
    },

    "more_than_ninety_days": {
        "count": more_than_ninety_days,
        "percentage": round(more_than_ninety_days / len(temporal_distances) * 100, 2)
    }
}

with open("similarity/json/temporal_distribution/temporal_statistics.json", "w", encoding="utf-8") as file:
    json.dump(temporal_statistics, file, indent=4, ensure_ascii=False)

# distributia dist Levenshtein
plt.figure(figsize=(10, 6))

plt.hist(
    normalized_edit_distances,
    bins=30,
    edgecolor="black"
)

plt.title("Distributia distantei Levenshtein normalizate")
plt.xlabel("Distanta Levenshtein normalizata")
plt.ylabel("Numar de perechi")

plt.tight_layout()
plt.savefig("similarity/graphical_representation/value_distributions/normalized_edit_distance_distribution.png")
plt.close()

#distributia similaritatii multilingve
plt.figure(figsize=(10, 6))

plt.hist(
    multilingual_similarities,
    bins=30,
    edgecolor="black"
)

plt.axvline(0.75, linestyle="--", label="Threshold 0.75")
plt.axvline(0.90, linestyle="--", label="Threshold 0.90")

plt.title("Distributia similaritatii pentru stirile in limbi diferite")
plt.xlabel("Cosine similarity")
plt.ylabel("Numar de perechi")

plt.legend()
plt.tight_layout()
plt.savefig("similarity/graphical_representation/different_language/multilingual_similarity_distribution.png")
plt.close()

similarity_0_50 = 0
similarity_50_60 = 0
similarity_60_70 = 0
similarity_70_75 = 0
similarity_75_80 = 0
similarity_80_85 = 0
similarity_85_90 = 0
similarity_90_95 = 0
similarity_95_100 = 0

for similarity in multilingual_similarities:
    if similarity < 0.50:
        similarity_0_50 = similarity_0_50 + 1
    elif similarity < 0.60:
        similarity_50_60 = similarity_50_60 + 1
    elif similarity < 0.70:
        similarity_60_70 = similarity_60_70 + 1
    elif similarity < 0.75:
        similarity_70_75 = similarity_70_75 + 1
    elif similarity < 0.80:
        similarity_75_80 = similarity_75_80 + 1
    elif similarity < 0.85:
        similarity_80_85 = similarity_80_85 + 1
    elif similarity < 0.90:
        similarity_85_90 = similarity_85_90 + 1
    elif similarity < 0.95:
        similarity_90_95 = similarity_90_95 + 1
    else:
        similarity_95_100 = similarity_95_100 + 1

multilingual_similarity_intervals = [
    "< 0.50",
    "0.50-0.60",
    "0.60-0.70",
    "0.70-0.75",
    "0.75-0.80",
    "0.80-0.85",
    "0.85-0.90",
    "0.90-0.95",
    "0.95-1.00"
]

multilingual_similarity_counts = [
    similarity_0_50,
    similarity_50_60,
    similarity_60_70,
    similarity_70_75,
    similarity_75_80,
    similarity_80_85,
    similarity_85_90,
    similarity_90_95,
    similarity_95_100
]

multilingual_similarity_percentages = []

for count in multilingual_similarity_counts:
    percentage = count / len(multilingual_similarities) * 100
    multilingual_similarity_percentages.append(percentage)

plt.figure(figsize=(12, 7))
plt.bar(multilingual_similarity_intervals, multilingual_similarity_percentages)

plt.title("Distributia valorilor de similaritate pentru limbi diferite")
plt.xlabel("Interval de similaritate")
plt.ylabel("Procent din perechile in limbi diferite")

plt.xticks(rotation=30, ha="right")

for index in range(len(multilingual_similarity_percentages)):
    plt.text(
        index,
        multilingual_similarity_percentages[index],
        str(round(multilingual_similarity_percentages[index], 2)) + "%",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.savefig("similarity/graphical_representation/different_language/multilingual_similarity_intervals.png")
plt.close()

# doar pt limbi diferite
different_language_category_names = [
    "possible_literal_translation",
    "similar_cross_language_news",
    "different_cross_language_news"
]

different_language_category_counts = [
    possible_literal_translation,
    similar_cross_language_news,
    different_cross_language_news
]

different_language_total = (
    possible_literal_translation
    + similar_cross_language_news
    + different_cross_language_news
)

different_language_category_percentages = []

for count in different_language_category_counts:
    percentage = count / different_language_total * 100
    different_language_category_percentages.append(percentage)

plt.figure(figsize=(10, 6))
plt.bar(
    different_language_category_names,
    different_language_category_percentages
)

plt.title("Categoriile de similaritate pentru stirile in limbi diferite")
plt.xlabel("Categorie")
plt.ylabel("Procent din perechile in limbi diferite")

plt.xticks(rotation=25, ha="right")

for index in range(len(different_language_category_percentages)):
    plt.text(
        index,
        different_language_category_percentages[index],
        str(round(different_language_category_percentages[index], 2)) + "%",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.savefig("similarity/graphical_representation/different_language/different_language_categories.png")
plt.close()

# cele mai frecvente combinatii de limbi
language_pair_items = []

for language_pair in language_pair_counts:
    language_pair_items.append({
        "language_pair": language_pair,
        "count": language_pair_counts[language_pair]
    })

language_pair_items = sorted(
    language_pair_items,
    key=lambda elem: elem["count"],
    reverse=True
)

top_language_pairs = language_pair_items[:15]

language_pair_names = []
language_pair_values = []

for elem in top_language_pairs:
    language_pair_names.append(elem["language_pair"])
    language_pair_values.append(elem["count"])

plt.figure(figsize=(12, 7))
plt.bar(language_pair_names, language_pair_values)

plt.title("Cele mai frecvente combinatii de limbi")
plt.xlabel("Pereche de limbi")
plt.ylabel("Numar de perechi")

plt.xticks(rotation=30, ha="right")

for index in range(len(language_pair_values)):
    plt.text(
        index,
        language_pair_values[index],
        str(language_pair_values[index]),
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.savefig("similarity/graphical_representation/different_language/top_language_pairs.png")
plt.close()

with open("similarity/json/value_distributions/language_pair_distribution.json", "w", encoding="utf-8") as file:
    json.dump(language_pair_items, file, indent=4, ensure_ascii=False)

#dist temporal limbi dif 
plt.figure(figsize=(10, 6))

plt.hist(
    different_language_temporal_distances,
    bins=30,
    edgecolor="black"
)

plt.title("Distributia distantelor temporale pentru limbi diferite")
plt.xlabel("Distanta temporala in zile")
plt.ylabel("Numar de perechi")

plt.tight_layout()
plt.savefig("similarity/graphical_representation/different_language/temporal_distance_different_language.png")
plt.close()

different_language_temporal_distances_30_days = []

for distance in different_language_temporal_distances:
    if distance <= 30:
        different_language_temporal_distances_30_days.append(distance)

plt.figure(figsize=(10, 6))

plt.hist(
    different_language_temporal_distances_30_days,
    bins=30,
    edgecolor="black"
)

plt.title("Distante temporale de maximum 30 de zile pentru limbi diferite")
plt.xlabel("Distanta temporala in zile")
plt.ylabel("Numar de perechi")

plt.tight_layout()
plt.savefig("similarity/graphical_representation/different_language/temporal_distance_different_language_30_days.png")
plt.close()

# calcule cu Pandas dist Levenshtein
normalized_edit_distance_series = pd.Series(normalized_edit_distances)

normalized_edit_distance_statistics = {
    "count": int(normalized_edit_distance_series.count()),
    "mean": round(normalized_edit_distance_series.mean(), 4),
    "standard_deviation": round(normalized_edit_distance_series.std(), 4),
    "minimum": round(normalized_edit_distance_series.min(), 4),
    "percentile_25": round(normalized_edit_distance_series.quantile(0.25), 4),
    "median": round(normalized_edit_distance_series.median(), 4),
    "percentile_75": round(normalized_edit_distance_series.quantile(0.75), 4),
    "maximum": round(normalized_edit_distance_series.max(), 4)
}

#similaritate multilingva
multilingual_similarity_series = pd.Series(multilingual_similarities)

multilingual_similarity_statistics = {
    "count": int(multilingual_similarity_series.count()),
    "mean": round(multilingual_similarity_series.mean(), 4),
    "standard_deviation": round(multilingual_similarity_series.std(), 4),
    "minimum": round(multilingual_similarity_series.min(), 4),
    "percentile_25": round(multilingual_similarity_series.quantile(0.25), 4),
    "median": round(multilingual_similarity_series.median(), 4),
    "percentile_75": round(multilingual_similarity_series.quantile(0.75), 4),
    "maximum": round(multilingual_similarity_series.max(), 4)
}

# dist temporala a stirilor pt limbi diferite
different_language_temporal_series = pd.Series(
    different_language_temporal_distances
)

different_language_temporal_statistics = {
    "count": int(different_language_temporal_series.count()),
    "mean_days": round(different_language_temporal_series.mean(), 2),
    "standard_deviation_days": round(different_language_temporal_series.std(), 2),
    "minimum_days": round(different_language_temporal_series.min(), 2),
    "percentile_25_days": round(different_language_temporal_series.quantile(0.25), 2),
    "median_days": round(different_language_temporal_series.median(), 2),
    "percentile_75_days": round(different_language_temporal_series.quantile(0.75), 2),
    "maximum_days": round(different_language_temporal_series.max(), 2)
}

value_distribution_statistics = {
    "same_language_pairs": len(normalized_edit_distances),
    "different_language_pairs": len(multilingual_similarities),

    "normalized_edit_distance": normalized_edit_distance_statistics,

    "multilingual_similarity": multilingual_similarity_statistics,

    "different_language_temporal_distance": different_language_temporal_statistics,

    "different_language_categories": {
        "possible_literal_translation": {
            "count": possible_literal_translation,
            "percentage": round(possible_literal_translation / different_language_total * 100, 2)
        },

        "similar_cross_language_news": {
            "count": similar_cross_language_news,
            "percentage": round(similar_cross_language_news / different_language_total * 100, 2)
        },

        "different_cross_language_news": {
            "count": different_cross_language_news,
            "percentage": round(different_cross_language_news / different_language_total * 100, 2)
        }
    }
}

with open("similarity/json/value_distributions/value_distribution_statistics.json", "w", encoding="utf-8") as file:
    json.dump(value_distribution_statistics, file, indent=4, ensure_ascii=False)

plt.figure(figsize=(8, 6))

plt.boxplot(
    multilingual_similarities,
    vert=True
)

plt.title("Boxplot pentru similaritatea stirilor in limbi diferite")
plt.ylabel("Cosine similarity")

plt.tight_layout()
plt.savefig("similarity/graphical_representation/different_language/multilingual_similarity_boxplot.png")
plt.close()