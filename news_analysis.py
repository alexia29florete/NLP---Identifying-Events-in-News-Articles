import json
import os
from langdetect import detect, DetectorFactory
from rapidfuzz.distance import Levenshtein
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import matplotlib.pyplot as plt
from datetime import datetime

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

        elem["normalized_edit_distance"] = None
        elem["multilingual_similarity"] = multilingual_similarity

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

print("trivial_duplicate:", trivial_dup, "pairs", round(trivial_dup / len(similar_news) * 100, 2), "%")
print("near_duplicate:", near_dup, "pairs", round(near_dup / len(similar_news) * 100, 2), "%")
print("different_news_same_event:", diff, "pairs", round(diff / len(similar_news) * 100, 2), "%")
print("possible_literal_translation:", possible_literal_translation, "pairs", round(possible_literal_translation / len(similar_news) * 100, 2), "%")
print("similar_cross_language_news:", similar_cross_language_news, "pairs", round(similar_cross_language_news / len(similar_news) * 100, 2), "%")
print("different_cross_language_news:", different_cross_language_news, "pairs", round(different_cross_language_news / len(similar_news) * 100, 2), "%")
    
with open("similar_news_analysis.jsonl", "w", encoding="utf-8") as file:
    for elem in analyzed_data:
        file.write(json.dumps(elem, ensure_ascii=False) + "\n")

# creez folderul similarity daca nu exista
os.makedirs("similarity", exist_ok=True)

with open("similarity/trivial_duplicate.jsonl", "w", encoding="utf-8") as file:
    for elem in trivial_duplicate_pairs:
        file.write(json.dumps(elem, ensure_ascii=False) + "\n")

with open("similarity/near_duplicate.jsonl", "w", encoding="utf-8") as file:
    for elem in near_duplicate_pairs:
        file.write(json.dumps(elem, ensure_ascii=False) + "\n")

with open("similarity/different_news_same_event.jsonl", "w", encoding="utf-8") as file:
    for elem in different_news_same_event_pairs:
        file.write(json.dumps(elem, ensure_ascii=False) + "\n")

with open("similarity/possible_literal_translation.jsonl", "w", encoding="utf-8") as file:
    for elem in possible_literal_translation_pairs:
        file.write(json.dumps(elem, ensure_ascii=False) + "\n")

with open("similarity/similar_cross_language_news.jsonl", "w", encoding="utf-8") as file:
    for elem in similar_cross_language_news_pairs:
        file.write(json.dumps(elem, ensure_ascii=False) + "\n")

with open("similarity/different_cross_language_news.jsonl", "w", encoding="utf-8") as file:
    for elem in different_cross_language_news_pairs:
        file.write(json.dumps(elem, ensure_ascii=False) + "\n")

with open("similarity/similar_news_analysis.jsonl", "w", encoding="utf-8") as file:
    for elem in analyzed_data:
        file.write(json.dumps(elem, ensure_ascii=False) + "\n")

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
plt.savefig("similarity/similarity_percentages.png")
plt.close()

# grafic pentru distanta temporala
plt.figure(figsize=(10, 6))

plt.hist(temporal_distances, bins=30, edgecolor="black")

plt.title("Distributia distantelor temporale pentru stirile similare")
plt.xlabel("Distanta temporala in zile")
plt.ylabel("Numar de perechi")

plt.tight_layout()
plt.savefig("similarity/temporal_distance_distribution.png")
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
plt.savefig("similarity/temporal_distance_distribution_30_days.png")
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
plt.savefig("similarity/temporal_distance_intervals.png")
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

with open("similarity/temporal_statistics.json", "w", encoding="utf-8") as file:
    json.dump(temporal_statistics, file, indent=4, ensure_ascii=False)