import json
import os


input_file = "ai_classification_cleaned.jsonl"

output_folder = "missing_pairs_analysis"
summary_file = os.path.join(output_folder, "summary.txt")
duplicates_file = os.path.join(output_folder, "duplicate_pairs.jsonl")

os.makedirs(output_folder, exist_ok=True)


all_ids = set()
existing_pairs = set()
duplicate_pairs = []

number_of_rows = 0


with open(input_file, "r", encoding="utf-8") as file:
    for line in file:
        data = json.loads(line)

        number_of_rows = number_of_rows + 1

        id1 = data["id1"]
        id2 = data["id2"]

        all_ids.add(id1)
        all_ids.add(id2)

        if id1 < id2:
            pair = (id1, id2)
        else:
            pair = (id2, id1)

        if pair in existing_pairs:
            duplicate_pairs.append({
                "id1": id1,
                "id2": id2,
                "normalized_id1": pair[0],
                "normalized_id2": pair[1]
            })
        else:
            existing_pairs.add(pair)


number_of_articles = len(all_ids)
number_of_existing_pairs = len(existing_pairs)

number_of_possible_pairs = number_of_articles * (number_of_articles - 1) // 2
number_of_missing_pairs = number_of_possible_pairs - number_of_existing_pairs


with open(summary_file, "w", encoding="utf-8") as file:
    file.write("Missing Pairs Analysis\n")
    file.write("-------------------------------\n\n")

    file.write("Rows in dataset: " + str(number_of_rows) + "\n")
    file.write("Unique articles: " + str(number_of_articles) + "\n")
    file.write("Unique existing pairs: " + str(number_of_existing_pairs) + "\n")
    file.write("Duplicate pair occurrences: " + str(len(duplicate_pairs)) + "\n")
    file.write("Possible pairs: " + str(number_of_possible_pairs) + "\n")
    file.write("Missing pairs: " + str(number_of_missing_pairs) + "\n")


with open(duplicates_file, "w", encoding="utf-8") as file:
    for pair in duplicate_pairs:
        file.write(json.dumps(pair) + "\n")