import json
import os
import pickle
import time


input_file = "ai_classification_cleaned.jsonl"

output_folder = "missing_pairs_analysis"

id_mapping_file = os.path.join(output_folder, "id_mapping.json")
reverse_id_mapping_file = os.path.join(output_folder, "index_to_id.json")
missing_pairs_file = os.path.join(output_folder, "missing_pairs.pkl")
summary_file = os.path.join(output_folder, "missing_pairs_summary.txt")

os.makedirs(output_folder, exist_ok=True)


all_ids = set()
existing_pairs = set()

number_of_rows = 0
duplicate_pair_occurrences = 0


start_time = time.time()


# ---------------------------------------------------------
# 1. Read all IDs and existing pairs
# ---------------------------------------------------------

with open(input_file, "r", encoding="utf-8") as file:
    for line in file:
        data = json.loads(line)

        number_of_rows = number_of_rows + 1

        id1 = str(data["id1"])
        id2 = str(data["id2"])

        all_ids.add(id1)
        all_ids.add(id2)

        if id1 < id2:
            pair = (id1, id2)
        else:
            pair = (id2, id1)

        if pair in existing_pairs:
            duplicate_pair_occurrences = duplicate_pair_occurrences + 1
        else:
            existing_pairs.add(pair)


# ---------------------------------------------------------
# 2. Sort IDs lexicographically
# ---------------------------------------------------------

all_ids = sorted(all_ids)

number_of_articles = len(all_ids)


# ---------------------------------------------------------
# 3. Create ID -> compact index mapping
# ---------------------------------------------------------

id_mapping = {}

for index in range(len(all_ids)):
    article_id = all_ids[index]
    id_mapping[article_id] = index + 1


# ---------------------------------------------------------
# 4. Create reverse mapping
# ---------------------------------------------------------

reverse_id_mapping = {}

for article_id in id_mapping:
    index = id_mapping[article_id]
    reverse_id_mapping[str(index)] = article_id


with open(id_mapping_file, "w", encoding="utf-8") as file:
    json.dump(id_mapping, file)


with open(reverse_id_mapping_file, "w", encoding="utf-8") as file:
    json.dump(reverse_id_mapping, file)


# ---------------------------------------------------------
# 5. Convert existing pairs to compact indices
# ---------------------------------------------------------

existing_index_pairs = set()

for id1, id2 in existing_pairs:
    index1 = id_mapping[id1]
    index2 = id_mapping[id2]

    if index1 > index2:
        index1, index2 = index2, index1

    existing_index_pairs.add((index1, index2))


# ---------------------------------------------------------
# 6. General statistics
# ---------------------------------------------------------

number_of_existing_pairs = len(existing_index_pairs)
number_of_possible_pairs = number_of_articles * (number_of_articles - 1) // 2
number_of_missing_pairs = number_of_possible_pairs - number_of_existing_pairs


# ---------------------------------------------------------
# 7. Generate ONLY missing pairs
#
# Structure written to PKL:
#
# {1: [2, 3, 4, 6, ...]}
# {2: [3, 4, 5, ...]}
# ...
#
# Each pickle.dump stores one article at a time.
#
# This avoids keeping all ~16 billion relations in RAM.
# ---------------------------------------------------------

generated_missing_pairs = 0

generation_start_time = time.time()


with open(missing_pairs_file, "wb") as file:
    for index1 in range(1, number_of_articles):
        missing_for_article = []

        for index2 in range(index1 + 1, number_of_articles + 1):
            if (index1, index2) not in existing_index_pairs:
                missing_for_article.append(index2)
                generated_missing_pairs = generated_missing_pairs + 1

        if len(missing_for_article) > 0:
            pickle.dump({index1: missing_for_article}, file, protocol=pickle.HIGHEST_PROTOCOL)


generation_end_time = time.time()

generation_time = generation_end_time - generation_start_time


# ---------------------------------------------------------
# 8. File size
# ---------------------------------------------------------

pkl_size_bytes = os.path.getsize(missing_pairs_file)
pkl_size_mb = pkl_size_bytes / (1024 * 1024)
pkl_size_gb = pkl_size_bytes / (1024 ** 3)


end_time = time.time()

total_time = end_time - start_time


# ---------------------------------------------------------
# 9. Summary
# ---------------------------------------------------------

with open(summary_file, "w", encoding="utf-8") as file:
    file.write("Missing Pair Representation\n")
    file.write("-------------------------------\n\n")

    file.write("Rows in dataset: " + str(number_of_rows) + "\n")
    file.write("Unique articles: " + str(number_of_articles) + "\n")
    file.write("Duplicate pair occurrences: " + str(duplicate_pair_occurrences) + "\n\n")

    file.write("All possible unordered pairs: " + str(number_of_possible_pairs) + "\n")
    file.write("Existing pairs excluded: " + str(number_of_existing_pairs) + "\n")
    file.write("Missing pairs expected: " + str(number_of_missing_pairs) + "\n")
    file.write("Missing pairs generated: " + str(generated_missing_pairs) + "\n\n")

    file.write("PKL size: " + str(pkl_size_mb) + " MB\n")
    file.write("PKL size: " + str(pkl_size_gb) + " GB\n")
    file.write("Generation time: " + str(generation_time) + " seconds\n")
    file.write("Total time: " + str(total_time) + " seconds\n")