import json
import os
import pickle
import time


input_file = "ai_classification_cleaned.jsonl"

output_folder = "missing_pairs_analysis"

summary_file = os.path.join(output_folder, "summary.txt")
duplicates_file = os.path.join(output_folder, "duplicate_pairs.jsonl")
id_mapping_file = os.path.join(output_folder, "id_mapping.json")
reverse_id_mapping_file = os.path.join(output_folder, "index_to_id.json")
benchmark_jsonl_file = os.path.join(output_folder, "benchmark_structure.jsonl")
benchmark_pkl_file = os.path.join(output_folder, "benchmark_structure.pkl")
benchmark_report_file = os.path.join(output_folder, "storage_comparison.txt")

os.makedirs(output_folder, exist_ok=True)


all_ids = set()
existing_pairs = set()
duplicate_pairs = []
number_of_rows = 0


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
            duplicate_pairs.append({"id1": id1, "id2": id2, "normalized_id1": pair[0], "normalized_id2": pair[1]})
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


all_ids = sorted(all_ids)

id_mapping = {}

for index in range(len(all_ids)):
    article_id = all_ids[index]
    id_mapping[article_id] = index + 1


with open(id_mapping_file, "w", encoding="utf-8") as file:
    json.dump(id_mapping, file, indent=4)


reverse_id_mapping = {}

for article_id in id_mapping:
    index = id_mapping[article_id]
    reverse_id_mapping[str(index)] = article_id


with open(reverse_id_mapping_file, "w", encoding="utf-8") as file:
    json.dump(reverse_id_mapping, file, indent=4)


existing_index_pairs = set()

for id1, id2 in existing_pairs:
    index1 = id_mapping[id1]
    index2 = id_mapping[id2]

    if index1 < index2:
        pair = (index1, index2)
    else:
        pair = (index2, index1)

    existing_index_pairs.add(pair)


benchmark_pair_count = 1000000

processed_pairs = 0
benchmark_structure = []

start_time = time.time()


for index1 in range(1, number_of_articles + 1):
    values = []
    index2 = index1 + 1

    while index2 <= number_of_articles:
        pair = (index1, index2)

        if pair in existing_index_pairs:
            values.append(True)
        else:
            values.append(False)

        processed_pairs = processed_pairs + 1

        if processed_pairs == benchmark_pair_count:
            break

        index2 = index2 + 1

    if len(values) > 0:
        benchmark_structure.append({index1: values})

    if processed_pairs == benchmark_pair_count:
        break


end_time = time.time()

benchmark_time = end_time - start_time


json_start_time = time.time()

with open(benchmark_jsonl_file, "w", encoding="utf-8") as file:
    for item in benchmark_structure:
        file.write(json.dumps(item) + "\n")

json_end_time = time.time()

json_write_time = json_end_time - json_start_time


pkl_start_time = time.time()

with open(benchmark_pkl_file, "wb") as file:
    pickle.dump(benchmark_structure, file, protocol=pickle.HIGHEST_PROTOCOL)

pkl_end_time = time.time()

pkl_write_time = pkl_end_time - pkl_start_time


jsonl_size_bytes = os.path.getsize(benchmark_jsonl_file)
pkl_size_bytes = os.path.getsize(benchmark_pkl_file)

jsonl_size_mb = jsonl_size_bytes / (1024 * 1024)
pkl_size_mb = pkl_size_bytes / (1024 * 1024)

jsonl_bytes_per_pair = jsonl_size_bytes / processed_pairs
pkl_bytes_per_pair = pkl_size_bytes / processed_pairs

estimated_jsonl_size_bytes = jsonl_bytes_per_pair * number_of_possible_pairs
estimated_pkl_size_bytes = pkl_bytes_per_pair * number_of_possible_pairs

estimated_jsonl_size_gb = estimated_jsonl_size_bytes / (1024 ** 3)
estimated_pkl_size_gb = estimated_pkl_size_bytes / (1024 ** 3)

average_time_per_pair = benchmark_time / processed_pairs
estimated_total_seconds = average_time_per_pair * number_of_possible_pairs
estimated_total_hours = estimated_total_seconds / 3600
estimated_total_days = estimated_total_hours / 24


with open(benchmark_report_file, "w", encoding="utf-8") as file:
    file.write("Compact Pair Structure Benchmark\n")
    file.write("-------------------------------\n\n")
    file.write("Benchmark pairs: " + str(processed_pairs) + "\n")
    file.write("Benchmark generation time: " + str(benchmark_time) + " seconds\n")
    file.write("Average generation time per pair: " + str(average_time_per_pair) + " seconds\n\n")

    file.write("JSONL\n")
    file.write("-------------------------------\n")
    file.write("Write time: " + str(json_write_time) + " seconds\n")
    file.write("Size: " + str(jsonl_size_mb) + " MB\n")
    file.write("Bytes per pair: " + str(jsonl_bytes_per_pair) + "\n")
    file.write("Estimated full size: " + str(estimated_jsonl_size_gb) + " GB\n\n")

    file.write("PKL\n")
    file.write("-------------------------------\n")
    file.write("Write time: " + str(pkl_write_time) + " seconds\n")
    file.write("Size: " + str(pkl_size_mb) + " MB\n")
    file.write("Bytes per pair: " + str(pkl_bytes_per_pair) + "\n")
    file.write("Estimated full size: " + str(estimated_pkl_size_gb) + " GB\n\n")

    file.write("Estimated full generation time\n")
    file.write("-------------------------------\n")
    file.write("Seconds: " + str(estimated_total_seconds) + "\n")
    file.write("Hours: " + str(estimated_total_hours) + "\n")
    file.write("Days: " + str(estimated_total_days) + "\n")