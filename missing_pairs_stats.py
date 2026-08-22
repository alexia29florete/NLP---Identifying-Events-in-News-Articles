import json
import os
import time
from itertools import combinations


input_file = "ai_classification_cleaned.jsonl"

output_folder = "missing_pairs_analysis"

summary_file = os.path.join(output_folder, "summary.txt")
duplicates_file = os.path.join(output_folder, "duplicate_pairs.jsonl")
benchmark_pairs_file = os.path.join(output_folder, "benchmark_100_pairs.jsonl")
benchmark_report_file = os.path.join(output_folder, "benchmark_report.txt")

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


all_ids = sorted(all_ids)

benchmark_pair_count = 1000000
generated_pairs = 0

start_time = time.time()


with open(benchmark_pairs_file, "w", encoding="utf-8") as file:
    for id1, id2 in combinations(all_ids, 2):

        if (id1, id2) not in existing_pairs:
            pair = {
                "id1": id1,
                "id2": id2
            }

            file.write(json.dumps(pair) + "\n")

            generated_pairs = generated_pairs + 1

            if generated_pairs == benchmark_pair_count:
                break


end_time = time.time()

total_time = end_time - start_time

if generated_pairs > 0:
    average_time_per_pair = total_time / generated_pairs
else:
    average_time_per_pair = 0


estimated_total_time_seconds = average_time_per_pair * number_of_missing_pairs
estimated_total_time_minutes = estimated_total_time_seconds / 60
estimated_total_time_hours = estimated_total_time_minutes / 60
estimated_total_time_days = estimated_total_time_hours / 24
estimated_total_time_years = estimated_total_time_days / 365


with open(benchmark_report_file, "w", encoding="utf-8") as file:
    file.write("Missing Pairs Benchmark\n")
    file.write("-------------------------------\n\n")

    file.write("Benchmark pairs: " + str(generated_pairs) + "\n")
    file.write("Total benchmark time: " + str(total_time) + " seconds\n")
    file.write("Average time per pair: " + str(average_time_per_pair) + " seconds\n\n")

    file.write("Estimated processing time for all missing pairs\n")
    file.write("-------------------------------\n")

    file.write("Seconds: " + str(estimated_total_time_seconds) + "\n")
    file.write("Minutes: " + str(estimated_total_time_minutes) + "\n")
    file.write("Hours: " + str(estimated_total_time_hours) + "\n")
    file.write("Days: " + str(estimated_total_time_days) + "\n")
    file.write("Years: " + str(estimated_total_time_years) + "\n")