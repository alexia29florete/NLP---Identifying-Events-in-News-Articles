import json
import os
import nltk
from nltk.tokenize import sent_tokenize

data = []
with open("ai_classification_cleaned.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        data.append(json.loads(line))

for elem in data:
    content1 = elem["content1"]
    content2 = elem["content2"]
    length1 = len(content1)
    length2 = len(content2)
    elem["length1"] = length1
    elem["length2"] = length2
    
    maximum_length = max(length1, length2)
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

os.makedirs("length_correlation", exist_ok=True)
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

