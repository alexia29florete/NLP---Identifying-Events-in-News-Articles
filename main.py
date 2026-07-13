import json

data = []
with open("ai_classification_initial.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        data.append(json.loads(line))

# Sanity Check

invalid_data = []
valid_data = []

for elem in data:
    classification = elem["classification_openai/gpt-oss-120b_v1"]

    if classification != "Yes" and classification != "No":
        invalid_data.append(elem)
    else:
        valid_data.append(elem)

# Salvez exemplele invalide

with open("invalid_classifications.jsonl", "w", encoding="utf-8") as file:
    for elem in invalid_data:
        file.write(json.dumps(elem, ensure_ascii=False) + "\n")


# adaug No la exemplul din invalid_data si il adaug in baza de date corecta
for elem in invalid_data:
    elem["classification_openai/gpt-oss-120b_v1"] = "No"
    valid_data.append(elem)

# ai_classification.jsonl fara exemplele invalide

with open("ai_classification.jsonl", "w", encoding="utf-8") as file:
    for elem in valid_data:
        file.write(json.dumps(elem, ensure_ascii=False) + "\n")

# Verificare corespondenta Yes -> 1 si No -> 0

correct_correlations = []
incorrect_correlations = []

for elem in valid_data:
    classification = elem["classification_openai/gpt-oss-120b_v1"]
    classified_value = elem["classification_openai/gpt-oss-120b_v1_classified"]

    if classification == "Yes" and classified_value == 1:
        correct_correlations.append(elem)

    elif classification == "No" and classified_value == 0:
        correct_correlations.append(elem)

    else:
        incorrect_correlations.append(elem)

print("Correct correlations:", len(correct_correlations))
print("Incorrect correlations:", len(incorrect_correlations))

# with open("incorrect_correlations.jsonl", "w", encoding="utf-8") as file:
#     for elem in incorrect_correlations:
#         file.write(json.dumps(elem, ensure_ascii=False) + "\n")