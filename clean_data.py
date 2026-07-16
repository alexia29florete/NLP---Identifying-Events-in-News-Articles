import json
import os
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0

def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

def empty_text(text):
    if text is None:
        return True

    text = text.strip()

    if text == "":
        return True

    return False

def corrupted_text(text):
    if text is None:
        return False

    text = text.strip()

    if text == "":
        return False

    replacement_characters = 0

    for character in text:
        if character == "�":
            replacement_characters = replacement_characters + 1

    replacement_percentage = replacement_characters / len(text)

    if replacement_percentage > 0.10:
        return True

    return False

def code_text(text):
    if text is None:
        return False

    text = text.strip().lower()

    if text == "":
        return False

    code_elements = 0

    if "window." in text:
        code_elements = code_elements + 1

    if "document." in text:
        code_elements = code_elements + 1

    if "queryselector" in text:
        code_elements = code_elements + 1

    if "settimeout" in text:
        code_elements = code_elements + 1

    if "adfoxcode" in text:
        code_elements = code_elements + 1

    if "yacontextcb" in text:
        code_elements = code_elements + 1

    if "containerid" in text:
        code_elements = code_elements + 1

    if "createelement" in text:
        code_elements = code_elements + 1

    starts_with_code = False

    if text.startswith("window."):
        starts_with_code = True

    if text.startswith("!function"):
        starts_with_code = True

    if text.startswith("function"):
        starts_with_code = True

    if text.startswith("<script"):
        starts_with_code = True

    if code_elements >= 2 and starts_with_code:
        return True
    
    return False

def video_text(text):
    if text is None:
        return False

    text = text.strip().lower()

    if text == "":
        return False

    words = text.split()

    if len(words) > 3:
        return False

    video_extensions = [
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".webm",
        ".mpeg",
        ".mpg"
    ]

    for extension in video_extensions:
        if extension in text:
            return True

    return False

def valid_text(text):
    if empty_text(text):
        return False

    if corrupted_text(text):
        return False

    if code_text(text):
        return False

    if video_text(text):
        return False

    for character in text:
        if character.isprintable() and character.isalnum():
            return True

    return False

def normalize_text(text):
    text = text.lower()
    text = " ".join(text.split())

    return text

os.makedirs("cleaning_results", exist_ok=True)
os.makedirs("cleaning_results/json", exist_ok=True)
os.makedirs("cleaning_results/reports", exist_ok=True)

data = []

with open("ai_classification.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        data.append(json.loads(line))


# ------articolele unice----------------------------------------------

articles = {}

for elem in data:
    id1 = elem["id1"]
    id2 = elem["id2"]

    if id1 not in articles:
        articles[id1] = {
            "title": elem["title1"],
            "content": elem["content1"],
            "region": elem["region1"]
        }

    if id2 not in articles:
        articles[id2] = {
            "title": elem["title2"],
            "content": elem["content2"],
            "region": elem["region2"]
        }


# ------articolele invalide si articolele cu limba unknown-----------

invalid_article_ids = set()
suspicious_articles = []

for article_id in articles:
    title = articles[article_id]["title"]
    content = articles[article_id]["content"]
    region = articles[article_id]["region"]

    content_is_empty = empty_text(content)
    content_is_corrupted = corrupted_text(content)
    content_is_code = code_text(content)
    content_is_video = video_text(content)

    title_is_valid = valid_text(title)

    content_language = detect_language(content)
    title_language = detect_language(title)

    invalid_article = False
    invalid_reason = ""

    # continut corupt -> elimin indiferent de titlu
    if content_is_corrupted:
        invalid_article = True
        invalid_reason = "corrupted_content"

    # continut format din cod JavaScript / reclama
    elif content_is_code:
        invalid_article = True
        invalid_reason = "code_content"

    # continut format doar dintr-un link catre un videoclip
    elif content_is_video:
        invalid_article = True
        invalid_reason = "video_only_content"

    # continut gol si titlu invalid
    elif content_is_empty and title_is_valid == False:
        invalid_article = True
        invalid_reason = "empty_content_and_invalid_title"

    if invalid_article:
        invalid_article_ids.add(article_id)

    if (invalid_article or content_language == "unknown" or title_language == "unknown" or content_language == "bn" or title_language == "bn"):
        suspicious_articles.append({
            "id": article_id,
            "region": region,
            "will_be_removed": invalid_article,
            "invalid_reason": invalid_reason,
            "title_language": title_language,
            "content_language": content_language,
            "title_length": len(title) if title is not None else 0,
            "content_length": len(content) if content is not None else 0,
            "title": title,
            "content": content
        })


with open("cleaning_results/json/suspicious_language_articles.jsonl", "w", encoding="utf-8") as file:
    for article in suspicious_articles:
        file.write(json.dumps(article, ensure_ascii=False) + "\n")

# ------elimin perechile care contin articole invalide---------------

cleaned_data = []
removed_pairs = []

for elem in data:
    id1 = elem["id1"]
    id2 = elem["id2"]

    if id1 in invalid_article_ids or id2 in invalid_article_ids:
        removed_pairs.append(elem)
    else:
        cleaned_data.append(elem)


with open("cleaning_results/json/removed_invalid_pairs.jsonl", "w", encoding="utf-8") as file:
    for elem in removed_pairs:
        file.write(json.dumps(elem, ensure_ascii=False) + "\n")


with open("ai_classification_cleaned.jsonl", "w", encoding="utf-8") as file:
    for elem in cleaned_data:
        file.write(json.dumps(elem, ensure_ascii=False) + "\n")


with open("cleaning_results/reports/cleaning_summary.txt", "w", encoding="utf-8") as file:
    file.write("Treshold: 10%\n")
    file.write("Invalid articles: " + str(len(invalid_article_ids)) + "\n")
    file.write("Initial pairs: " + str(len(data)) + "\n")
    file.write("Removed pairs: " + str(len(removed_pairs)) + "\n")
    file.write("Remaining pairs: " + str(len(cleaned_data)) + "\n")


# ------articolele unice dupa curatare-------------------------------

cleaned_articles = {}

for elem in cleaned_data:
    id1 = elem["id1"]
    id2 = elem["id2"]

    if id1 not in cleaned_articles:
        cleaned_articles[id1] = {
            "title": elem["title1"],
            "content": elem["content1"],
            "region": elem["region1"]
        }

    if id2 not in cleaned_articles:
        cleaned_articles[id2] = {
            "title": elem["title2"],
            "content": elem["content2"],
            "region": elem["region2"]
        }


# ------limbile articolelor pentru fiecare tara----------------------

languages_by_region = {}

for article_id in cleaned_articles:
    title = cleaned_articles[article_id]["title"]
    content = cleaned_articles[article_id]["content"]
    region = cleaned_articles[article_id]["region"]

    if empty_text(content):
        text = title
    else:
        text = content

    language = detect_language(text)

    if region == "ES" and language == "id":
        language = "es"

    if region == "US" and language == "id":
        language = "en"

    if region == "HU" and language == "id":
        language = "hu"

    if language == "unknown" or language == "bn":
        continue

    if region not in languages_by_region:
        languages_by_region[region] = {}

    if language not in languages_by_region[region]:
        languages_by_region[region][language] = 0

    languages_by_region[region][language] += 1


# ------numarul si procentul articolelor-----------------------------

region_language_percentages = {}

for region in languages_by_region:
    total_articles = 0

    for language in languages_by_region[region]:
        total_articles += languages_by_region[region][language]

    region_language_percentages[region] = {"total_articles": total_articles, "languages": {}}

    for language in languages_by_region[region]:
        count = languages_by_region[region][language]
        percentage = count / total_articles * 100

        region_language_percentages[region]["languages"][language] = {"count": count, "percentage": round(percentage, 2)}


with open("cleaning_results/json/languages_by_region.json", "w", encoding="utf-8") as file:
    json.dump(region_language_percentages, file, indent=4, ensure_ascii=False)