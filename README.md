# Dataset Cleaning

Before performing the similarity analysis, the dataset was cleaned to remove invalid articles.

An article was considered invalid when:

*   **High Corruption Index:** The proportion of Unicode replacement characters (`�`) exceeds a predefined threshold ($\tau = 10\%$);
*   **Boilerplate/Source Code Contamination:** The content contains script remnants (e.g., JavaScript fragments, tracking parameters, or advertisement injection tokens) rather than prose.
*   **Missing or Corrupted Payloads:** The document body consists exclusively of an external video reference or metadata link.
*   **Null Representations:** Both the title and body text fields evaluate to null or empty strings.

> **Handling Asymmetry:** If a document body is empty but its corresponding title remains structurally valid, the title is promoted to act as the primary textual representation for downstream tasks. To preserve pair integrity, if either document in a paired sample $(d_1, d_2)$ is flagged as invalid, the entire pair is pruned from the dataset.

### Cleaning results

```text
Invalid articles: 786
Initial pairs: 100000
Removed pairs: 433
Remaining pairs: 99567
```

*Methodological Note:* A primary challenge identified during language classification via `langdetect` was the systematic misclassification of heavily corrupted text payloads (dominated by  characters) as Bengali (`bn`), rather than generating an `unknown` or low-confidence exception.
For `langdetect`, an `unknown` language was the lack of information, for instance no context, spaces or \n or \t. Although, there were some articles that had no context (so detected as unknown language), the title could have given us enough information in order to continue our research on the provieded dataset. Another problem was that there were some articles that only had a link to a video or some JavaScript code that we couldn't have processed as NLP, and we have decided to drop them out from our database.

# News Similarity Analysis

## Overview

This project analyzes multilingual news pairs classified as similar.

The analysis focuses on:

- textual similarity for articles written in the same language;
- semantic similarity for articles written in different languages;
- the most frequent language combinations;
- the temporal distance between similar news articles.

## Same-Language Similarity

For articles written in the same language, textual similarity was measured using normalized Levenshtein distance.

The distance is calculated as:

```python
edit_distance = Levenshtein.distance(content1, content2)
normalized_edit_distance = edit_distance / max(len(content1), len(content2))
```

A value close to `0` indicates nearly identical texts, while a value close to `1` indicates substantial textual differences.

The following operational categories were used:

| Normalized distance | Category |
|---:|---|
| `0.00–0.05` | `trivial_duplicate` |
| `0.05–0.20` | `near_duplicate` |
| `> 0.20` | `different_news_same_event` |

These categories describe surface-level similarity. Levenshtein distance does not directly measure semantic equivalence.

## Normalized Levenshtein Distance Distribution

![Normalized Levenshtein distance distribution](similarity/graphical_representation/value_distributions/normalized_edit_distance_distribution.png)

The distribution contains two major regions.

The first region is concentrated close to `0`. These pairs are likely to represent:

- exact duplicates;
- syndicated content;
- minimally edited copies;
- articles differing only through formatting or punctuation.

The second and larger region is concentrated approximately around `0.70–0.80`.

This indicates that many pairs labeled as similar are lexically and structurally different, despite being related at the event level. These articles may contain:

- different wording;
- different paragraph organization;
- additional contextual information;
- different quotations;
- distinct journalistic framing.

The shape of the distribution suggests that the `Yes` class contains at least two different phenomena:

1. direct textual duplication;
2. semantic relatedness without strong lexical overlap.

Therefore, normalized Levenshtein distance is suitable for duplicate detection, but it is insufficient as a general measure of news-event similarity.


## Cross-Language Semantic Similarity

For articles written in different languages, multilingual sentence embeddings were generated using:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

The semantic representations were compared using cosine similarity.

```python
embedding1 = model.encode(content1, convert_to_tensor=True)
embedding2 = model.encode(content2, convert_to_tensor=True)

multilingual_similarity = cos_sim(embedding1, embedding2).item()
```

The following operational thresholds were used:

| Cosine similarity | Category |
|---:|---|
| `< 0.75` | `different_cross_language_news` |
| `0.75–0.90` | `similar_cross_language_news` |
| `>= 0.90` | `possible_literal_translation` |

These thresholds are analytical decisions rather than empirically discovered boundaries.


## Multilingual Similarity Distribution

For articles written in different languages, multilingual embeddings and cosine similarity were used.

![Multilingual similarity distribution](similarity/graphical_representation/different_language/multilingual_similarity_distribution.png)

The distribution is strongly concentrated between `0.75` and `0.90`, indicating that most cross-language pairs classified as similar share substantial semantic content.

A smaller but relevant group exceeds `0.90`, representing very high semantic similarity.

The thresholds used in the analysis are:

- below `0.75`: `different_cross_language_news`;
- between `0.75` and `0.90`: `similar_cross_language_news`;
- at least `0.90`: `possible_literal_translation`.

However, a score above `0.90` does not necessarily prove that one article is a literal translation of the other. It only indicates very high semantic similarity.


## Cross-Language Similarity Intervals

The similarity scores were also divided into intervals.

![Cross-language similarity intervals](similarity/graphical_representation/different_language/multilingual_similarity_intervals.png)

Most values are concentrated in the following intervals:

- `31.10%` between `0.80` and `0.85`;
- `26.66%` between `0.85` and `0.90`;
- `17.95%` between `0.75` and `0.80`;
- `10.17%` between `0.90` and `0.95`.

Only a small percentage of pairs have similarity scores below `0.60` or above `0.95`.

The largest group, representing `31.10%` of the cross-language pairs, has similarity scores between `0.80` and `0.85`.

These pairs are generally semantically close and are likely to describe the same event or strongly related information. They usually share the main facts, actors and topic, but may differ in wording, additional details, emphasis, or article structure.

Therefore, this interval indicates high semantic similarity, but not necessarily literal translation or identical content.


## Most Frequent Language Combinations

The following graph shows the most frequent language combinations among cross-language pairs.

![Most frequent language combinations](similarity/graphical_representation/different_language/top_language_pairs.png)

The most frequent combination is `en-ro`, with `519` pairs, followed by `hr-ro`, with `319` pairs.

Other frequent combinations include:

- `de-en`;
- `fr-ro`;
- `en-hr`;
- `ro-ru`;
- `en-es`;
- `en-pt`;
- `en-fr`.

Romanian appears in many of the most frequent combinations. This means that the multilingual analysis is influenced more strongly by language pairs that include Romanian.

The dataset is not equally balanced across all language combinations, so this should be considered when interpreting the overall results.


## Temporal Distance Distribution

The temporal distance was calculated for all article pairs classified as similar.

![Temporal distance distribution](similarity/graphical_representation/temporal_distance_intervals.png)

The results are:

- `53.35%` were published on the same day;
- `17.37%` were published within 1–7 days;
- `12.01%` were published within 8–30 days;
- `10.72%` were published within 31–90 days;
- `6.55%` were published more than 90 days apart.

More than `70%` of the similar pairs were published within one week.

This suggests that temporal proximity is strongly associated with news similarity.

Pairs separated by more than 90 days may represent:

- retrospective articles;
- recurring events;
- reused content;


## Extreme Similarity Pairs

Evaluating extreme similarity instances exposes critical edge cases and systemic limitations within standard NLP tracking baselines:

### Case Alpha: The False Cross-Lingual Positive (Maximum Similarity Score)
*   **Cosine Score:** $0.9893$ | **Temporal Delta:** $\approx 0.17\text{ days}$
*   **Linguistic Mapping Failure:** Classified by the pipeline as a cross-lingual pair (Croatian `hr` vs. Macedonian `mk`).
*   **Diagnostic Inspection:** Human-in-the-loop validation revealed the documents were identical Serbian texts, with one rendered in Latin script and the other transliterated into Cyrillic. The off-the-shelf language identifier failed to recognize the orthographic shift, mistakenly attributing the script change to a language change. The near-perfect embedding score reflects script invariance rather than exceptional cross-lingual translation modeling.

### Case Beta: The Structural Noise Outlier (Minimum Similarity Score)
*   **Cosine Score:** $-0.1083$ | **Temporal Delta:** $\approx 101.70\text{ days}$
*   **Linguistic Mapping:** French (`fr`) vs. Romanian (`ro`).
*   **Diagnostic Inspection:** While both documents share a baseline semantic entity (e.g., "Donald Trump/Harvard University"), severe format discrepancies led to an artificial drop in vector alignment. Document $d_1$ was an compressed, abstract summary, while document $d_2$ suffered from uncleaned HTML boilerplate, styling elements, and WordPress metadata injections. This demonstrates that document-level pooling remains highly sensitive to uncleaned structural noise.

# Length correlation analysis

To determine whether article length is associated with the similarity label, the analysis compared pairs classified as `Yes` and `No` using:

- average article length;
- absolute and relative length differences;
- the proportion of pairs with exactly or approximately equal lengths;
- the frequency of texts truncated at `1000` characters;
- differences between the numbers of detected sentences.

The article language is detected using `langdetect`. When the detected language is supported by NLTK, the corresponding language-specific sentence tokenizer is used. For unsupported languages, sentence boundaries are estimated using a regular-expression fallback.

This approach is methodologically more appropriate than applying the default English tokenizer to every language.

For each pair, the relative length difference was calculated as:

```python
relative_length_difference = abs(length1 - length2) / max(length1, length2)
```

## Average Article Length Distribution

![Percentage distribution of average article lengths](length_correlation/graphical_representation/average_length_distribution.png)

The distribution is strongly concentrated near `1000` characters for both classes. The concentration is higher for `Yes` pairs, indicating that similar pairs more frequently contain two long texts.

However, this result must be interpreted carefully because `1000` characters appears to be a technical truncation limit rather than a natural article-length boundary. Consequently, the peak close to `1000` partly reflects dataset construction rather than an intrinsic characteristic of similar news articles.

A value close to `0` indicates that the two texts have similar lengths, while a value close to `1` indicates a substantial imbalance.

## Relative Length Difference

![Relative length difference by classification](length_correlation/graphical_representation/relative_length_difference_boxplot.png)

The median relative length differences are:

- `Yes`: `1.44%`;
- `No`: `6.00%`.

This shows that pairs classified as similar generally contain texts with more closely matched lengths. The wider box and higher median for the `No` class indicate greater structural variation among dissimilar pairs.

Nevertheless, both classes contain extreme cases with relative differences close to `1`. Therefore, article length alone cannot reliably determine whether two articles describe the same event.

## Similar-Length Thresholds

![Percentage of pairs with similar lengths](length_correlation/graphical_representation/similar_length_percentages.png)

The percentage of pairs satisfying increasingly permissive length-similarity criteria is:

| Length criterion | `Yes` | `No` | Difference |
|---|---:|---:|---:|
| Exactly equal length | `43.54%` | `35.20%` | `8.34` percentage points |
| Relative difference `<= 5%` | `56.75%` | `48.28%` | `8.47` percentage points |
| Relative difference `<= 10%` | `63.34%` | `55.73%` | `7.61` percentage points |
| Relative difference `<= 20%` | `71.96%` | `65.11%` | `6.85` percentage points |

Across all thresholds, `Yes` pairs are more likely to have similar lengths. The difference between the classes is consistent but moderate.

The high percentages observed for the `No` class are also important: almost half of the dissimilar pairs differ in length by no more than `5%`. Thus, similar length is neither a sufficient nor a decisive indicator of semantic similarity.

## Relative Sentence-Count Difference

![Relative sentence count difference by classification](length_correlation/graphical_representation/relative_sentence_difference_boxplot.png)

The median relative sentence-count difference is `33.33%` for both `Yes` and `No` pairs. The two distributions overlap strongly, although the `No` class shows slightly greater variability.

This suggests that sentence count is a weaker structural indicator than character length. Articles about the same event may use different journalistic styles, including:

- many short sentences;
- fewer long sentences;
- different quotation and paragraph structures;
- different levels of contextual detail.

Sentence segmentation also introduces additional noise because supported languages use language-specific NLTK tokenizers, while unsupported languages use a regular-expression fallback.

## Absolute Sentence Count Difference

![Absolute sentence-count difference](length_correlation/graphical_representation/sentence_count_difference_categories.png)

The absolute sentence-count difference measures the concrete difference between the two articles in each pair:

```python
sentence_count_difference = abs(sentence_count1 - sentence_count2)
```

Unlike the relative sentence-count difference, this measure is expressed directly in numbers of sentences and is therefore easier to interpret.

The distribution shows that most article pairs differ by only a small number of sentences:

| Sentence-count difference | `Yes` | `No` |
|---:|---:|---:|
| `0` | `20.82%` | `17.62%` |
| `1` | `21.63%` | `21.79%` |
| `2` | `17.29%` | `17.41%` |
| `3` | `13.35%` | `13.23%` |
| `4` | `9.83%` | `9.48%` |
| `5` | `6.63%` | `6.71%` |
| `6` | `3.82%` | `4.63%` |
| `7` | `2.36%` | `2.94%` |
| `8` | `1.36%` | `1.79%` |
| `9` | `0.66%` | `1.04%` |
| `10` | `0.47%` | `0.78%` |
| `> 10` | `1.78%` | `2.58%` |

Pairs with exactly the same number of sentences represent `20.82%` of `Yes` pairs and `17.62%` of `No` pairs. The proportion is therefore `3.20` percentage points higher for similar pairs.

The analysis of values close to zero confirmed that the peak in the previous relative-difference histogram was caused almost entirely by exact equality:

```text
Exact zero Yes: 7312
Near zero Yes: 0
Exact zero No: 11358
Near zero No: 1
```

Thus, the first interval did not contain a large hidden group of small positive differences. It primarily represented article pairs with exactly the same number of detected sentences.

The cumulative results also show a small structural advantage for the `Yes` class:

- `42.45%` of `Yes` pairs and `39.41%` of `No` pairs differ by at most one sentence;
- `59.74%` of `Yes` pairs and `56.82%` of `No` pairs differ by at most two sentences;
- `73.09%` of `Yes` pairs and `70.05%` of `No` pairs differ by at most three sentences.

The average sentence-count difference is:

- `Yes`: `2.6693` sentences;
- `No`: `3.0264` sentences.

Therefore, `No` pairs differ by approximately `0.3571` more sentences on average. This difference is observable but small, and the distributions still overlap considerably.

## Word Count Analysis

Word counts were computed for each article using a Unicode-aware regular expression. The analysis includes both individual article size and the difference between the two articles in each pair.

### Word Counts per Article

![Percentage distribution of word counts per article](length_correlation/graphical_representation/word_count_per_article_distribution.png)

The individual article-size statistics are:

| Statistic | `Yes` | `No` |
|---|---:|---:|
| Article occurrences | `70238` | `128896` |
| Total words | `10018594` | `17941175` |
| Mean words per article | `142.6378` | `139.1911` |
| Median words per article | `148` | `146` |

The total number of words is substantially higher for the `No` class because the dataset contains many more `No` pairs. Therefore, total counts should not be interpreted as evidence that `No` articles are naturally longer.

The means and medians provide a fairer comparison. Articles from `Yes` pairs contain only `3.4467` more words on average than articles from `No` pairs. This corresponds to a difference of approximately `2.5%`.

The two distributions are strongly overlapping and are both concentrated roughly between `130` and `180` words. Consequently, the individual number of words in an article provides little separation between the two classes.

The concentration in this interval is also affected by the technical `1000`-character truncation limit. The observed word counts describe the available text fragments rather than necessarily the full original articles.

## Absolute Word Count Difference

![Absolute word-count difference](length_correlation/graphical_representation/word_count_difference_categories.png)

The absolute word-count difference is defined as:

```python
word_count_difference = abs(word_count1 - word_count2)
```

The distribution was divided into concrete intervals:

| Word-count difference | `Yes` | `No` |
|---:|---:|---:|
| `0–10` | `45.76%` | `37.19%` |
| `11–25` | `21.21%` | `23.08%` |
| `26–50` | `12.92%` | `15.50%` |
| `51–75` | `7.40%` | `8.88%` |
| `76–100` | `4.42%` | `5.63%` |
| `> 100` | `8.29%` | `9.71%` |

The most important difference appears in the `0–10` interval. Almost half of the `Yes` pairs have article lengths that differ by at most ten words, compared with `37.19%` of the `No` pairs.

For every interval above ten words, the `No` percentage is higher. This indicates that dissimilar pairs are more frequently structurally imbalanced.

The average word-count differences are:

- `Yes`: `38.0306` words;
- `No`: `45.6460` words.

Therefore, `No` pairs differ by approximately `7.6154` more words on average. Relative to the `Yes` mean, the average difference is approximately `20%` larger for the `No` class.

This result is more informative than the individual article-size comparison. The two classes contain articles of broadly similar length, but the two articles within a `Yes` pair tend to have more closely matched word counts.

## Length Analysis Conclusions

The length analysis supports the following conclusions:

1. **Similar pairs tend to have more closely matched lengths.** The `Yes` class has a lower median relative length difference and consistently higher percentages across the `5%`, `10%`, and `20%` similarity thresholds.
2. **The association is informative but weak.** Large portions of the `Yes` and `No` distributions overlap, so length features cannot independently classify article similarity.
3. **The `1000`-character truncation limit strongly influences the results.** A substantial proportion of exact-length matches is caused by both texts reaching the same technical limit.
4. **Sentence count provides less separation than character length.** Its median relative difference is identical across the two classes, and the distributions overlap substantially.
5. **Length should be treated as an auxiliary feature.** It may support semantic or lexical similarity measures, but it should not replace them.
6. **Articles from `Yes` and `No` pairs have similar individual sizes.** Their average word counts differ by only `3.4467` words per article, and their sentence-count means are almost identical.
7. **Within-pair differences are more informative than individual article size.** `Yes` pairs have smaller average word-count and sentence-count differences.
8. **Exact sentence-count equality is more frequent for similar pairs.** It occurs in `20.82%` of `Yes` pairs and `17.62%` of `No` pairs.
9. **Small word-count differences are substantially more common for `Yes` pairs.** A difference of at most ten words occurs in `45.76%` of `Yes` pairs, compared with `37.19%` of `No` pairs.
10. **Large structural differences are more frequent for `No` pairs.** The `No` class has higher percentages in every word-difference interval above ten words and in the larger sentence-difference categories.