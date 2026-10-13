# 06 - Extractive Text Summarizer Application

## Overview
This application is part of the **Text and Speech Analysis** college lab curriculum. It demonstrates **Extractive Text Summarization**, an automated NLP technique that distills a large body of text into a concise summary by identifying and extracting the most informative, salient sentences directly from the source document.

---

## Theory & Algorithm
Unlike *abstractive summarization* (which paraphrases and rewrites sentences), *extractive summarization* scores existing sentences using statistical term distributions:

1. **Sentence & Word Tokenization**: Divides the document into sentence units and word tokens.
2. **Stopword Removal**: Eliminates grammatical glue words (e.g., "the", "is", "at") that do not carry semantic weight.
3. **Word Frequency Weighting**: Calculates the frequency distribution $f(w)$ for each word and normalizes by the maximum term frequency:
   $$W(w) = \frac{f(w)}{\max_{v} f(v)}$$
4. **Sentence Scoring**: Computes the score of sentence $S$ by summing the normalized weights of its constituent words, normalized by length:
   $$\text{Score}(S) = \frac{\sum_{w \in S} W(w)}{\sqrt{|S|}}$$
5. **Top-K Ranking & Chronological Order**: Selects the top $N$ ranked sentences and re-orders them according to their original position in the text to preserve conversational coherence.

---

## Features
- **Pure Python & NLTK Engine**: Uses NLTK tokenizers with a built-in fallback regex tokenizer and stopword lexicon in case corpora are not pre-downloaded.
- **Configurable Summary Length**: Choose the exact number of summary sentences desired.
- **Compression Metrics**: Reports original sentence/word counts, summary size, and percentage compression reduction.
- **Top Term Extraction**: Highlights the most frequent non-stopword keywords driving the summary.

---

## Prerequisites & Installation

1. Navigate to this folder:
   ```bash
   cd 06_Text_Summarizer
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## How to Run

```bash
python summarizer.py
```

### Example Usage:
```text
Main Menu:
1. Summarize custom text input
2. Summarize sample article on NLP & Speech Analysis
3. Summarize a text file (.txt)
4. Exit

Enter choice (1-4): 2
Enter number of summary sentences (default: 3): 2

============================================================
                 GENERATED SUMMARY
============================================================
Natural language processing (NLP) is an interdisciplinary subfield of computer science and linguistics. Text and speech analysis algorithms allow systems to transcribe spoken audio, summarize long articles, detect sentiments, and extract critical keywords.
============================================================
Original: 10 sentences, 154 words
Summary : 2 sentences, 35 words
Compression Reduction: 77.3%
Top Extracted Terms: speech (4), analysis (3), nlp (3), text (2), systems (2)
```

---

## Technical Notes & Limitations
- **Preserved Order**: Preserves chronological sentence ordering so the summary flows naturally.
- **Extractive vs Abstractive**: Extractive summarizers cannot generate novel sentence phrasing; they extract key verbatim sentences.
