"""
Extractive Text Summarizer Mini Application
Subject: Text and Speech Analysis
Description: Implements classic frequency-based extractive text summarization.
             Identifies key sentences by calculating term frequency distributions,
             scores candidate sentences, and selects top sentences while preserving flow.
"""

import collections
import re
import sys
import os

# Built-in fallback English stopwords in case NLTK data isn't downloaded yet
FALLBACK_STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
    "at", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s",
    "t", "can", "will", "just", "don", "should", "now", "also", "well"
}

try:
    import nltk
    # Try downloading quietly if missing
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords
    from nltk.tokenize import sent_tokenize, word_tokenize
    STOP_WORDS = set(stopwords.words("english"))
    USE_NLTK = True
except Exception:
    # Graceful fallback to regex tokenization
    USE_NLTK = False
    STOP_WORDS = FALLBACK_STOPWORDS


def split_sentences(text):
    """Splits text into sentences using NLTK or regex fallback."""
    if USE_NLTK:
        try:
            return sent_tokenize(text)
        except Exception:
            pass
    # Regex fallback: split on period, exclamation, or question mark followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def split_words(text):
    """Tokenizes text into words using NLTK or regex fallback."""
    if USE_NLTK:
        try:
            return word_tokenize(text.lower())
        except Exception:
            pass
    # Regex fallback
    return re.findall(r'\b[a-zA-Z0-9_\'-]+\b', text.lower())


def summarize_text(text, num_sentences=3):
    """
    Extractive summarization algorithm:
    1. Sentence Tokenization
    2. Word Tokenization & Frequency Count
    3. Normalization of Word Frequencies
    4. Sentence Scoring based on Word Weights
    5. Ranking and Original-Order Assembly
    """
    if not text.strip():
        print("[WARNING] Input text is empty.")
        return "", {}

    sentences = split_sentences(text)
    total_sentences = len(sentences)

    if total_sentences == 0:
        return "", {}

    if total_sentences <= num_sentences:
        print(f"[INFO] Document has only {total_sentences} sentences. Returning entire text.")
        return text, {
            "orig_sentences": total_sentences,
            "summary_sentences": total_sentences,
            "orig_words": len(split_words(text)),
            "summary_words": len(split_words(text)),
            "compression_ratio": 0.0
        }

    # 1. Calculate word frequencies (excluding stopwords & punctuation)
    words = split_words(text)
    meaningful_words = [w for w in words if w.isalnum() and w not in STOP_WORDS]

    if not meaningful_words:
        print("[WARNING] No meaningful content words detected.")
        return " ".join(sentences[:num_sentences]), {}

    freq_dist = collections.Counter(meaningful_words)
    max_freq = max(freq_dist.values())

    # Normalize frequencies (0.0 to 1.0)
    normalized_weights = {word: count / max_freq for word, count in freq_dist.items()}

    # 2. Score each sentence
    sentence_scores = {}
    for idx, sentence in enumerate(sentences):
        sent_words = split_words(sentence)
        score = 0.0
        for w in sent_words:
            if w in normalized_weights:
                score += normalized_weights[w]
        # Normalize slightly by length to prevent bias towards unnaturally long sentences
        word_count = len(sent_words)
        if word_count > 0:
            sentence_scores[idx] = score / (word_count ** 0.5)

    # 3. Pick top N sentences
    top_indices = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
    # Sort selected indices back to original chronological order
    top_indices.sort()

    summary_sentences = [sentences[i] for i in top_indices]
    summary_text = " ".join(summary_sentences)

    # 4. Compute statistics
    orig_words_count = len(words)
    summary_words_count = len(split_words(summary_text))
    compression = (1.0 - (summary_words_count / max(1, orig_words_count))) * 100.0

    stats = {
        "orig_sentences": total_sentences,
        "summary_sentences": len(summary_sentences),
        "orig_words": orig_words_count,
        "summary_words": summary_words_count,
        "compression_ratio": compression,
        "top_keywords": freq_dist.most_common(5)
    }

    return summary_text, stats


SAMPLE_ARTICLE = """
Natural language processing (NLP) is an interdisciplinary subfield of computer science and linguistics.
It is primarily concerned with giving computers the ability to support and manipulate human speech and text.
The history of NLP generally started in the 1950s, although work can be found from earlier periods.
In 1950, Alan Turing published an article titled 'Computing Machinery and Intelligence' which proposed what is now called the Turing test as a criterion of intelligence.
Modern NLP algorithms are based on statistical machine learning, deep learning, and transformer architectures.
Text and speech analysis algorithms allow systems to transcribe spoken audio, summarize long articles, detect sentiments, and extract critical keywords.
Speech recognition systems convert analog sound pressure waves into digital waveforms, extracting acoustic features like Mel-frequency cepstral coefficients.
These representations enable machines to understand spoken accents and linguistic patterns across diverse applications.
Through text summarization, students and researchers can quickly extract core findings from massive technical publications.
Overall, text and speech analysis forms the backbone of modern interactive conversational agents and search engines.
"""


def main():
    print("=" * 60)
    print("      EXTRACTIVE TEXT SUMMARIZER - MINI LAB")
    print("=" * 60)
    print("Extracts key sentences based on Term Frequency & Sentence Weighting.")
    print(f"Backend Tokenizer: {'NLTK Tokenizer' if USE_NLTK else 'Built-in Regex (Fallback)'}")

    while True:
        print("\nMain Menu:")
        print("1. Summarize custom text input")
        print("2. Summarize sample article on NLP & Speech Analysis")
        print("3. Summarize a text file (.txt)")
        print("4. Exit")

        choice = input("\nEnter choice (1-4): ").strip()

        if choice == "1":
            print("\nEnter or paste your text (End with an empty line or press Ctrl+Z / Enter twice):")
            lines = []
            while True:
                line = input()
                if not line:
                    break
                lines.append(line)
            raw_text = " ".join(lines).strip()
            if not raw_text:
                print("[WARNING] Text cannot be empty.")
                continue

            try:
                num_s = int(input("Enter desired number of summary sentences (e.g. 2, 3, 5): ") or "3")
            except ValueError:
                num_s = 3

            summary, stats = summarize_text(raw_text, num_sentences=num_s)
            print("\n" + "=" * 60)
            print("                 GENERATED SUMMARY")
            print("=" * 60)
            print(summary)
            print("=" * 60)
            if stats:
                print(f"Original: {stats['orig_sentences']} sentences, {stats['orig_words']} words")
                print(f"Summary : {stats['summary_sentences']} sentences, {stats['summary_words']} words")
                print(f"Compression Reduction: {stats['compression_ratio']:.1f}%")

        elif choice == "2":
            try:
                num_s = int(input("Enter number of summary sentences (default: 3): ") or "3")
            except ValueError:
                num_s = 3

            summary, stats = summarize_text(SAMPLE_ARTICLE, num_sentences=num_s)
            print("\n" + "=" * 60)
            print("                 GENERATED SUMMARY")
            print("=" * 60)
            print(summary)
            print("=" * 60)
            if stats:
                print(f"Original: {stats['orig_sentences']} sentences, {stats['orig_words']} words")
                print(f"Summary : {stats['summary_sentences']} sentences, {stats['summary_words']} words")
                print(f"Compression Reduction: {stats['compression_ratio']:.1f}%")
                if "top_keywords" in stats:
                    print("Top Extracted Terms:", ", ".join(f"{w} ({c})" for w, c in stats["top_keywords"]))

        elif choice == "3":
            filepath = input("Enter path to .txt file: ").strip().strip("\"'")
            if not os.path.exists(filepath):
                print(f"[ERROR] File not found: {filepath}")
                continue
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                file_text = f.read()

            try:
                num_s = int(input("Enter number of summary sentences (default: 3): ") or "3")
            except ValueError:
                num_s = 3

            summary, stats = summarize_text(file_text, num_sentences=num_s)
            print("\n" + "=" * 60)
            print("                 GENERATED SUMMARY")
            print("=" * 60)
            print(summary)
            print("=" * 60)
            if stats:
                print(f"Original: {stats['orig_sentences']} sentences, {stats['orig_words']} words")
                print(f"Summary : {stats['summary_sentences']} sentences, {stats['summary_words']} words")
                print(f"Compression Reduction: {stats['compression_ratio']:.1f}%")

        elif choice == "4":
            print("\nExiting Text Summarizer. Goodbye!\n")
            break

        else:
            print("[ERROR] Invalid choice. Please enter 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()
