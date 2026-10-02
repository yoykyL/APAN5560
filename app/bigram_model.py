"""Bigram language model, adapted from Module 1 Practical 2 (Word Sampling)."""

import random
import re
from collections import Counter, defaultdict


class BigramModel:
    def __init__(self, corpus, frequency_threshold=None):
        """
        Build bigram probabilities from a corpus.

        Args:
            corpus: a single string or a list of strings.
            frequency_threshold: drop words that appear fewer than this many
                times (None keeps every word).
        """
        if isinstance(corpus, str):
            corpus = [corpus]
        self.frequency_threshold = frequency_threshold
        self.vocab, self.bigram_probs = self._analyze_bigrams(" ".join(corpus))

    def _tokenize(self, text):
        """Lowercase the text and split it into words."""
        tokens = re.findall(r"\b\w+\b", text.lower())
        if not self.frequency_threshold:
            return tokens
        counts = Counter(tokens)
        return [t for t in tokens if counts[t] >= self.frequency_threshold]

    def _analyze_bigrams(self, text):
        """Compute p(w2 | w1) = C(w1, w2) / C(w1)."""
        words = self._tokenize(text)
        bigram_counts = Counter(zip(words[:-1], words[1:]))
        unigram_counts = Counter(words)

        bigram_probs = defaultdict(dict)
        for (w1, w2), count in bigram_counts.items():
            bigram_probs[w1][w2] = count / unigram_counts[w1]

        return list(unigram_counts.keys()), bigram_probs

    def generate_text(self, start_word, length=20):
        """Generate up to `length` words, starting from `start_word`."""
        current_word = start_word.lower()
        generated = [current_word]

        for _ in range(length - 1):
            next_words = self.bigram_probs.get(current_word)
            if not next_words:  # no known continuation, stop early
                break
            current_word = random.choices(
                list(next_words.keys()), weights=list(next_words.values())
            )[0]
            generated.append(current_word)

        return " ".join(generated)
