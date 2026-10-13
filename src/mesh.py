"""MeSH vocabulary and keyword extraction.

Every MeSH descriptor has a preferred name plus entry terms (synonyms, abbreviations, inverted forms).
All of them are loaded into a token trie that maps a word sequence to the descriptor ID, so
"breast cancer", "Breast Neoplasms" and "Cancer of the Breast" all count as the same keyword.
"""
import gzip
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter

from .fetch_data import DATA, MESH

TERMS_CACHE = DATA / "mesh_terms.json.gz"
TOKEN = re.compile(r"[A-Za-z0-9]+")

# Single words whose usual meaning in trial text is not their MeSH sense: "will" is not Volition,
# a study "arm" is not the body part, "gold standard" is not the metal. Picked by reviewing the
# 400 most frequent single-word matches in 20,000 trials. Multi-word terms still match.
STOPWORDS = {
    "ability", "administration", "affect", "affects", "arm", "arms", "association", "back",
    "community", "compliance", "consumption", "culture", "electronic", "future", "goal", "goals",
    "gold", "history", "indicators", "intention", "inventory", "judgment", "lead", "light",
    "literature", "maintenance", "measure", "measures", "minor", "nature", "needs", "opinion",
    "pilot", "play", "power", "production", "record", "records", "report", "reports", "rest",
    "role", "schedule", "self", "sensitivity", "thought", "time", "understanding", "will", "work",
}


def load_terms():
    """Return {descriptor UI: [preferred name, entry term, ...]} for topical descriptors.

    DescriptorClass 1 is topical. Classes 2-4 are publication types, check tags
    (Male, Female, Humans) and geographic names, which are not useful topic keywords.
    """
    if TERMS_CACHE.exists():
        with gzip.open(TERMS_CACHE, "rt") as f:
            return json.load(f)
    terms = {}
    with gzip.open(MESH) as f:
        for _, rec in ET.iterparse(f):
            if rec.tag != "DescriptorRecord":
                continue
            if rec.get("DescriptorClass") == "1":
                name = rec.findtext("DescriptorName/String")
                strings = {t.text for t in rec.iterfind("ConceptList/Concept/TermList/Term/String")}
                terms[rec.findtext("DescriptorUI")] = [name] + sorted(strings - {name})
            rec.clear()
    with gzip.open(TERMS_CACHE, "wt") as f:
        json.dump(terms, f)
    return terms


def uninvert(term):
    """MeSH writes many terms inverted: "Neoplasms, Breast" -> "Breast Neoplasms"."""
    parts = term.split(", ")
    return f"{parts[1]} {parts[0]}" if len(parts) == 2 else term


class Extractor:
    def __init__(self, terms):
        self.names = {ui: strings[0] for ui, strings in terms.items()}
        # Two tries: case-insensitive for normal terms, case-sensitive for all-caps
        # abbreviations, so the word "who" in "patients who..." is not read as WHO (World Health Organization).
        self.lower, self.exact = {}, {}
        for ui, strings in terms.items():
            for s in strings:
                for variant in {s, uninvert(s)}:
                    toks = TOKEN.findall(variant)
                    if len(toks) == 1:
                        word = toks[0]
                        if variant.isupper():
                            self._add(self.exact, toks, ui)
                            continue
                        if len(word) < 3 or not any(c.isalpha() for c in word) or word.lower() in STOPWORDS:
                            continue
                    if toks:
                        self._add(self.lower, [t.lower() for t in toks], ui)

    @staticmethod
    def _add(trie, toks, ui):
        node = trie
        for t in toks:
            node = node.setdefault(t, {})
        node.setdefault("", ui)  # "" marks the end of a term; first descriptor wins on clashes

    @staticmethod
    def _walk(trie, toks, i):
        """Longest term starting at position i, as (end position, descriptor UI)."""
        best, node = (i, None), trie
        for j in range(i, len(toks)):
            node = node.get(toks[j])
            if node is None:
                break
            if "" in node:
                best = (j + 1, node[""])
        return best

    def extract(self, text):
        """Count MeSH descriptors in text, scanning left to right with longest match."""
        toks = TOKEN.findall(text)
        low = [t.lower() for t in toks]
        found, i = Counter(), 0
        while i < len(toks):
            end, ui = max(self._walk(self.lower, low, i), self._walk(self.exact, toks, i), key=lambda m: m[0])
            if ui:
                found[ui] += 1
                i = end
            else:
                i += 1
        return found
