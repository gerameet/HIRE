"""WordNet interface for semantic knowledge.

Provides access to WordNet synsets and hierarchical relationships.
Wrapper around NLTK's WordNet interface.
"""

import logging
from typing import List, Optional, Tuple, Set, Dict
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset

logger = logging.getLogger(__name__)


class WordNetKnowledge:
    """Access WordNet for semantic relationships."""

    def __init__(self, download_if_missing: bool = True):
        """Initialize WordNet interface.

        Args:
            download_if_missing: Automatically download WordNet data if missing.
        """
        if download_if_missing:
            self._ensure_data()

    def _ensure_data(self):
        """Ensure required NLTK data is available."""
        try:
            wn.ensure_loaded()
        except LookupError:
            logger.info("Downloading WordNet data...")
            nltk.download("wordnet")
            nltk.download("omw-1.4")

    def get_synsets(self, word: str, pos: Optional[str] = None) -> List[Synset]:
        """Get all synsets for a word.

        Args:
            word: The word to look up (e.g., "dog")
            pos: Part of speech (e.g., 'n' for noun). None for all.

        Returns:
            List of Synset objects.
        """
        # Replace spaces (if multi-word concept) with underscores
        word = word.replace(" ", "_")
        return wn.synsets(word, pos=pos)

    def get_hypernyms(self, synset: Synset) -> List[Synset]:
        """Get parent concepts (more general).

        Args:
            synset: Input synset

        Returns:
            List of immediate hypernyms.
        """
        return synset.hypernyms()

    def get_all_hypernyms(self, synset: Synset) -> Set[Synset]:
        """Get all ancestor concepts recursively."""
        hypernyms = set()
        for hyper in synset.hypernyms():
            hypernyms.add(hyper)
            hypernyms.update(self.get_all_hypernyms(hyper))
        return hypernyms

    def get_hyponyms(self, synset: Synset) -> List[Synset]:
        """Get child concepts (more specific).

        Args:
            synset: Input synset

        Returns:
            List of immediate hyponyms.
        """
        return synset.hyponyms()

    def get_holonyms(self, synset: Synset) -> List[Synset]:
        """Get 'part of' relationships (what this is a part of).

        Args:
            synset: Input synset

        Returns:
            List of member/part/substance holonyms.
        """
        return (
            synset.member_holonyms()
            + synset.part_holonyms()
            + synset.substance_holonyms()
        )

    def get_meronyms(self, synset: Synset) -> List[Synset]:
        """Get parts of this concept."""
        return (
            synset.member_meronyms()
            + synset.part_meronyms()
            + synset.substance_meronyms()
        )

    def semantic_similarity(self, word1: str, word2: str) -> float:
        """Compute max semantic similarity between two words.

        Uses path similarity between best matching synsets.

        Args:
            word1: First word
            word2: Second word

        Returns:
            Similarity score (0-1). 0 if no path found.
        """
        synsets1 = self.get_synsets(word1, pos=wn.NOUN)
        synsets2 = self.get_synsets(word2, pos=wn.NOUN)

        if not synsets1 or not synsets2:
            return 0.0

        max_sim = 0.0
        for s1 in synsets1:
            for s2 in synsets2:
                sim = s1.path_similarity(s2)
                if sim and sim > max_sim:
                    max_sim = sim
        return max_sim

    def is_hypernym(self, parent_word: str, child_word: str) -> bool:
        """Check if parent_word is an ancestor of child_word."""
        parents = self.get_synsets(parent_word, pos=wn.NOUN)
        children = self.get_synsets(child_word, pos=wn.NOUN)

        if not parents or not children:
            return False

        # Check if any parent synset is an ancestor of any child synset
        for child in children:
            ancestors = self.get_all_hypernyms(child)
            for parent in parents:
                if parent in ancestors:
                    return True
        return False
