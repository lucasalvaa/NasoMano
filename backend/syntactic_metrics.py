from __future__ import annotations

import asyncio
import logging
import re
from typing import TypedDict

from textstat import textstat

# To build the docker image
# from .lang_tool_singleton import get_languagetool_instance
from lang_tool_singleton import get_languagetool_instance

logger = logging.getLogger(__name__)

# The word count threshold beyond which a prompt is considered "too long"
# when calculating the Complexity-Length Score.
WC_MAX = 60.0

# The divisor used to normalize the Gunning-Fog index to the range [0, 1].
GFI_NORMALIZATION_DIVISOR = 20.0

# Formatting score weights
CAPITALIZATION_WEIGHT = 0.4
PUNCTUATION_WEIGHT = 0.4
LAYOUT_WEIGHT = 0.2


class SyntacticMetrics(TypedDict):
    complexity_length_score: float
    grammatical_correctness_score: float | None
    readability_score: float
    formatting_score: float
    prompt_quality_score: float | None


class SyntacticMetricsEvaluator:
    def __init__(self) -> None:
        """
        Initialize the connection to the LanguageTool instance.
        """
        self.lang_tool = get_languagetool_instance()

    async def evaluate(self, prompt: str) -> SyntacticMetrics:
        """
        Calculate five syntactic metrics defined in the study by Della
        Porta et al.

        Parameters:
            prompt (str): The prompt to be evaluated.

        Raises:
            ValueError: se il prompt è vuoto o non è una stringa valida.
        """
        if not prompt or not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("The prompt provided is empty.")

        return await asyncio.to_thread(self._evaluate_sync, prompt)

    def _evaluate_sync(self, prompt: str) -> SyntacticMetrics:
        cls = self._calculate_cls(prompt)
        g_score = self._calculate_g(prompt)
        c_score = self._calculate_c(prompt)
        f_score = self._calculate_f(prompt)

        # If LanguageTool did not respond, g_score is None.
        # The Prompt Quality Score depends on g_score, so it cannot
        # be calculated reliably and is in turn propagated as None.
        pq_score = (
            round((g_score + f_score + c_score) / 3, 4)
            if g_score is not None
            else None
        )

        return {
            "complexity_length_score": cls,
            "grammatical_correctness_score": g_score,
            "readability_score": c_score,
            "formatting_score": f_score,
            "prompt_quality_score": pq_score,
        }

    def _calculate_cls(self, prompt: str) -> float:
        """
        Calculate the **Complexity-Length Score** (CLS) of a given prompt
        using the formula:

        *CLS* = 1 - min(1, ((*WC* / *WC_max*) + (*GFI* / 20)) / 2)

        Where *WC*: word count; *WC_max*: length threshold; *GFI*: Gunning
        Fog Index.
        """
        wc = textstat.lexicon_count(prompt, removepunct=True)
        gfi = textstat.gunning_fog(prompt)
        inner_term = ((wc / WC_MAX) + (gfi / GFI_NORMALIZATION_DIVISOR)) / 2.0
        cls = 1.0 - min(1.0, inner_term)

        return round(cls, 4)

    def _calculate_g(self, prompt: str) -> float | None:
        """
        Calculate *Grammatical correctness* (G) of a given prompt using
        the formula:

        *G* = 1 - (*n_matches* / max(1, *n_words*))

        Where *n_matches*: grammar/spelling issues; *n_words*: word count.

        Returns:
            The score, or None if the LanguageTool server could not be reached.
        """
        n_words = textstat.lexicon_count(prompt, removepunct=True)

        try:
            matches = self.lang_tool.check(prompt)
        except Exception:
            logger.exception(
                "LanguageTool check failed; grammatical correctness score "
                "unavailable for this prompt."
            )
            return None

        n_matches = len(matches)
        g_score = 1.0 - (n_matches / max(1, n_words))

        return round(g_score, 4)

    def _calculate_c(self, prompt: str) -> float:
        """
        Calculate *Readability* (C) of a given prompt using the Flesch
        Reading Ease score.

        The score is normalized to a 0.0 - 1.0 range (where 1.0 is
        maximum readability). Standard Flesch Reading Ease can
        occasionally exceed 100 or drop below 0 for extreme texts, so the
        final output is clamped strictly between 0 and 1.
        """
        raw_score = textstat.flesch_reading_ease(prompt)
        normalized_score = max(0.0, min(1.0, raw_score / 100.0))
        return round(normalized_score, 4)

    def _calculate_f(self, prompt: str) -> float:
        """
        Calculate *Formatting* (F) score as a normalized combination of
        punctuation, capitalization and layout indicators.
        """
        prompt = prompt.strip()

        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', prompt) if s.strip()]
        if not sentences:
            sentences = [prompt]

        # 1. Capitalization Indicator [0-1]: quante frasi iniziano con
        # lettera maiuscola.
        capitalized_sentences = sum(1 for s in sentences if s[0].isupper())
        cap_score = capitalized_sentences / len(sentences)

        # 2. Punctuation Indicator [0-1]: quante frasi terminano con
        # punteggiatura.
        punctuated_sentences = sum(1 for s in sentences if s[-1] in ".!?")
        punct_score = punctuated_sentences / len(sentences)

        # 3. Layout Indicator [0-1]: presenza di a-capo o liste.
        # Nota: una singola regex combinata per tutti e tre i pattern non
        # è soggetta a backtracking catastrofico (nessun quantificatore
        # annidato), quindi non introduce un rischio ReDoS.
        has_layout = bool(re.search(r'\n|- |\* |\d+\.', prompt))
        layout_score = 1.0 if has_layout else 0.0

        f_score = (
            (cap_score * CAPITALIZATION_WEIGHT)
            + (punct_score * PUNCTUATION_WEIGHT)
            + (layout_score * LAYOUT_WEIGHT)
        )
        return round(max(0.0, min(1.0, f_score)), 4)