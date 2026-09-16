import os
import jdk4py
import re
import textstat
import language_tool_python

# Imposta la variabile d'ambiente JAVA_HOME puntando al Java portatile del venv
os.environ["JAVA_HOME"] = str(jdk4py.JAVA_HOME)
os.environ["PATH"] = str(jdk4py.JAVA_HOME / "bin") + os.pathsep + os.environ.get("PATH", "")

class ContinuousMetricsEvaluator:
    def __init__(self):
        # Inizializza la connessione all'istanza locale o remota di LanguageTool
        self.lang_tool = language_tool_python.LanguageTool(
            'en-US',
            remote_server='http://127.0.0.1:8081/'
        )

    def evaluate(self, prompt: str) -> dict:
        """
        Calcola tutte le metriche continue per un dato prompt orchestrando
        i metodi privati della classe.
        """
        # Se il prompt è nullo o vuoto, restituiamo i valori massimi di default
        if not prompt or not isinstance(prompt, str) or prompt.strip() == "":
            return {
                "complexity_length_score": None,
                "grammatical_correctness_score": None,
                "readability_score": None,
                "formatting_score": None,
                "prompt_quality_score": None,
            }

        # 6. Complexity Length
        cls = self.__calculate_cls(prompt)
        CL_THRESHOLD = 0.75

        # 7. Grammatical Correctness
        g_score = self.__calculate_g(prompt)
        G_THRESHOLD = 0.9

        # 8. Readability (C)
        c_score = self.__calculate_c(prompt)
        # Un punteggio normalizzato < 0.5 (ovvero < 50 nel Flesch Reading Ease standard)
        # corrisponde a un testo di difficile lettura (livello college o superiore)
        C_THRESHOLD = 0.5

        # 9. Formatting (F)
        f_score = self.__calculate_f(prompt)
        F_THRESHOLD = 0.75  # Soglia sotto la quale il prompt viene considerato mal formattato

        # 10. Prompt Quality (PQS)
        pqs_score = round((g_score + f_score + c_score) / 3, 4)
        PQS_THRESHOLD = 0.7

        return {
            "complexity_length_score": cls,
            "grammatical_correctness_score": g_score,
            "readability_score": c_score,
            "formatting_score": f_score,
            "prompt_quality_score": round((g_score + f_score + c_score) / 3, 4)
        }

    def __calculate_cls(self, prompt: str) -> float:
        """
        Calculate the Complexity-Length Score (CLS) of a given prompt using the formula:
        CLS = 1 - min(1, ((WC / WC_max) + (GFI / 20)) / 2)
        Where WC: word count; WC_max: length threshold; GFI: Gunning Fog Index.
        """
        if not prompt or prompt.strip() == "":
            return 1.0

        # Length threshold constant
        WC_MAX = 60.0

        # Complexity-Length Score calculation
        wc = textstat.lexicon_count(prompt, removepunct=True)
        gfi = textstat.gunning_fog(prompt)
        inner_term = ((wc / WC_MAX) + (gfi / 20.0)) / 2.0
        cls = 1.0 - min(1.0, inner_term)

        return round(cls, 4)

    def __calculate_g(self, prompt: str) -> float:
        """
        Calculate Grammatical correctness (G) using the formula:
        G = 1 - (n_matches / max(1, n_words))
        Where n_matches: grammar/spelling issues; n_words: word count.
        """
        if not prompt or prompt.strip() == "":
            return 1.0

        # Grammatical Correctness Score calculation
        n_words = textstat.lexicon_count(prompt, removepunct=True)
        matches = self.lang_tool.check(prompt)
        n_matches = len(matches)
        g_score = 1.0 - (n_matches / max(1, n_words))

        return round(g_score, 4)

    def __calculate_c(self, prompt: str) -> float:
        """
        Calculate Readability (C) using the Flesch Reading Ease score.
        The score is normalized to a 0.0 - 1.0 range (where 1.0 is maximum readability).
        Standard Flesch Reading Ease can occasionally exceed 100 or drop below 0 for extreme texts,
        so we clamp the final output strictly between 0 and 1.
        """
        if not prompt or prompt.strip() == "":
            return 1.0

        raw_score = textstat.flesch_reading_ease(prompt)
        normalized_score = max(0.0, min(1.0, raw_score / 100.0))
        return round(normalized_score, 4)

    def __calculate_f(self, prompt: str) -> float:
        """
        Calculate Formatting (F) score as a normalized combination of
        punctuation (40%), capitalization (40%) and layout indicators (20%).
        """
        if not prompt or prompt.strip() == "":
            return 1.0

        prompt = prompt.strip()

        # Split text into sentences using a regular expression
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', prompt) if s.strip()]
        if not sentences:
            sentences = [prompt]

        # 1. Calculate Capitalization Indicator [0-1] by counting
        # how many sentences begin with a capital letter
        capitalized_sentences = sum(1 for s in sentences if s[0].isupper())
        cap_score = capitalized_sentences / len(sentences)

        # 2. Calculate Punctuation Indicator [0-1] by counting
        # how many sentences terminate with punctuation
        punctuated_sentences = sum(1 for s in sentences if s[-1] in ".!?")
        punct_score = punctuated_sentences / len(sentences)

        # 3. Calculate Layout Indicator [0-1] by analyzing the presence of newlines or lists
        has_layout = bool(re.search(r'\n|- |\* |\d+\.', prompt))
        layout_score = 1.0 if has_layout else 0.0

        # Compute the normalized combination using the following weights:
        # Capitalization 40%, Punctuation 40%, Layout 20%
        f_score = (cap_score * 0.4) + (punct_score * 0.4) + (layout_score * 0.2)
        return round(max(0.0, min(1.0, f_score)), 4)