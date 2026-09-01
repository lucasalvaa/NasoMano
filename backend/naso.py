import re
import textstat
import language_tool_python

import os
import jdk4py

# Imposta la variabile d'ambiente JAVA_HOME puntando al Java portatile del venv
os.environ["JAVA_HOME"] = str(jdk4py.JAVA_HOME)
os.environ["PATH"] = str(jdk4py.JAVA_HOME / "bin") + os.pathsep + os.environ.get("PATH", "")

class PromptSmellDetector:
    """
    Detect the presence of the following six “prompt smells” in English-Language prompts:
    1. Reasoning Suppression
    2. Lack of Self-Reflection
    3. Role Suppression
    4. Unspecified Output Structure
    5. Lack of Examples
    6. Complexity-Length
    """

    def __init__(self):
        # 1. REASONING SUPPRESSION (Classic CoT only)
        cot_patterns = [
            r"let'?s\s+(?:think|reason|solve|work)\s+step[\s-]by[\s-]step",
            r"think\s+(?:through\s+this\s+)?step[\s-]by[\s-]step",
            r"reason\s+step[\s-]by[\s-]step",
            r"\bfirst\b.*?\bthen\b.*?\bfinally\b",
            r"break\s+down\s+(?:the|this)\s+(?:problem|task|code)\s+step[\s-]by[\s-]step",
            r"walk\s+me\s+through\s+your\s+(?:thought\s+process|logic|reasoning)",
            r"explain\s+your\s+reasoning\s+(?:step[\s-]by[\s-]step|clearly)",
            r"work\s+through\s+this\s+(?:systematically|methodically|step[\s-]by[\s-]step)",
            r"take\s+a\s+deep\s+breath\s+and\s+think",
            r"show\s+your\s+(?:work|working|steps|thought\s+process)",
            r"think\s+logically\s+and\s+step[\s-]by[\s-]step"
        ]
        self.cot_regex = re.compile(r"|".join(cot_patterns), re.IGNORECASE | re.DOTALL)

        # 2. LACK OF SELF-REFLECTION
        self_reflection_patterns = [
            r"check\s+(?:to\s+see\s+)?if\s+(?:your|the)\s+(?:answer|code|solution|output)\s+is\s+correct",
            r"double[\s-]check\s+(?:your|the)\s+(?:work|code|answer|solution|output)",
            r"review\s+(?:the|your)\s+(?:code|answer|solution|response|output)(?:\s+before\s+replying)?",
            r"verify\s+(?:your|the)\s+(?:answer|code|solution|correctness)",
            r"validate\s+(?:the|your)\s+(?:code|solution|output)",
            r"make\s+sure\s+(?:there\s+are\s+no\s+(?:bugs|errors)|it\s+is\s+correct)",
            r"ensure\s+(?:that\s+)?(?:the\s+code\s+works|it\s+is\s+bug[\s-]free|correctness)",
            r"reflect\s+on\s+your\s+(?:answer|solution|output)",
            r"self[\s-](?:correct|debug|check|review|reflect)",
            r"critique\s+(?:your|the)\s+(?:solution|code)",
            r"audit\s+the\s+(?:code|output|solution)",
            r"test\s+(?:your|the)\s+(?:code|solution)\s+for\s+edge\s+cases"
        ]
        self.self_reflection_regex = re.compile(r"|".join(self_reflection_patterns), re.IGNORECASE)

        # 3. ROLE SUPPRESSION
        role_patterns = [
            r"\bact\s+as\s+(?:a|an)?\b",
            r"\byou\s+are\s+(?:a|an)?\b",
            r"\bas\s+an?\s+expert\s+in\b",
            r"\bpretend\s+to\s+be\b",
            r"\bassume\s+the\s+role\s+of\b",
            r"\btake\s+on\s+the\s+role\s+of\b",
            r"\bimagine\s+you\s+are\b",
            r"\bbehave\s+like\s+(?:a|an)?\b",
            r"\byou\s+will\s+act\s+as\b",
            r"\brole\s*:\s*\w+",
            r"\bpersona\s*:\s*\w+",
            r"\bin\s+your\s+capacity\s+as\b"
        ]
        self.role_regex = re.compile(r"|".join(role_patterns), re.IGNORECASE)

        # 4. UNSPECIFIED OUTPUT STRUCTURE
        structure_patterns = [
            r"output\s+format\s*:",
            r"output\s+structure\s*:",
            r"structure\s+the\s+output\s+as",
            r"format\s+(?:your|the)\s+output\s+as",
            r"the\s+output\s+must\s+be\b",
            r"output\s+length\s+must\s+be",
            r"return\s+(?:only|strictly)\b",
            r"respond\s+(?:only|strictly)\s+in",
            r"format\s*:\s*(?:json|markdown|yaml|xml|csv|table|python\s+code|code\s+block)",
            r"produce\s+(?:the\s+)?output\s+in",
            r"use\s+the\s+following\s+(?:schema|template|format|structure)",
            r"limit\s+(?:the\s+)?output\s+to",
            r"keep\s+(?:your\s+)?response\s+under"
        ]
        self.structure_regex = re.compile(r"|".join(structure_patterns), re.IGNORECASE)

        # 5. LACK OF EXAMPLES
        example_patterns = [
            r"(?:input|in)\s*:\s*.*?\s*(?:output|out)\s*:",
            r"example\s*(?:\d+|[a-z])?\s*:",
            r"sample\s+(?:input|output|code)\s*:",
            r"for\s+example\s*:",
            r"here\s+is\s+an?\s+example\s*:",
            r"e\.g\.\s*,?",
            r"test\s+case\s*\d*\s*:",
            r"input\s+example\s*:"
        ]
        self.example_regex = re.compile(r"|".join(example_patterns), re.IGNORECASE | re.DOTALL)

        self.lang_tool = language_tool_python.LanguageTool('en-US')


    def __calculate_cls(self, prompt: str) -> float:
        """
        Calculate the Complexity-Length Score (CLS) of a given prompt using the formula:
        CLS = 1 - min(1, ((WC / WC_max) + (GFI / 20)) / 2)
        Where WC: word count; WC_max: length threshold; GFI: Gunning Fog Index.
        """
        # Given the structure of the formula, if the prompt is empty
        # or consists only of spaces, 1.0 may be returned directly.
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
        # Given the structure of the formula, if the prompt is empty
        # or consists only of spaces, 1.0 may be returned directly.
        if not prompt or prompt.strip() == "":
            return 1.0

        # Grammatical Correctness Score calculation
        n_words = textstat.lexicon_count(prompt, removepunct=True)
        matches = self.lang_tool.check(prompt)
        n_matches = len(matches)
        g_score = 1.0 - (n_matches / max(1, n_words))

        return round(g_score, 4)


    def analyze_prompt(self, prompt: str) -> dict:
        """
        Analyze an individual prompt and return metrics and the presence of smells.
        """
        # 1. Reasoning Suppression
        cot_matches = self.cot_regex.findall(prompt)
        reasoning_score = 0.5 if len(cot_matches) > 0 else 0.0

        # 2. Self Reflection
        self_reflection_matches = self.self_reflection_regex.findall(prompt)
        has_self_reflection = 1 if len(self_reflection_matches) > 0 else 0

        # 3. Role Assignment
        role_matches = self.role_regex.findall(prompt)
        has_role = 1 if len(role_matches) > 0 else 0

        # 4. Output Structure Specification
        structure_matches = self.structure_regex.findall(prompt)
        has_structure = 1 if len(structure_matches) > 0 else 0

        # 5. Examples Count
        example_matches = self.example_regex.findall(prompt)
        examples_count = len(example_matches)

        # 6. Complexity Length
        cls = self.__calculate_cls(prompt)
        CL_THRESHOLD = 0.75

        # 7. Grammatical Correctness
        g_score = self.__calculate_g(prompt)
        G_THRESHOLD = 0.9

        return {
            "metrics": {
                "reasoning_score": reasoning_score,
                "self_reflection_present": has_self_reflection,
                "role_assigned": has_role,
                "structure_specified": has_structure,
                "examples_count": examples_count,
                "complexity_length_score": cls,
                "grammatical_correctness_score": g_score
            },
            "smells_detected": {
                "reasoning_suppression": reasoning_score == 0.0,
                "lack_of_self_reflection": has_self_reflection == 0,
                "role_suppression": has_role == 0,
                "unspecified_output_structure": has_structure == 0,
                "lack_of_examples": examples_count == 0,
                "complexity_length": cls > CL_THRESHOLD,
                "poor_grammar": g_score < G_THRESHOLD
            }
        }