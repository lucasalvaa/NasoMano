import re


class PromptSmellDetector:
    """
    Detect the presence of the following five “prompt smells” in English-Language prompts:
    1. Reasoning Suppression
    2. Lack of Self-Reflection
    3. Role Suppression
    4. Unspecified Output Structure
    5. Lack of Examples
    """

    def __init__(self):
        # 1. REASONING SUPPRESSION (Classic CoT only)
        cot_patterns = [
            r"let'?s\s+(?:think|reason|solve|work)\s+step[\s-]by[\s-]step",
            r"think\s+logically\s+and\s+step[\s-]by[\s-]step",
            r"think\s+(?:through\s+this\s+)?step[\s-]by[\s-]step",
            r"reason\s+step[\s-]by[\s-]step",
            r"\bfirst\b.*?\bthen\b.*?\bfinally\b",
            r"break\s+down\s+(?:the|this)\s+(?:problem|task|code)\s+step[\s-]by[\s-]step",
            r"walk\s+me\s+through\s+your\s+(?:thought\s+process|logic|reasoning)",
            r"explain\s+how\s+you\s+(?:arrived|got|reached)",
            r"explain\s+your\s+reasoning\s+(?:step[\s-]by[\s-]step|clearly)",
            r"explain\s+(?:your|the)\s+(?:reasoning|logic|methodology|thought\s+process)",
            r"work\s+through\s+this\s+(?:systematically|methodically|step[\s-]by[\s-]step)",
            r"take\s+a\s+deep\s+breath\s+and\s+think",
            r"show\s+your\s+(?:work|working|steps|thought\s+process)",
            r"break\s*(?:down|this|it)\s+(?:down\s+)?into\s+steps",
            r"breakdown\s+of\s+(?:the\s+)?steps",
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

            # -- patterns added --
            r"(?:review|check)\s+(?:this|my|the)\s+code",
            r"double[\s-]check\b",
            r"make\s+sure\s+(?:it|this|that)\s+(?:works|is\s+correct|runs|compiles)",
            r"ensure\s+(?:that\s+)?(?:this|it)\s+(?:works|is\s+correct|runs)",
            r"sanity[\s-]check",
            r"confirm\s+(?:that\s+)?(?:it|this)\s+(?:works|is\s+correct)",
        ]
        self.self_reflection_regex = re.compile(r"|".join(self_reflection_patterns), re.IGNORECASE)

        # 3. ROLE SUPPRESSION
        role_patterns = [
            r"\bact\s+as\s+(?:a|an)?\b",
            r"\byou\s+(?:'re|are)\s+(?:a|an)?\b",
            r"\byou\s+(?:'ll|will)\s+act\s+as\b",
            r"\bas\s+an?\s+expert\s+in\b",
            r"\bpretend\s+to\s+be\b",
            r"\bassume\s+the\s+role\s+of\b",
            r"\btake\s+on\s+the\s+role\s+of\b",
            r"\bimagine\s+you\s+are\b",
            r"\bbehave\s+like\s+(?:a|an)?\b",
            r"\brole\s*:\s*\w+",
            r"\bpersona\s*:\s*\w+",
            r"\bin\s+your\s+capacity\s+as\b",
            r"\bin\s+the\s+role\s+of\b",
            r"\bworking\s+as\s+(?:a|an)\b",
            r"\b(?:(?:think|act|behave|write|code)\s+like|(?:acting\s+)?as)\s+(?:a|an)\s+(?:\w+\s+){0,2}(?:programmer|"
            r"developer|expert|engineer|scientist|analyst|assistant|professional|consultant|specialist|"
            r"architect|coder|designer|writer|tutor|teacher|translator|reviewer|researcher)\b"
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
            r"\bin\s+json\s+format\b",
            r"\bjson\s+(?:array|object|string)\b",
            r"\b(?:in|as)\s+(?:the\s+)?following\s+format\b",
            r"\bas\s+follows\s*:",
            r"\brespond\s+in\s+plain\s+text\b",
            r"\b(?:without|no)\s+(?:any\s+)?markdown\b",
            r"\bin\s+markdown\b",
            r"\breturn\s+(?:the\s+)?(?:results?|output|answer|response)\s+in\b",
            r"\b(?:numbered|bulleted)\s+list\b",
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

        return {
            "metrics": {
                "reasoning_score": reasoning_score,
                "self_reflection_present": has_self_reflection,
                "role_assigned": has_role,
                "structure_specified": has_structure,
                "examples_count": examples_count
            },
            "smells_detected": {
                "reasoning_suppression": reasoning_score == 0.0,
                "lack_of_self_reflection": has_self_reflection == 0,
                "role_suppression": has_role == 0,
                "unspecified_output_structure": has_structure == 0,
                "lack_of_examples": examples_count == 0
            }
        }