class PromptSmellFixer:
    """
    Tools for correcting prompts smells 'Role Suppression',
    'Reasoning Suppression', and 'Lack of Self-Reflection'
    """

    def __init__(self):
        self.ROLE_FIX = "Act as an expert software engineer."
        self.REASONING_FIX = "Let's think step by step."
        self.REFLECTION_FIX = "Review your output before replying."

    def fix_prompt(self, smelly_prompt: str, smells_detected: dict) -> str:
        """
        It receives the original prompt and a dictionary that lists which “prompt smells” have been detected.
        It returns the corrected prompt by applying the specified rules.
        """
        # Initially remove spaces and add a period at the end, if one is missing
        fixed_prompt = smelly_prompt.strip()

        if not fixed_prompt.endswith("."):
            fixed_prompt += "."

        # Role Suppression correction.
        # The sentence is added at the beginning of the prompt
        if smells_detected.get("role_suppression"):
            fixed_prompt = f"{self.ROLE_FIX} {fixed_prompt}"

        # Reasoning Suppression correction.
        # The sentence is added at the end of the prompt
        if smells_detected.get("reasoning_suppression"):
            fixed_prompt = f"{fixed_prompt} {self.REASONING_FIX}"

        # Lack of Self-Reflection correction.
        # The sentence is added at the end of the prompt, after the reasoning request
        if smells_detected.get("lack_of_self_reflection"):
            fixed_prompt = f"{fixed_prompt} {self.REFLECTION_FIX}"

        return fixed_prompt