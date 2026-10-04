class Mano:
    def __init__(self):
        self.ROLE_FIX = "Act as an experienced software engineer."
        self.REASONING_FIX = "Let's think step by step."
        self.REFLECTION_FIX = "Review your output before replying."

    def fix(self, prompt: str, smells_detected: dict) -> str:
        """
        Correct three prompts smells if presents, namely 'Role Suppression',
        'Reasoning Suppression', and 'Lack of Self-Reflection'.

        Parameters:
            prompt (str): The prompt to be fixed.
            smells_detected (dict): A dict of boolean values, one for each of the three metrics.

        Returns:
            str: The fixed prompt.

        Raises:
            ValueError: if the prompt provided is empty or an invalid string.
        """

        if not prompt or not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("The prompt provided is empty.")

        # Remove spaces
        fixed_prompt = prompt.strip()

        # Add a period at the end, if one is missing
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