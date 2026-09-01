import uvicorn

from naso import PromptSmellDetector
from mano import PromptSmellFixer
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

### FastAPI configuration ###
app = FastAPI(
    title="NasoMano API",
    description="Naso detects prompt smells, Mano fixes them",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# The tools are instantiated only once when the application starts
detector = PromptSmellDetector()
fixer = PromptSmellFixer()


### Pydantic models ###
# Detect Request
class DetectRequest(BaseModel):
    prompt: str

# Fix Request
class SmellsDetected(BaseModel):
    reasoning_suppression: bool = False
    lack_of_self_reflection: bool = False
    role_suppression: bool = False
    unspecified_output_structure: bool = False
    lack_of_examples: bool = False

class FixRequest(BaseModel):
    prompt: str
    smells_detected: SmellsDetected


### Endpoints ###
@app.post("/api/detect")
async def detect_smells_endpoint(request: DetectRequest):
    """
    Receive a prompt as a string and returns the results of the analysis,
    including metrics and identified smells.
    """

    # Basic validation to avoid processing completely empty prompts
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="The prompt provided is empty.")

    analysis = detector.analyze_prompt(request.prompt)
    return analysis


@app.post("/api/fix")
async def fix_prompt_endpoint(request: FixRequest):
    """
    Receives a prompt and the result from the prompt-smells detector
    Corrects the following smells, if any: Role Suppression, Reasoning Suppression, and Lack of Self-Reflection.
    Returns the corrected prompt.
    """
    # Converts the Pydantic submodel into a standard Python dictionary
    smells_dict = request.smells_detected.model_dump()

    if (smells_dict.get("role_suppression")
            or smells_dict.get("reasoning_suppression")
            or smells_dict.get("lack_of_self_reflection")):
        corrected_prompt = fixer.fix_prompt(request.prompt, smells_dict)
        return corrected_prompt
    else:
        return request.prompt

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)