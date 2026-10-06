import uvicorn

# To build the docker image
# from .naso import Naso
# from .mano import Mano
from naso import Naso
from mano import Mano

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
detector = Naso()
fixer = Mano()


### Pydantic models ###
class DetectRequest(BaseModel):
    prompt: str

class SmellsDetected(BaseModel):
    reasoning_suppression: bool = False
    lack_of_self_reflection: bool = False
    role_suppression: bool = False

class FixRequest(BaseModel):
    prompt: str
    smells_detected: SmellsDetected

class RefactorRequest(BaseModel):
    prompt: str


### Endpoints ###
@app.post("/api/detect")
async def detect_smells_endpoint(request: DetectRequest):
    """
    Receive a prompt as a string and returns the results of the analysis,
    including metrics and identified smells.
    """

    try:
        detection_report = await detector.detect_smells(request.prompt)
        return detection_report
    except ValueError as e:
        # Edge case in which the prompt is empty
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/fix")
async def fix_prompt_endpoint(request: FixRequest):
    """
    Receives a prompt and a dictionary containing boolean flags about the presence of
    the following smells, if any: Role Suppression, Reasoning Suppression, and Lack of Self-Reflection.
    Returns the corrected prompt.
    """
    try:
        # Converts the Pydantic submodel into a standard Python dictionary
        smells_dict = request.smells_detected.model_dump()
        return fixer.fix(request.prompt, smells_dict)
    except ValueError as e:
        # Edge case in which the prompt is empty
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/refactor")
async def refactor_prompt_endpoint(request: RefactorRequest):
    """
    Receive a prompt as a string and returns both the corrected prompt
    and the results of the analysis carried by Naso.
    """
    try:
        detection_report = await detector.detect_smells(request.prompt)
    except ValueError as e:
        # Edge case in which the prompt is empty
        raise HTTPException(status_code=400, detail=str(e))

    smells_detected = detection_report["smells_detected"]
    smells_for_fix = SmellsDetected(
        reasoning_suppression=bool(smells_detected.get("reasoning_suppression")),
        lack_of_self_reflection=bool(smells_detected.get("lack_of_self_reflection")),
        role_suppression=bool(smells_detected.get("role_suppression")),
    )

    fixed_prompt = fixer.fix(request.prompt, smells_for_fix.model_dump())

    return {
        "fixed_prompt": fixed_prompt,
        "detection_report": detection_report,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
