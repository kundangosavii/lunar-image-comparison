from fastapi import APIRouter, UploadFile, File, Form
from app.services.pipeline import run_pipeline

router = APIRouter()

@router.post("/match")
async def match_images(
    source: UploadFile = File(...),
    reference: UploadFile = File(...),
    source_sensor: str = Form(...),
    reference_sensor: str = Form(...),
):
    source_bytes = await source.read()
    reference_bytes = await reference.read()

    result = run_pipeline(
        source_bytes,
        reference_bytes,
        source_sensor=source_sensor,
        reference_sensor=reference_sensor
    )

    return result