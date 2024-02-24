
from fastapi import APIRouter, File, UploadFile

from scripts.log import logger
from scripts.api.model_connector import ModelConnector


# Contains End_Points that are exposed externally

router = APIRouter()

model_connector_obj = ModelConnector()


@router.post("/get_expenses", tags=["split"])
def get_expenses(image: UploadFile = File(...)):
    result = {"expenses": [], "error": ""}
    try:
        content = image.file.read()
        image_path = "temp_image.png"

        # Write the content of the image to a file
        with open(image_path, "wb") as img_file:
            img_file.write(content)

        result["expenses"] = model_connector_obj.expenses(image_path)
    except Exception as e:
        logger.error(str(e))
        result["error"] = str(e)
    return result

@router.post("/get_expenses_gemini", tags=["split"])
def get_expenses_gemini(image: UploadFile = File(...)): #image: UploadFile = File(...)
    result = {"expenses": [], "error": ""}
    try:
        content = image.file.read()
        image_path = "temp_image.png"

        # Write the content of the image to a file
        with open(image_path, "wb") as img_file:
            img_file.write(content)

        result["expenses"] = model_connector_obj.expenses_gemini(image_path)
    except Exception as e:
        logger.error(str(e))
        result["error"] = str(e)
    return result

@router.post("/split", tags=["split"])
def split(data: dict):
    result = {"split_result": "", "error": ""}
    try:
        result["split_result"] = model_connector_obj.split(data)
    except Exception as e:
        logger.error(str(e))
        result["error"] = str(e)
    return result
