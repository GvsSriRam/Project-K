from pathlib import Path

from scripts.log import logger
from scripts.model.ocr import Ocr


PROJECT_ROOT_DIR = str(Path(__file__).parent.parent.parent)


class ModelConnector:
    """Connects model operations to endppoint-router"""

    def __init__(self) -> None:
        pass

    def expenses(self, image_path):
        res = []
        try:
            ocr_obj = Ocr()
            text = ocr_obj.text_extraction(image_path)
            expenses = ocr_obj.expenses_detection_walmart(text)

            res = [{"expense_id": i, "text": expense.title, "amount": expense.amt} for i, expense in enumerate(expenses)]
        except Exception as e:
            logger.error(str(e))
        return res
    
    def expenses_gemini(self, image_path):
        res = []
        try:
            ocr_obj = Ocr()
            expenses = ocr_obj.expenses_detection_gemini(image_path)

            res = [{"expense_id": i, "text": expense.get("title"), "quantity": expense.get("quantity"), "amount": expense.get("amt")} for i, expense in enumerate(expenses)]
        except Exception as e:
            logger.error(str(e))
        return res
    
    def split(self, data):
        res = {}
        try:
            ocr_obj = Ocr()
            res = ocr_obj.split_from_web_data(data)
        except Exception as e:
            logger.error(str(e))
        return res
