import cv2
import re
import pytesseract
from typing import Optional, List

import base64
import google.auth
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part, Image
import vertexai.preview.generative_models as generative_models

from scripts.log import logger

# pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'

class Ocr:

    def __init__(self) -> None:
        pass

    def text_extraction(self, image_path):
        img = cv2.imread(image_path)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        custom_config = r'--oem 3 --psm 6'  # Tesseract configurations
        extracted_text = pytesseract.image_to_string(thresh, config=custom_config)

        return extracted_text
    
    def expenses_detection_walmart(self, text: str):
        text = text.split('Walmart ><.\n')[1]
        text, total_amt = re.split(r'\bTOTAL\b', text)
        expenses = text.splitlines()
        expenses = expenses[4:]

        total_amt = total_amt.splitlines()[0]
        total_amt = float(total_amt)

        expense_objs = []

        for expense in expenses:
            if expense.find("SUBTOTAL") != -1:
                break
            obj = Expense()
            obj(expense)
            expense_objs.append(obj)
        
        return expense_objs
    
    def expenses_detection_gemini(self, image_path: str):
        _, project_id = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        img = Image.load_from_file("walmart_bill_pic.jpeg")
        img = Part.from_image(img)

        vertexai.init(project = project_id, location = "us-central1")
        model = GenerativeModel("gemini-pro-vision")
        responses = model.generate_content(
            [
                img, 
                """List only the expenses and their quantity, amount in the bill in json format - {
                    "name": "xxxxxx",
                    "quantity": "yyyy",
                    "price": "zzzz"
                    }. Also, show taxes, totals."""
            ],
            generation_config={
                "max_output_tokens": 2048,
                "temperature": 0.4,
                "top_p": 1,
                "top_k": 32
            },
            safety_settings={
                    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            },
            stream=True,
        )
        
        res = ""
        for response in responses:
            res += response.text
        
        res = res[res.find("{"):res.rfind("}")+1]
        res = res.replace("\n", "")
        res = "[" + res.strip(",") + "]"
        res = res.replace(""""name":""", """"title":""")
        res = res.replace(""""price":""", """"amt":""")
        res = eval(res)
        
        # print("result from OCR:")
        # print(res)

        return res


    
    def split_from_web_data(self, data):
        user_id = {user: i for i, user in enumerate(data["participants"])}
        paid_by = user_id[data["paidBy"]]
        expenses = [Expense(bill["text"], bill["amount"], paid_by, [user_id[i] for i in bill["splitTo"]]) for bill in data["bills"]]
        
        amt_paid, amt_share, balance_amt = self.final_split(expenses, list(user_id.keys()))

        res = []
        for user in data["participants"]:
            tmp = {user: {"amount_paid": amt_paid[user], 
                    "amount_share": amt_share[user], 
                    "balance_amount": balance_amt[user]}}
            res.append(tmp)
        
        return res
    
    def final_split(self, expenses, user_ids):
        res = {}
        amounts_paid = {}
        amounts_taken = {}
        try:
            id_users = {i: user for i, user in enumerate(user_ids)}
            for i in user_ids:
                amounts_paid[i] = 0
                amounts_taken[i] = 0
            for expense in expenses:
                amounts_paid[id_users[expense.paid_by]] += expense.amt
                for i in expense.split_to:
                    amounts_taken[id_users[i]] += expense.amt / len(expense.split_to)
            
            for i in user_ids:
                res[i] = amounts_paid[i] - amounts_taken[i]
        except Exception as e:
            logger.error(str(e))
        
        return amounts_paid, amounts_taken, res
        

class Expense():

    def __init__(self, title: Optional[str] = None, amt: Optional[float] = None, paid_by: Optional[int] = None, split_to: Optional[List[int]] = None):
        self.title = title
        self.amt = float(amt) if amt is not None else None
        self.paid_by = int(paid_by) if paid_by is not None else None
        self.split_to = [int(x) for x in split_to] if split_to is not None else None
    
    def split(self, paid_by, split_to):
        self.paid_by = int(paid_by)
        self.split_to = [int(x) for x in split_to.split(",")]
    
    def __call__(self, row):
        row_entities = row.strip(r'^[A-Z][^?!.]*[?.!]$').rsplit()[:-1]
        self.title = " ".join(row_entities[:-1])

        try:
            self.amt = float(row_entities[-1])
        except ValueError:
            self.amt = 0
        
