import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


class GroqService:

    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model  = "meta-llama/llama-4-scout-17b-16e-instruct"

    def _chat(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip()

    def _parse_json(self, raw: str) -> dict:
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())

    def extract_receipt_data(self, ocr_text: str) -> dict:
        prompt = f"""
You are extracting structured data from a fuel bill receipt.
The text below was extracted via OCR. It may contain tables, special characters,
currency symbols (Rs. or rupee symbol), dashes, colons, and multi-column item rows.

Extract exactly these fields and return ONLY valid JSON:

{{
                    "receipt_no"    : "receipt number",
                    "fcc_id"        : "FCC ID",
                    "fpi_no"        : "FPI number",
                    "nozzle_no"     : "nozzle number",
                    "product"       : "fuel product type (Petrol/Diesel/CNG etc.)",
                    "density"       : <float — fuel density>,
                    "present_type"  : "present/dispensed type",
                    "rate"          : <float — price per unit>,
                    "volume"        : <float — fuel volume in liters>,
                    "amount"        : <float — total amount for this transaction>,
                    "atot"          : <float — total amount>,
                    "vtot"          : <float — total volume>,
                    "date"          : "DD/MM/YYYY",
                    "time"          : "HH:MM:SS"
}}

Rules:
- Read every line carefully including all table rows.
             - Extract all fields as they appear on the fuel bill.
- If a field is not found, set its value to null.
- Do not guess or make up values not present in the text.
- Return ONLY the JSON object. No explanation, no extra text.

OCR text:
{ocr_text}
"""
        raw = self._chat(prompt)
        return self._parse_json(raw)

    def get_spending_insights(self, receipts: list) -> str:
        if not receipts:
            return "No fuel bills found to analyse."

        summary = json.dumps(
            [r["data"] for r in receipts if r.get("data")],
            indent=2
        )
        prompt = f"""
Analyse these fuel bills and give a clear summary:
- Total fuel purchased across all bills
- Total amount spent across all bills
- Most frequently purchased fuel type
- Average fuel quantity per transaction
- Average fuel price per unit
- Date range of transactions

Fuel bill data:
{summary}

Give a clean, readable summary. No JSON needed.
"""
        return self._chat(prompt)
