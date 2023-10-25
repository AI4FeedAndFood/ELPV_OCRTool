import pandas as pd
import json
import os

JSON   = r"CONFIG\\OCR_config.json"
OCR_HELPER = json.load(open(JSON, encoding="utf-8"))

df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in OCR_HELPER["lists"].items() ]))

df.to_excel(os.path.join(r"C:\Users\CF6P\Desktop\ELPV\ELPV_OCRTool\CONFIG\lists.xlsx"))