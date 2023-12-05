import os
import pandas as pd
import json 
from datetime import date
import time
today = str(date.today().strftime("%b-%d-%Y"))
import numpy as np

from LaunchTool import TextCVTool
from JaroDistance import jaro_distance

OCR_HELPER_JSON_PATH  = r"CONFIG\\OCR_config.json"
OCR_HELPER = json.load(open(OCR_HELPER_JSON_PATH)) 

def _condition_fuction(col, proposition, data):
    if data in ["None", "Nan", "NaN", "nan", "NAN", np.nan]:
        return None
    if col == "parasite_recherche":
        data = data.strip('][').split(', ')
        proposition = proposition.strip('][').split(', ')
        if len(proposition) == len(data):
            return 2
        else: return 0
                
    else:
        proposition = str(proposition).lower()
        data = str(data).lower()
        if proposition == data:
            return 2
        if jaro_distance(proposition, data)>0.8:
            return 1
        else :
            return 0
        
def eval_text_extraction(path_to_eval, eval_path = r"C:\Users\CF6P\Desktop\cv_text\Evaluate",
                         result_name = "0_results", config= ["paddle", "structure", "en", "no_bin"]):
    
    result_excel_path = os.path.join(eval_path, result_name+".xlsx")
    data_col = ["root_path", "page_name"]
    zones_col = list(OCR_HELPER["hand"].keys())+["format"]
    
    if os.path.exists(result_excel_path):
        eval_df = pd.read_excel(result_excel_path, sheet_name="results")
        proba_df= pd.read_excel(result_excel_path, sheet_name="proba")
    else :
        eval_df = pd.DataFrame(columns=data_col+zones_col) # All detected text for all zones
        proba_df = pd.DataFrame(columns=data_col+zones_col)
    res_image_name, res_dict_per_image, res_image = TextCVTool(path_to_eval, config=config)
    for image, zone_dict in res_dict_per_image["RESPONSE"].items():
        row = [path_to_eval, image]
        for _, landmark_text_dict in zone_dict.items():
            row.append(landmark_text_dict["sequence"])
        if len(row) < len(zones_col):
            row.insert(5, [])
            row.insert(9, [])
            row.insert(10, [])
        row.append(landmark_text_dict["format"])
        eval_df.loc[len(eval_df)] = row
    eval_df.to_excel(result_excel_path, sheet_name="results", index=False)

    for image, zone_dict in res_dict_per_image["RESPONSE"].items():
        row = [path_to_eval, image]
        for _, landmark_text_dict in zone_dict.items():
            row.append(int(landmark_text_dict["confidence"]*100))
        if len(row) < len(zones_col):
            row.insert(5, [])
            row.insert(9, [])
            row.insert(10, [])
        row.append(landmark_text_dict["format"])
        proba_df.loc[len(proba_df)] = row
    
    with pd.ExcelWriter(result_excel_path, mode = 'a') as writer:
        proba_df.to_excel(writer, sheet_name="proba", index=False)
    
    
def get_score(result_name, eval_path = r"C:\Users\CF6P\Desktop\cv_text\Evaluate\V3",
              data_excel_path = r"C:\Users\CF6P\Desktop\ELPV\Data\annotated_data.xlsx"):
    
    result_excel_path = os.path.join(eval_path, result_name+".xlsx")
    data_df = pd.read_excel(data_excel_path).fillna("NaN") # The real value of the scan
    eval_df = pd.read_excel(result_excel_path)
    zones_col = list(OCR_HELPER["hand"].keys())
    print("Evaluation is starting")
    
    data_df.drop(columns=["sheet_format"])
    data_df = data_df[data_df["page_name"].isin(eval_df["page_name"])].reset_index()
    score_df= eval_df[eval_df["page_name"].isin(data_df["page_name"])].reset_index()
    missing_annot = list(eval_df[~eval_df["page_name"].isin(score_df["page_name"])]["page_name"])
    if len(missing_annot)!=0:
        print(f" /!\ {missing_annot} Rows are missing in data annotation /!\ ")
    print(eval_df.head(3))
    for col in zones_col:
        apply_df = score_df[["page_name", col]].merge(data_df[["page_name", col]], how='inner', on=["page_name"])
        apply_df.columns = ["page_name", col+"_score", col+"_data"]
        score_df[col] = apply_df.apply(lambda x : _condition_fuction(col, x[col+"_score"], x[col+"_data"]), axis=1)
    with pd.ExcelWriter(result_excel_path, mode = 'a') as writer:
        score_df.to_excel(writer, sheet_name="score", index=False)

    print("Evaluation is done")
    return score_df[zones_col].stack().value_counts()

if __name__ == "__main__":
    
    result_name = "prod_V2.4_Bin"
    eval_path = r"C:\Users\CF6P\Desktop\ELPV\Eval"
    start = time.time()
    eval_text_extraction(eval_path, eval_path=eval_path, result_name=result_name)
    score = get_score(result_name=result_name, eval_path=eval_path)
    taken_time = time.time() - start
    print("############ status #########\n",score,"\n le tout en : (min) ", taken_time/60)
    with open(os.path.join(eval_path, "stack.txt"), 'a') as f:
        f.write("\n"+str(score)+" " + str(taken_time))
    