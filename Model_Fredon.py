import numpy as np
import json
from copy import deepcopy
from unidecode import unidecode
import cv2
import matplotlib.pyplot as plt

import sys
import os

import locale
locale.setlocale(locale.LC_TIME,'fr_FR.UTF-8')
from datetime import datetime
year = datetime.now().year

from paddleocr import PPStructure, PaddleOCR
from paddleocr.ppstructure.recovery.recovery_to_doc import sorted_layout_boxes, convert_info_docx

from JaroDistance import jaro_distance
from ProcessCheckboxes import Template, get_checkboxes
from ProcessPDF import  binarized_image, get_rectangles, get_format_and_adjusted_image
from ConditionFilter import condition_filter

if 'AppData' in sys.executable:
    application_path = os.getcwd()
else : 
    application_path = os.path.dirname(sys.executable)

MODEL = "Fredon"
OCR_HELPER_JSON_PATH  = os.path.join(application_path, "CONFIG\OCR_config.json")
OCR_HELPER = json.load(open(OCR_HELPER_JSON_PATH, encoding="utf-8"))
OCR_PATHES = OCR_HELPER["PATHES"]

NULL_OCR = {"text" : "",
            "box" : [],
            "proba" : 0
           }

class KeyMatch:
    def __init__(self, seq_index, confidence, number_of_match, last_place_word, key_index, OCR):
        self.seq_index = seq_index
        self.confidence = confidence
        self.number_of_match = number_of_match
        self.last_place_word = last_place_word
        self.key_index = key_index
        self.OCR = OCR

class ZoneMatch:
    def __init__(self, local_OCR, match_indices, confidence, res_seq):
        self.local_OCR = local_OCR
        self.match_indices = match_indices
        self.confidence = confidence
        self.res_seq = res_seq

def paddle_OCR(image):

    def _cleanPaddleOCR(OCR_text):
        res = []
        for line in OCR_text:
            for t in line:
                    model_dict = {
                        "text" : "",
                        "box" : [],
                        "proba" : 0
                    }
                    model_dict["text"] = t[1][0]
                    model_dict["box"] = t[0][0]+t[0][2]
                    model_dict["proba"] = t[1][1]
                    res.append(model_dict)
        
        return res

    def _order_by_tbyx(OCR_text):
        res = sorted(OCR_text, key=lambda r: (r["box"][1], r["box"][0]))
        for i in range(len(res) - 1):
            for j in range(i, 0, -1):
                if abs(res[j + 1]["box"][1] - res[j]["box"][1]) < 20 and \
                        (res[j + 1]["box"][0] < res[j]["box"][0]):
                    tmp = deepcopy(res[j])
                    res[j] = deepcopy(res[j + 1])
                    res[j + 1] = deepcopy(tmp)
                else:
                    break
        return res
    
    ocr = PaddleOCR(use_angle_cls=True, lang='fr', show_log = False) # need to run only once to download and load model into memory
    results = ocr.ocr(image, cls=True)
    results = _cleanPaddleOCR(results)
    results = _order_by_tbyx(results)

    # if True:
    #     im = deepcopy(image)
    #     for i, cell in enumerate(results):
    #         x1,y1,x2,y2 = cell["box"]
    #         cv2.rectangle(
    #             im,
    #             (int(x1),int(y1)),
    #             (int(x2),int(y2)),
    #             (0,0,0),2)
    #     plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    #     plt.show()
    
    return results

def find_match(key_sentences, paddleOCR, box, eta=0.95): # Could be optimized
    """
    Detect if the key sentence is seen by the OCR.
    If it's the case return the index where the sentence can be found in the text returned by the OCR,
    else return an empty array
    Args:
        key_sentences (list) : contains a list with one are more sentences, each word is a string.
        text (list) : text part of the dict returned by pytesseract 

    Returns:
        res_indexes (list) : [[start_index, end_index], [empty], ...]   Contains for each key sentence of the landmark the starting and ending index in the detected text.
                            if the key sentence is not detected res is empty.
    """
    def _get_best(base_match, new_match):

        best = base_match
        if new_match == None:
            return best
        elif base_match.number_of_match < new_match.number_of_match: # Choose best match
            best = new_match
        elif (base_match.number_of_match == new_match.number_of_match): # If same number of match, choose the first one
            best=base_match
        return best
    
    xmin,ymin,xmax, ymax = box
    best_matches = None
    for i_place, dict_sequence in enumerate(paddleOCR):
        x1,y1 = dict_sequence["box"][:2]
        seq_match = None
        if xmin<x1<xmax and ymin<y1<ymax:
            sequence = dict_sequence["text"]
            for i_key, key in enumerate(key_sentences): # for landmark sentences from the json
                key_match = None
                for i_word, word in enumerate(sequence):
                    word = unidecode(word).lower()
                    for _, key_word in enumerate(key): # among all words of the landmark
                        key_word = unidecode(key_word).lower()
                        if word[:min(len(word),len(key_word))] == key_word:
                            distance = 1
                        else :
                            distance = jaro_distance("".join(key_word), "".join(word)) # compute the neighborood matching
                        if distance > eta : # take the matching neighborood among all matching words
                            if key_match == None:
                                key_match = KeyMatch(i_place, distance, 1, i_word, i_key, dict_sequence)
                            elif key_match.last_place_word<i_word:
                                key_match.confidence = min(key_match.confidence, distance)
                                key_match.number_of_match+=1
                if seq_match==None : 
                    seq_match=key_match
                else:
                    seq_match = _get_best(seq_match, key_match)

        if best_matches==None : 
            best_matches=seq_match
        else:
            best_matches = _get_best(best_matches, seq_match)
    
    # if best_matches != None : print(best_matches.OCR["text"], key_sentences[best_matches.key_index], best_matches.number_of_match)
    
    return best_matches

def clean_sequence(paddle_list, full = "|\[]_!<>{}—;$€&*‘§—~", left="'(*): |\[]_!.<>{}—;$€&-"):
    res_dicts = []
    for dict_seq in paddle_list:
        text = dict_seq["text"]
        text = text.replace(" :", ":") if " :" in text else text
        text = text.replace(":", ": ") if ":" in text else text
        text = text.replace(":  ", ": ") if ":  " in text else text

        text = text.replace("_", " ") if "_" in text else text
        text = text.replace("I'", "l'") if "I'" in text else text

        if not text in full+left:
            text = [word.strip(full) for word in text.split(" ")]
            dict_seq["text"] = [word for word in text if word]
            res_dicts.append(dict_seq)

    return res_dicts

def get_key_matches_and_OCR(format, cropped_image, JSON_HELPER=OCR_HELPER):
    """
    Perform the OCR on the processed image, find the landmarks and make sure there are in the right area 
    Args:
        cropped_image (array)

    Returns:
        zone_match_dict (dict) :  { zone : Match,
        }
        The coordinate of box around the key sentences for each zone, empty if not found
        OCR_data (dict) : pytesseract returned dict
    """
    image_height, image_width = cropped_image.shape[:2]
    zone_match_dict = {}

    # Search text on the whole image
    full_img_OCR =  paddle_OCR(cropped_image)
    full_img_OCR = clean_sequence(full_img_OCR)
    for zone, key_points in JSON_HELPER[format].items():
        landmark_region = key_points["subregion"] # Area informations
        ymin, ymax = image_height*landmark_region[0][0],image_height*landmark_region[0][1]
        xmin, xmax = image_width*landmark_region[1][0],image_width*landmark_region[1][1]

        match = find_match(key_points["key_sentences"], full_img_OCR, (xmin,ymin,xmax, ymax))
        # print("base : ", zone, (xmin, ymin, xmax, ymax))
        # plt.imshow(cropped_image[int(ymin):int(ymax), int(xmin):int(xmax)])
        # plt.show()

        if match != None:
            # print("found : ", zone, " - ", match.OCR["box"])
            zone_match_dict.update({zone : match })
            # cv2.rectangle(cropped_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else :
           base_match = deepcopy(KeyMatch(0, -1, 0, 0, 0, NULL_OCR))
           base_match.OCR["box"] = [int(xmin), int(ymin), int(xmax), int(ymax)]
           zone_match_dict.update({zone : base_match})

    return zone_match_dict, full_img_OCR

def get_area(cropped_image, box, relative_position, corr_ratio=1.1):
    """
    Get the area coordinates of the zone thanks to the landmark and the given relative position
    Args:
        box (list): detected landmark box [x1,y1,x2,y2]
        relative_position ([[vertical_min,vertical_max], [horizontal_min,horizontal_max]]): number of box height and width to go to search the tet
    """
    im_y, im_x = cropped_image.shape[:2]
    x1,y1,x2,y2 = box
    h, w = abs(y2-y1), abs(x2-x1)
    h_relative, w_relative = h*(relative_position[0][1]-relative_position[0][0])//2, w*(relative_position[1][1]-relative_position[1][0])//2
    y_mean, x_mean = y1+h*relative_position[0][0]+h_relative, x1+w*relative_position[1][0]+w_relative
    x_min, x_max = max(x_mean-w_relative*corr_ratio,0), min(x_mean+w_relative*corr_ratio*2, im_x)
    y_min, y_max = max(y_mean-h_relative*corr_ratio, 0), min(y_mean+h_relative*corr_ratio*2, im_y)
    (y_min, x_min) , (y_max, x_max) = np.array([[y_min, x_min], [y_max, x_max]]).astype(int)[:2]
    return x_min, y_min, x_max, y_max

def get_checkboxes_table_format(checkbox_dict, area_image):
    TRANSFORM = [lambda x: x, lambda x: cv2.flip(x,0), lambda x: cv2.flip(x,1), lambda x: binarized_image(cv2.resize(cv2.cvtColor(x, cv2.COLOR_GRAY2BGR), (int(x.shape[1]*1.15), x.shape[0]))),
                lambda x: binarized_image(cv2.resize(cv2.cvtColor(x, cv2.COLOR_GRAY2BGR), (x.shape[1], int(x.shape[0]*1.15))))]   
    templates = [Template(image_path=checkbox_dict["cross_path"], label="cross", color=(0, 0, 255), matching_threshold=0.71, transform_list=TRANSFORM)]        
    checkboxes = get_checkboxes(area_image, TEMPLATES=templates, show=False)
    sorted_checkboxes = sorted([checkbox for checkbox in checkboxes if checkbox["LABEL"]=="cross"], key=lambda obj: obj["MATCH_VALUE"], reverse=True)
    return sorted_checkboxes

def get_para_table_format(sorted_checkboxes, parasites, match_indices, candidate_dicts):

    check_parasite, parasite_indices, matched_check_box = [], [], []

    for checkbox in sorted_checkboxes:
        top_mid_bottom = [checkbox["TOP_LEFT_Y"], (checkbox["TOP_LEFT_Y"]+checkbox["BOTTOM_RIGHT_Y"])/2, checkbox["BOTTOM_RIGHT_Y"]]
        for i_cand, parasite in enumerate(parasites):
            x1, y1, x2, y2 = candidate_dicts[match_indices[i_cand]]["box"] # Parasite position    
            if any([y1<point<y2 for point in top_mid_bottom]): # Can select multiple choices
                check_parasite.append(parasite)
                parasite_indices.append(match_indices[i_cand])
                matched_check_box.append(checkbox)
        
    for checkbox in [check for check in sorted_checkboxes if not check in matched_check_box]:
        distance_list = [] 
        for i_cand, parasite in enumerate(parasites):
            if parasite not in check_parasite:
                x1, y1, x2, y2 = candidate_dicts[match_indices[i_cand]]["box"] # Parasite position
                dist = abs((y1+y2)-(checkbox["TOP_LEFT_Y"]+checkbox["BOTTOM_RIGHT_Y"]))/2
                if dist < 100:
                    distance_list.append((dist, parasite, match_indices[i_cand]))
            else: pass

            if distance_list != []:
                nearest_para = min(distance_list, key=lambda x: x[0])
                check_parasite.append(nearest_para[1])
                parasite_indices.append(nearest_para[2])

    clean_para_list, clean_indicies = [], []
    for i_para, para in enumerate(check_parasite):
        if para in ["Meloidogyne chitwoodi", "Meloidogyne fallax"]:
            clean_para_list+= ["Meloidogyne chitwoodi", "Meloidogyne fallax"]
            clean_indicies+= [parasite_indices[i_para], parasite_indices[i_para]]
        if para in ["Globodera pallida", "Globodera rostochiensis"]:
            clean_para_list+= ["Globodera pallida", "Globodera rostochiensis"]
            clean_indicies+= [parasite_indices[i_para], parasite_indices[i_para]]
        else:
            clean_para_list+= [para]
            clean_indicies+= [parasite_indices[i_para]]
    
    clean_para_list, clean_indicies = list(set(clean_para_list)), list(set(clean_indicies))
                
    return clean_indicies, clean_para_list          

def _post_extraction_cleaning(text):

    if text == "PAYS BAS":
        text = "PAYS-BAS"
    if "Ö" in text:
        text = text.replace("Ö", "")
    if "Ä" in text:
        text = text.replace("Ä", "A")
    if "SARL" in text:
        text = text.replace("SARL", "SARL ")
    if "EARL" in text:
        text = text.replace("EARL", "EARL ")
    if "SCEA" in text:
        text = text.replace("SCEA", "SCEA ")

    if "  " in text:
        text = text.replace("  ", " ")

    if "EUROFINS" in text:
        text = text.split("EUROFINS")[0]
    
    return text

def get_wanted_text(cropped_image, zone_key_match_dict, format, full_img_OCR, JSON_HELPER=OCR_HELPER):
    zone_matches = {}

    if format == "Fredon tableau":
        checkbox_dict = JSON_HELPER["PATHES"]["checkbox"][format]
        sorted_checkboxes = get_checkboxes_table_format(checkbox_dict, cropped_image)

    for zone, key_points in JSON_HELPER[format].items():
        key_match =  zone_key_match_dict[zone]
        i_key, box = key_match.key_index,  key_match.OCR["box"]
        condition, relative_position = key_points["conditions"], key_points["relative_position"][i_key]
        xmin, ymin, xmax, ymax = box if key_match.confidence==-1 else get_area(cropped_image, box, relative_position, corr_ratio=1.15)

        if format == "Fredon tableau" and zone in ["N_d_echantillon", "N_de_scelle"]:
            if len(sorted_checkboxes)>0:
                checkbox = sorted(sorted_checkboxes, key=lambda obj: obj["TOP_LEFT_Y"])[0]
                up, down = checkbox["TOP_LEFT_Y"], checkbox["BOTTOM_RIGHT_Y"]
                ymin, ymax = up - 3*abs(down-up), up + 3*abs(down-up)

                # print(box, key_match.confidence, (xmin, ymin, xmax, ymax))
                # plt.imshow(cropped_image[ymin:ymax, xmin:xmax])
                # plt.show()
        
        candidate_dicts = [dict_sequence for dict_sequence in full_img_OCR if 
                      (xmin<dict_sequence["box"][0]<xmax) and (ymin<dict_sequence["box"][1]<ymax)]
                
        zone_match = ZoneMatch(candidate_dicts, [], 0, [])

        if (format, zone) == ("Fredon avec cases", "type_lot"):
            # For now, SORE by default
            match_indices, res_seq = [0], ["SORE"]
                
        else :
            match_indices, res_seq = condition_filter(candidate_dicts, condition, model=MODEL, application_path=application_path, ocr_pathes=OCR_PATHES)

        if (format, zone) == ("Fredon tableau", "analyse"):
            match_indices, res_seq = get_para_table_format(sorted_checkboxes,res_seq, match_indices, candidate_dicts)

        zone_match.match_indices, zone_match.res_seq = match_indices, res_seq
        zone_match.confidence = min([candidate_dicts[i]["proba"] for i in zone_match.match_indices]) if zone_match.match_indices else 0

        if zone != "analyse":
            res_seq = " ".join(zone_match.res_seq).upper().strip(",_( ").lstrip(" ._-!*:-").strip(" ")
            zone_match.res_seq = _post_extraction_cleaning(res_seq)

        print(zone, " : ", zone_match.res_seq)

        zone_matches[zone] = {
                "sequence" : zone_match.res_seq,
                "confidence" : float(zone_match.confidence),
                "area" : (int(xmin), int(ymin), int(xmax), int(ymax))
            }

    return zone_matches 

def textExtraction(format, cropped_image, JSON_HELPER=OCR_HELPER):
    """
    The main fonction to extract text from FDA

    Returns:
        zone_matches (dict) : { zone : {
                                    "sequence": ,
                                    "confidence": ,
                                    "area": }
        }
    """
    zone_key_match_dict, full_img_OCR = get_key_matches_and_OCR(format, cropped_image)
    zone_matches = get_wanted_text(cropped_image, zone_key_match_dict, format, full_img_OCR, JSON_HELPER)
    
    # for zone, res_dict in zone_matches.items():
    #     if not res_dict["sequence"]:
    #         NEW_HELPER = deepcopy(OCR_HELPER)
    #         xmin, ymin, xmax, ymax = res_dict["area"]
    #         NEW_HELPER[format] = {zone : JSON_HELPER[format][zone]}
    #         local_cropped_image = cropped_image[ymin:ymax, xmin:xmax]
    #         zone_key_match_dict, full_img_OCR = get_key_matches_and_OCR(format, local_cropped_image, JSON_HELPER=NEW_HELPER)
    #         new_zone_matches = get_wanted_text(local_cropped_image, zone_key_match_dict, format, full_img_OCR, JSON_HELPER=NEW_HELPER)
    #         zone_matches.update(new_zone_matches)

    # Backup in case format is wrongly assigate
    non_empty_field = 0
    for _, value_dict in zone_matches.items():
        if value_dict["sequence"] != []:
            non_empty_field+=1
    
    if format == "table" and non_empty_field < 2: 
        format = "hand"
        zone_key_match_dict, full_img_OCR = get_key_matches_and_OCR(format, cropped_image)
        zone_matches = get_wanted_text(cropped_image, zone_key_match_dict, format, full_img_OCR, JSON_HELPER=JSON_HELPER)
        
    return zone_matches

def main(scan_dict):
    
    pdfs_res_dict = {}
    last_format = "Fredon sans case"

    for pdf, images_dict in scan_dict.items():
        print("###### Traitement de :", pdf, " ######")
        pdfs_res_dict[pdf] = {}
        for i_image, (image_name, sample_image) in enumerate(list(images_dict.items())):
            print("--------------", image_name, "--------------")

            bin_image = binarized_image(sample_image) 
            rectangles = get_rectangles(bin_image)
    
            # Extract the found format and the image
            format, cropped_image = get_format_and_adjusted_image(bin_image, rectangles, sample_image, input_format="")

            # Communicate the last format to the new one if no one found
            if format:
                last_format = format
            else:
                format = last_format

            print(f"Le format de la fiche {pdf}_sample{i_image} est :", format)
        
            sample_matches = textExtraction(format, cropped_image, JSON_HELPER=OCR_HELPER)

            pdfs_res_dict[pdf][f"sample_{i_image}"] = {"IMAGE" : image_name,
                                "EXTRACTION" : sample_matches} # Image Name

    return pdfs_res_dict
   
if __name__ == "__main__":
    from ProcessCheckboxes import  Template
    from ProcessPDF import PDF_to_images, binarized_image, get_rectangles, get_format_and_adjusted_image

    print("start")
    path = r"C:\Users\CF6P\Desktop\ELPV\Data\scan5.pdf"
    images = PDF_to_images(path)
    
    start = 0
    images = images[:]
    res_dict_per_image = {}
    for i, image in enumerate(images,start+1):
        print(f"\n -------------{i}----------------- \nImage {i} is starting")
        bin_image = binarized_image(image)
        rectangles = get_rectangles(bin_image)
        format, cropped_image = get_format_and_adjusted_image(bin_image, rectangles, image)
        print(f"Image with format : {format} is cropped.")
        textExtraction(format, cropped_image, JSON_HELPER=OCR_HELPER)