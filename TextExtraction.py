import numpy as np
import pytesseract
import json
from copy import deepcopy
from unidecode import unidecode
import cv2

import locale
locale.setlocale(locale.LC_TIME,'fr_FR.UTF-8')
from datetime import datetime
year = datetime.now().year

from ProcessCheckboxes import crop_image_and_sort_format, get_format_or_checkboxes, get_lines, Template
from ProcessPDF import PDF_to_images, binarized_image, delete_lines
from JaroDistance import jaro_distance

whitelist =  "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz(),:.-/°&=àéçëôùê''"
pytesseract.pytesseract.tesseract_cmd = r'exterior_program\Tesseract4-OCR\tesseract.exe'
LANG = 'eng+eng2'
TESSCONFIG = [1, 6, whitelist, LANG]
OCR_HELPER_JSON_PATH  = r"CONFIG\\OCR_config.json"
OCR_HELPER = json.load(open(OCR_HELPER_JSON_PATH, encoding="utf-8"))

def _text_filter(text):
    for i, word in enumerate(text):
        if "£" in word:
            text[i] = text[i].replace("£", "E")
        if "N*" in word:
            text[i] = text[i].replace("N*", "N°")
        if "{" in word:
            text[i] = text[i].replace("{", "(")
        if "{" in word:
            text[i] = text[i].replace("{", ")")
        if "’" in word:
            text[i] = text[i].replace("’", "'")
        if "!" in word:
            text[i] = text[i].replace("!", "l")
    return text 

def _find_landmarks_index(key_sentences, text): # Could be optimized
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
    def _landmark_word_filter(sequence): # Aim to make sure the found sequence of word has no empty string at the start or the end
        left, right = 0, len(sequence)-1
        while sequence[left].isspace() or len(sequence[left]) == 0:
            left+=1
        while sequence[right].isspace() or len(sequence[right]) == 0:
            right-=1
        return sequence[left:right], (left, len(sequence)-1-right)

    res_indexes = []
    for key_sentence in key_sentences: # for landmark sentences from the json
        key = key_sentence
        best_key_candidate = key
        res = [] # 
        best_dist = 0.80
        if "NÂ°" in key_sentence :
            key_sentence = list(map(lambda x : x.replace("NÂ°", "N°"),key_sentence)) #Correction of json format
        for i_key, key_word in enumerate(key_sentence): # among all words of the landmark
            for i_word, word in enumerate(text):
                if key_word.lower() == word.lower(): # if there is a perfect fit in an word (Maybe should be softer but would take more time)
                    key_candidate = text[i_word-i_key:i_word-i_key+len(key_sentence)]
                    distance = jaro_distance("".join(key), "".join(key_candidate)) # compute the neighborood matching
                    if distance > best_dist : # take the best distance
                        best_dist = distance
                        best_key_candidate = key_candidate
                        res = [i_word-i_key, i_word-i_key+len(key_sentence)] # Start and end indexes of the found key sentence
                        
        cor_ratio = len(key) / (len(key) - (len(key) -len(best_key_candidate))) + 0.15
        cor_ratio = max(1, min(cor_ratio, 1.3))
            
        if len(res)>0:
            res_text =  text[res[0]:res[-1]] 
            res_text, (start_removed, end_removed) = _landmark_word_filter(res_text)
            res[0], res[1] = res[0]+start_removed, res[1]-end_removed
        res_indexes.append((res, cor_ratio)) # Empty if not found
    return res_indexes

def get_data_and_landmarks(format, cropped_image, JSON_HELPER=OCR_HELPER, ocr_config=TESSCONFIG):
    """
    Perform the OCR on the processed image, find the landmarks and make sure there are in the right area 
    Args:
        cropped_image (array)

    Returns:
        res_landmarks (dict) :  { zone : {
                                "landmark" = [[x,y,w,h], []],
            }
        }
        The coordinate of box around the key sentences for each zone, empty if not found
        OCR_data (dict) : pytesseract returned dict
    """
    image_height, image_width = cropped_image.shape[:2]
    res_landmarks = {}
    # Search text on the whole image
    config = f"--oem {ocr_config[0]} --psm {ocr_config[1]} -c tessedit_char_whitelist=" + ocr_config[2]
    data = ocr_config[3]
    OCR_data =  pytesseract.image_to_data(cropped_image, output_type=pytesseract.Output.DICT, lang=data, config=config)
    text = _text_filter(OCR_data["text"])
    for zone, key_points in JSON_HELPER[format].items():
        detected_indexes_cor = _find_landmarks_index(key_points["key_sentences"], text)
        landmark_region = key_points["subregion"] # Area informations
        xmin, xmax = image_width*landmark_region[1][0],image_width*landmark_region[1][1]
        ymin, ymax = image_height*landmark_region[0][0],image_height*landmark_region[0][1]
        # print("base : ", zone)
        landmarks_coord = []
        null = 0 # Relay if an OCR on a biggest area is needed
        for indexes_cor in detected_indexes_cor:
            indexes, cor = indexes_cor
            if len(indexes)!=0 :
                i_min, i_max = indexes
                x, y = OCR_data['left'][i_min], OCR_data['top'][i_min]
                w = abs(OCR_data['left'][i_max-1] - x + OCR_data['width'][i_max-1])
                h = abs(int(np.mean(np.array(OCR_data['height'][i_min:i_max]))))
                if xmin<x<xmax and ymin<y<ymax: # Check if the found landmark is in the right area
                    landmarks_coord.append(("found", [x,y,w,h], cor))
                    # cv2.rectangle(cropped_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                else : 
                    landmarks_coord.append(("default", [], cor))
                    null+=1
            else : 
                landmarks_coord.append(("default", [], cor))
                null+=1
                
        # No detected landmark: Let's search landmark on a smaller region  
        if null>0:
            OCR_default_region = pytesseract.image_to_data(cropped_image[int(ymin):int(ymax), int(xmin):int(xmax)], 
                                                           output_type=pytesseract.Output.DICT,  lang=data, config=config)
            text = _text_filter(OCR_default_region["text"])
            for i, coord in enumerate(landmarks_coord):
                if len(coord[1])==0:
                    detected_index, cor = _find_landmarks_index([key_points["key_sentences"][i]], text)[0]
                    if len(detected_index)!=0 :
                        i_min, i_max = detected_index
                        x_relative, y_relative = OCR_default_region['left'][i_min], OCR_default_region['top'][i_min]
                        x, y = x_relative + int(xmin), y_relative+int(ymin)
                        w = abs(OCR_default_region['left'][i_max-1] - x_relative + OCR_default_region['width'][i_max-1])
                        h = abs(int(np.mean(np.array(OCR_default_region['height'][i_min:i_max]))))
                        landmarks_coord[i] = ("found", [x,y,w,h], cor)
                    else : 
                        landmarks_coord[i] = ("default", [int(xmin), int(ymin), int(xmax-xmin), int(ymax-ymin)], cor)
        res_landmarks[zone] = {"landmark" : landmarks_coord}
    return OCR_data, res_landmarks

def _process_raw_text_to_sequence(OCR_text):
    candidate_sequences = [] # Store all sequence
    candidate_indexes = []
    sequence = [] # Stack a chunck of non-separated word
    indexes = []
    for i, word in enumerate(OCR_text):
        if not(word.isspace() or len(word) == 0 or word=="|"):
            sequence.append(word)
            indexes.append(i)
        elif len(sequence) != 0: # If space : new sequence
            if sequence not in candidate_sequences:
                indexes.append(i)
                candidate_indexes.append(indexes)
                candidate_sequences.append(sequence)
            sequence=[]
            indexes=[]
        if i == len(OCR_text)-1 and len(sequence)!=0: # Add last sequence
            if sequence not in candidate_sequences:
                candidate_sequences.append(sequence)
                indexes.append(i+1)
                candidate_indexes.append(indexes)
    return candidate_sequences, candidate_indexes

def _get_area(cropped_image, box, relative_position, corr_ratio=1.15):
    """
    Get the area coordinates of the zone thanks to the landmark and the given relative position
    Args:
        box (list): detected landmark box [x,y,w,h]
        relative_position ([[vertical_min,vertical_max], [horizontal_min,horizontal_max]]): number of box height and width to go to search the tet
    """
    im_y, im_x = cropped_image.shape[:2]
    x,y,w,h = box
    h_relative, w_relative = h*(relative_position[0][1]-relative_position[0][0])//2, w*(relative_position[1][1]-relative_position[1][0])//2
    y_mean, x_mean = y+h*relative_position[0][0]+h_relative, x+w*relative_position[1][0]+w_relative
    x_min, x_max = max(x_mean-w_relative*corr_ratio,0), min(x_mean+w_relative*corr_ratio, im_x)
    y_min, y_max = max(y_mean-h_relative*corr_ratio, 0), min(y_mean+h_relative*corr_ratio, im_y)
    (y_min, x_min) , (y_max, x_max) = np.array([[y_min, x_min], [y_max, x_max]]).astype(int)[:2]
    return y_min, y_max, x_min, x_max

def get_candidate_local_OCR(cropped_image, landmark_boxes, relative_positions, format, ocr_config=TESSCONFIG):
    OCRs_and_candidates_list = []
    for n_landmark, relative_position in enumerate(relative_positions):
        res_dict = {}
        box_type, box, cor = landmark_boxes[n_landmark]
        if box_type == "default" : 
            relative_position = [[0,1], [0,1]]
        y_min, y_max, x_min, x_max = _get_area(cropped_image, box, relative_position, corr_ratio=cor)
        searching_area = cropped_image[y_min:y_max, x_min:x_max]
        searching_area = np.pad(searching_area, 5,  constant_values=255)
        config = f' --oem {ocr_config[0]} --psm {ocr_config[1]} -c tessedit_char_whitelist=' + ocr_config[2]
        data = ocr_config[3]
        local_OCR = pytesseract.image_to_data(searching_area, output_type=pytesseract.Output.DICT, lang=data, config=config)
        local_OCR["text"] = _text_filter(local_OCR["text"])
        candidate_sequences, candidate_indexes = _process_raw_text_to_sequence(local_OCR["text"])
        res_dict["OCR"], res_dict["type"], res_dict["box"] = local_OCR, box_type, [int(y_min), int(y_max), int(x_min), int(x_max)]
        res_dict["sequences"], res_dict["indexes"] = candidate_sequences, candidate_indexes
        res_dict["risk"] = 0
        res_dict["format"] = format
        if res_dict["type"] == "default":
            res_dict["risk"] = 1
        OCRs_and_candidates_list.append(res_dict)

    check_seq = []
    for i_candidate, candidate_dict in enumerate(OCRs_and_candidates_list): # delete double sequence
        res_seq, res_index = [], []
        for i_seq, seq in enumerate(candidate_dict["sequences"]):
            if not seq in check_seq:
                check_seq.append(seq)
                res_seq.append(seq)
                res_index.append(candidate_dict["indexes"][i_seq])
        if res_seq == [] and check_seq != []:
            del OCRs_and_candidates_list[i_candidate]
        else:
            OCRs_and_candidates_list[i_candidate]["sequences"], OCRs_and_candidates_list[i_candidate]["indexes"] = res_seq, res_index
    return OCRs_and_candidates_list

def _list_process(check_word, candidate_sequence, candidate_index):
    max_jaro = 0.89
    max_stack = [False, None, 0]
    for i_word, word in enumerate(candidate_sequence):
        jaro = jaro_distance(word.lower(), check_word.lower())
        if jaro == 1 :
            return True, candidate_index[i_word], 1
        elif jaro > max_jaro:
            max_stack = [True, candidate_index[i_word], jaro]
    return max_stack

def _after_key_process(key_sequences, clean_sequences, clean_indexes, similarity=0.9):
    def _get_match(start, start_seq, stop):
        for place in range(start, min(start+search_range+2, len(start_seq))):
            word = unidecode(start_seq[place])
            try:
                index = word.rindex(stop)
                res_word = word[index+len(stop):]
                if res_word == "" and place < len(start_seq)-1:
                    return place+1, start_seq[place+1]
                if res_word == "" and place == len(start_seq)-1:
                    return -1, stop
                return place, res_word
            except ValueError:
                pass
        return start, start_seq[start]
    
    strip = '().*:‘;,§"'+"'"
    matching_key_place_stack, matching_word_place_stack = [[], []], [[], []]
    # Get the begining and and the end of the sequence to take what is between
    for _, candidate_sequence in enumerate(clean_sequences):
        matching_key_place, matching_word_place = [[], []], [[], []] # Store the matching key index and the place on the candidate seq
        for state, key_seq in enumerate(key_sequences):
            for i_key, key_word in enumerate(key_seq):
                    for i_word, word in enumerate(candidate_sequence):
                        try:
                            if int(key_word) == int(word): # Number key_word case
                                matching_key_place[state].append(i_key)
                                matching_word_place[state].append(i_word)
                                break
                        except:
                            if (jaro_distance(key_word, unidecode(word).strip(strip))>similarity):
                                matching_key_place[state].append(i_key)
                                matching_word_place[state].append(i_word)
                                break
                            if key_word.strip("°").isalpha():
                                    if key_word in word:
                                        matching_word_place[state].append(i_word)
                                        matching_key_place[state].append(i_key)
                                        break
                            
        matching_key_place_stack[0].append(matching_key_place[0]) # 0 place is start, 1 is end
        matching_word_place_stack[0].append(matching_word_place[0])
        matching_key_place_stack[1].append(matching_key_place[1])
        matching_word_place_stack[1].append(matching_word_place[1])
        
    if matching_key_place_stack[0] == [] :
        return [], []
    # Clean the first sentence from the landmark
    _get_candidate = lambda stack : sorted([(i, len(l_stack)) for i, l_stack in enumerate(stack)], key=lambda y: (-y[1], y[0]))[0][0] # Most matched sentence, the firt one in case of tie
    start_seq_id, end_seq_id = _get_candidate(matching_key_place_stack[0]), _get_candidate(matching_key_place_stack[1])
    start_seq, start_index =  clean_sequences[start_seq_id], clean_indexes[start_seq_id]
    end_seq, _ =  clean_sequences[end_seq_id], clean_indexes[end_seq_id]
    i_keyword_found, i_keymatch_candidate = matching_key_place_stack[0][start_seq_id], matching_word_place_stack[0][start_seq_id]
    if i_keyword_found == []:
        return [], []
    search_range = (len(key_sequences[0]) - i_keyword_found[-1]) # Last detetcted word by all word you want to detect
    stop_elmt = key_sequences[0][-1].split(" ") + ["(*)", ":"]
    start = i_keymatch_candidate[-1]
    for stop in stop_elmt:
        place_word = _get_match(start, start_seq, stop)
        if place_word[0] != start:
            start+=1
        if place_word[0] == -1:
            if start_seq_id < len(clean_sequences)-1:
                start = 0
                start_seq_id+=1
                start_seq, start_index =  clean_sequences[start_seq_id], clean_indexes[start_seq_id]
                if start_seq_id == end_seq_id:
                    end_seq_id+=1
            else: 
                return [], []
    start_seq[place_word[0]] = place_word[1]
    start_seq = start_seq[place_word[0]:]
    start_index = start_index[place_word[0]:]
    if start_seq == []:
        start_seq_id, end_seq_id = start_seq_id+1, end_seq_id+1
        start_seq, start_index =  clean_sequences[start_seq_id], clean_indexes[start_seq_id]        
    if start_seq[0].strip(strip) == "":
        start_seq = start_seq[1:]
        start_index = start_index[1:]
    if start_seq != []:
        start_seq[0] = start_seq[0].lstrip(strip)
    clean_sequences[start_seq_id], clean_indexes[start_seq_id]  = start_seq, start_index
    if end_seq_id - start_seq_id <= 1: # One sequence to res
        return clean_sequences[start_seq_id], clean_indexes[start_seq_id]
    res_seq, res_index = [], [] # More than one seq to res
    
    for i in range(start_seq_id,end_seq_id):
        res_seq+= clean_sequences[i]
        res_index+= clean_indexes[i]
    return res_seq, res_index

def _clean_local_sequences(sequence_index_zips, key_main_sentences, conditions):
    strip_string_after_key = " |\[]_!.<>{}—;"
    strip_string_others = "()*: |\/[]_!.<>{}—;-&"
    cleaned_candidate, cleaned_indexes = [], []
    key_sentences = [word for sentence in key_main_sentences for word in sentence]
    for candidate_sequence, candidate_indexes in sequence_index_zips:
        res_candidate_sequence, res_candidate_indexes = [], []
        for i_word, word in enumerate(candidate_sequence):
            if "after_key" not in [condition[0] for condition in conditions]:
                if any(c.isalnum() for c in unidecode(word)):
                    if word not in key_sentences:
                        res_candidate_sequence.append(word.strip(strip_string_others))
                        res_candidate_indexes.append(candidate_indexes[i_word])
            else:
                res_candidate_sequence.append(word.strip(strip_string_after_key))
                res_candidate_indexes.append(candidate_indexes[i_word])
                    
        cleaned_candidate.append(res_candidate_sequence)
        cleaned_indexes.append(res_candidate_indexes)
    return cleaned_candidate, cleaned_indexes

def condition_filter(candidates_dicts, key_main_sentences, conditions):
    """_summary_

    Args:
        candidates_dicts (_type_): _description_
        key_main_sentences (_type_): _description_
        conditions (_type_): _description_

    Returns:
        _type_: _description_
    """
    OCRs_and_candidates = deepcopy(candidates_dicts)
    OCRs_and_candidates_filtered = []
    for candidate_dict in OCRs_and_candidates:
        strip_string_others = "()* |\/[]_!.<>‘{}:—;~-+"
        zipped_seq = zip(candidate_dict["sequences"], candidate_dict["indexes"])
        clean_sequence, clean_indexes = _clean_local_sequences(zipped_seq, key_main_sentences, conditions)
        zipped_seq = zip(clean_sequence, clean_indexes)
        for condition in conditions:
            new_sequence, new_indexes = [], []
            if condition[0] == "after_key": 
                key_sequences = condition[1] # Start and end key_sentences
                found_sequence, found_index = _after_key_process(key_sequences, clean_sequence, clean_indexes)
                new_sequence.append(found_sequence)
                new_indexes.append(found_index)
                    
            if condition[0] == "date": # Select a date format
                for candidate_sequence, candidate_index in zipped_seq:
                    for i_word, word in enumerate(candidate_sequence):
                        word_init = word
                        try:
                            word = word.lower().strip(strip_string_others+"abcdefghijklmnopqrstuvwxyz")
                            _ = bool(datetime.strptime(word, "%d/%m/%Y"))
                            new_sequence.append([word])
                            new_indexes.append([candidate_index[i_word]])
                        except ValueError:
                            pass
                        
                        try: # Case dd/mm/yy
                            word = word.lower().strip(strip_string_others+"abcdefghijklmnopqrstuvwxyz")
                            word = word[:-2] + "20" + word[-2:]
                            _ = bool(datetime.strptime(word, "%d/%m/%Y"))
                            new_sequence.append([word])
                            new_indexes.append([candidate_index[i_word]])
                        except ValueError:
                            word = word_init
                        try: # Case month in letter
                            word = word.lower().strip(strip_string_others)
                            _ = bool(datetime.strptime(word, "%B"))
                            full_date = "".join(candidate_sequence[i_word-1:i_word+2])
                            _ = bool(datetime.strptime(full_date, "%d%B%Y"))
                            new_sequence.append(candidate_sequence[i_word-1:i_word+2])
                            new_indexes.append(candidate_index[i_word-1:i_word+2])
                        except ValueError:
                            pass
                    
            if condition[0] == "echantillon": # A special filter for numero d'echantillon
                base1, base2 = condition[1] # Thresholds
                for candidate_sequence, candidate_index in zipped_seq: # Detected sequences wich need iteration over themselves
                    for i_word, word in enumerate(candidate_sequence):
                        if len(word)>7:
                            try_list = [(base1+2000, base2+2000), (base1,base2)]
                            for date_tuple in try_list:
                                num1, num2 = date_tuple[0], date_tuple[1]
                                if word[:len(str(num1))].isnumeric():
                                    date_num, code_num = word[:len(str(num1))], word[len(str(num1)):].upper()
                                    if num1 <= int(date_num) < num2 : # Try to avoid strings shorter than NUM
                                        res="".join(candidate_sequence[i_word:])
                                        res_upper = res.upper()
                                        # Replace common mistake
                                        correction_list = [("MPA", "MP4"), ("N0", "NO"), ("AUOP", "AU0P"), ("CEOP", "CE0P"), ("GEL0", "GELO"), ("PLOP", "PL0P"), 
                                         ("PLIP", "PL1P"), ("NCIP", "NC1P"), ("NCIE", "NC1E"), ("S0R", "SOR"), ("1F", "IF")]
                                        for cor_tuple in correction_list:
                                            error, correction = cor_tuple
                                            if code_num[:len(error)] == error:
                                                res_upper = res_upper.replace(error, correction, 1)
                                                break # One possibility
                                        if code_num[5:9] == "S0PDT": # Unique case
                                            res_upper.replace("SP0DT", "SOPODT", 1)
                                        
                                        new_sequence.append([res_upper])
                                        new_indexes.append(candidate_index[i_word:])
                        if "GECA" in word:
                            res_upper = str(year)+word
                            index = [candidate_index[i_word-1], candidate_index[i_word]]
                            try :
                                if candidate_sequence[i_word+1].isnumeric():
                                    res_upper += candidate_sequence[i_word+1]
                                    index.append(candidate_index[i_word+1])
                            except:
                                pass
                            new_sequence.append([res_upper])
                            new_indexes.append(index)

            if condition[0] == "list": # In this case itertion is over element in the condition list
                concat_seq, concat_index = [], []
                matched_jaro = []
                mode = condition[2]
                for candidate_sequence, candidate_index in zipped_seq:
                    concat_seq+=candidate_sequence
                    concat_index+= candidate_index
                check_list = OCR_HELPER["lists"][condition[1]]
                for check_elmt in check_list:
                    check_indexes=[]
                    jaro_elmt=[]
                    check_words = check_elmt.split(" ")
                    for check_word in check_words:
                        status, index, jaro = _list_process(check_word, concat_seq, concat_index)
                        if status: # If a check word is found : stack it
                            check_indexes.append(index)
                            jaro_elmt.append(jaro)
                        if len(check_indexes) == len(check_words) and check_elmt not in new_sequence: # All word of the checking elements are in the same candidate sequence                          print(check_elmt, jaro_elmt)
                            new_sequence.append([check_elmt])
                            new_indexes.append(check_indexes)
                            matched_jaro.append(min(jaro_elmt))
                if mode == "multiple": # Return all found elements sorted by index
                    sorted_res = sorted(zip(new_sequence, new_indexes, matched_jaro), key=lambda x: x[1][0])
                if mode == "single":
                    sorted_res = sorted(zip(new_sequence, new_indexes, matched_jaro), key=lambda x: x[2], reverse=True)
                
                if sorted_res != []:
                    new_sequence , new_indexes, jaro = zip(*sorted_res)
                    if mode == "single":
                        new_sequence, new_indexes = [new_sequence[0]], [new_indexes[0]]
                    
            new_sequence_res, new_indexes_res = [], []            
            for i in range(len(new_sequence)):
                if new_sequence[i] != []:
                    new_sequence_res.append(new_sequence[i])
                    new_indexes_res.append(new_indexes[i])
            zipped_seq = zip(new_sequence_res, new_indexes_res)
            
        candidate_dict["sequences"], candidate_dict["indexes"] = new_sequence_res, new_indexes_res
        OCRs_and_candidates_filtered.append(candidate_dict)
    return OCRs_and_candidates_filtered
                
def get_checkbox_check_format(format, checkbox_dict, cropped_image, landmark_boxes, relative_positions):
    TRANSFORM = [lambda x: x, lambda x: cv2.flip(x,0), lambda x: cv2.flip(x,1), lambda x: binarized_image(cv2.resize(cv2.cvtColor(x, cv2.COLOR_GRAY2BGR), (int(x.shape[1]*1.15), x.shape[0]))),
             lambda x: binarized_image(cv2.resize(cv2.cvtColor(x, cv2.COLOR_GRAY2BGR), (x.shape[1], int(x.shape[0]*1.15))))]
    
    def _get_checkbox_word(sequence, index, ref_word_list, strips = ["1I ‘1[]JL_-|", "CI 1()[]‘JL_-|"]): # Several strips to be careful with word starting with C or I
        for i_word, word in enumerate(sequence):
            for strip in strips:
                strip_word = word.strip(strip)
                if strip_word in ref_word_list:
                    return [strip_word], [index[i_word]]
        return [], []

    OCRs_and_candidates_list = []
    for n_landmark, relative_position in enumerate(relative_positions):
        res_dict = {}
        box_type, box, _ = landmark_boxes[n_landmark]
        if box_type == "default" : relative_position = [[0,1], [0,1]]
        y_min, y_max, x_min, x_max = _get_area(cropped_image, box, relative_position)
        searching_area = cropped_image[y_min:y_max, x_min:x_max]
        templates = [Template(image_path=checkbox_dict["cross_path"], label="cross", color=(0, 0, 255), matching_threshold=0.4, transform_list=TRANSFORM)]
        checkboxes = get_format_or_checkboxes(searching_area, mode="get_boxes", TEMPLATES=templates, show=False)
        sorted_checkboxes = sorted([checkbox for checkbox in checkboxes if checkbox["LABEL"]=="cross"], key=lambda obj: obj["MATCH_VALUE"], reverse=True)[:3]
        sorted_checkboxes = sorted([checkbox for checkbox in sorted_checkboxes], key=lambda obj: obj["TOP_LEFT_X"], reverse=False)[:2]
        res_dict["OCR"] = {}
        res_dict["type"], res_dict["box"] = box_type,  [int(y_min), int(y_max), int(x_min), int(x_max)]
        res_dict["sequences"], res_dict["indexes"] = [], []
        res_dict["risk"] = 0
        res_dict["format"] = format
        for cross in sorted_checkboxes:
            x1,y1, x2,y2= cross["TOP_LEFT_X"], cross["TOP_LEFT_Y"], cross["BOTTOM_RIGHT_X"], cross["BOTTOM_RIGHT_Y"]
            h, w = abs(y2-y1), abs(x2-x1)
            end_x = min(x_max, x2+2*w)
            y1, y2 = max(int(y1-h/2),0) , min(y_max,int(y2+h/2))
            new_area = searching_area[y1:y2, x2:end_x]
            sequence = []
            c=0 # Make sure it's not looping forever
            while c<2:
                c+=1
                local_OCR = pytesseract.image_to_data(np.pad(new_area, 20,  constant_values=255), output_type=pytesseract.Output.DICT, config = '--oem 1 --psm 6')
                sequence, index = _process_raw_text_to_sequence(local_OCR["text"])
                if len(sequence)>0:
                    res_word, res_index = _get_checkbox_word(sequence[0], index[0], checkbox_dict["list"]) # POSTULATE : sequence is contains only one list
                    if res_word != []:
                        res_dict["OCR"] = local_OCR
                        res_dict["sequences"], res_dict["indexes"] = [res_word], [res_index]
                        OCRs_and_candidates_list.append(res_dict)
                        break
                end_x += 2*w
                end_x = min(x_max, end_x)
                new_area = searching_area[y1:y2, x2:end_x]
                
            if res_dict["sequences"] != []: # Stop the iteration over crosses 
                break
        if OCRs_and_candidates_list == []:
            OCRs_and_candidates_list.append(res_dict)
        
    return OCRs_and_candidates_list
    
def get_checkbox_table_format(checkbox_dict, clean_OCRs_and_candidates, cropped_image):
    TRANSFORM = [lambda x: x, lambda x: cv2.flip(x,0), lambda x: cv2.flip(x,1), lambda x: binarized_image(cv2.resize(cv2.cvtColor(x, cv2.COLOR_GRAY2BGR), (int(x.shape[1]*1.15), x.shape[0]))),
                lambda x: binarized_image(cv2.resize(cv2.cvtColor(x, cv2.COLOR_GRAY2BGR), (x.shape[1], int(x.shape[0]*1.15))))]
    templates = [Template(image_path=checkbox_dict["cross_path"], label="cross", color=(0, 0, 255), matching_threshold=0.51, transform_list=TRANSFORM)]
    
    OCRs_and_candidates_list = []
    templates = [Template(image_path=checkbox_dict["cross_path"], label="cross", color=(0, 0, 255), matching_threshold=0.51, transform_list=TRANSFORM)]        
    for candidate_dict in clean_OCRs_and_candidates:
        res_dict = candidate_dict
        y_min, y_max, x_min, x_max = candidate_dict["box"]
        searching_area = cropped_image[y_min:y_max, x_min:x_max]
        checkboxes = get_format_or_checkboxes(searching_area, mode="get_boxes", TEMPLATES=templates, show=False)
        sorted_checkboxes = sorted([checkbox for checkbox in checkboxes if checkbox["LABEL"]=="cross"], key=lambda obj: obj["MATCH_VALUE"], reverse=True)
        
        parasite_location = []
        for parasite, indexes in zip(candidate_dict["sequences"], candidate_dict["indexes"]):
            parasite_dict =  {}
            parasite_dict["parasite"], parasite_dict["indexes"] = parasite, indexes
            last_index = indexes[-1]
            parasite_dict["top"], parasite_dict["height"] = res_dict["OCR"]["top"][last_index], res_dict["OCR"]["height"][last_index]
            parasite_location.append(parasite_dict)
            
        parasite_list, parasite_index = [], []
        check_not_matched = []
        for checkbox in sorted_checkboxes:
            top_mid_bottom = [checkbox["TOP_LEFT_Y"], (checkbox["TOP_LEFT_Y"]+checkbox["BOTTOM_RIGHT_Y"])/2, checkbox["BOTTOM_RIGHT_Y"]]
            found = False
            for parasite_dict in parasite_location:
                if any([parasite_dict["top"]<point<(parasite_dict["top"]+parasite_dict["height"]) for point in top_mid_bottom]): # Can select multiple choices
                    parasite_list.append(parasite_dict["parasite"])
                    parasite_index.append(parasite_dict["indexes"])
                    found = True
            if found == False:
                check_not_matched.append(checkbox)
        
        for checkbox in check_not_matched:
            distance_list = [] 
            for parasite_dict in parasite_location:
                if parasite_dict["parasite"] not in parasite_list:
                    dist = abs((parasite_dict["top"]+parasite_dict["height"]/2)-(checkbox["TOP_LEFT_Y"]+checkbox["BOTTOM_RIGHT_Y"])/2)
                    if dist < 100:
                        distance_list.append((dist, parasite_dict))
                else: pass

            if distance_list != []:
                nearest_para = min(distance_list, key=lambda x: x[0])
                parasite_list.append(nearest_para[1]["parasite"])
                parasite_index.append(nearest_para[1]["indexes"])
                
        if (["Meloidogyne chitwoodi"] in parasite_list) or (["Meloidogyne fallax"] in parasite_list):
            parasite_list+= [["Meloidogyne chitwoodi"], ["Meloidogyne fallax"]]
        if (["Globodera pallida"]) in parasite_list or (["Globodera rostochiensis"] in parasite_list):
            parasite_list+= [["Globodera pallida"], ["Globodera rostochiensis"]]
            
        clean_para_list = []
        for para in parasite_list:
            if para not in clean_para_list:
                clean_para_list.append(para)
                
        res_dict["sequences"]  = clean_para_list
        res_dict["indexes"] = parasite_index
        OCRs_and_candidates_list.append(res_dict)
    return OCRs_and_candidates_list    

def common_mistake_filter(OCRs_and_candidates, zone):      
    clean_OCRs_and_candidates = []
    for candidate_dict in OCRs_and_candidates:
        res_seq, res_index = [], []
        sequences, indexes = candidate_dict["sequences"], candidate_dict["indexes"]
        for sequence, index in zip(sequences, indexes):
            to_del = []
            for i, word in enumerate(sequence):
                if word in ["", " "]:
                    to_del.append(i)
                if ":" in word:
                    if word[0] != ":":
                        p = word.index(":")
                        if word[p-1] != " ":
                            sequence[i] = word.replace(":", " :")
                if "," in word:
                    p = word.index(",")
                    num = [str(i) for i in range(10)]
                    if p-1>0 and p+1<len(word)-1:
                        if word[p-1] and word[p+1] in num:
                            sequence[i] = word.replace(",",".")
                if i>0: # Filter by confidance
                    Pi0 = candidate_dict["OCR"]["conf"][index[i]]
                    if Pi0 <0: # Delete very unreliable word          
                        to_del.append(i)
                    if len(word)==1 : # Delete single word 
                        if candidate_dict["OCR"]["conf"][index[i-1]] - Pi0 > 35 and Pi0 < 55: # Delete confidence drop 
                            if i == len(sequence)-1:
                                to_del.append(i)
                            elif abs(candidate_dict["OCR"]["conf"][index[i+1]] - Pi0) > 10: # delete last word
                                to_del.append(i)
                            
                if zone == "nom":
                    if word.lower()[:5] == "eurof":
                        sequence = sequence[:i]
                        index = index[:i]
                        break
                    
            new_sequence, new_index = [word for i, word in enumerate(sequence) if i not in to_del], [id for i, id in enumerate(index) if i not in to_del]
            res_seq.append(new_sequence)
            res_index.append(new_index)
        candidate_dict["sequences"], candidate_dict["indexes"] = res_seq, res_index
        clean_OCRs_and_candidates.append(candidate_dict)
    return clean_OCRs_and_candidates

def select_text(OCRs_and_candidates, zone): # Very case by case function ; COULD BE IMPROVE WITH RECURSIVITY
    final_OCRs_and_text_dict = []
    final_text_list, proba_text_list = [], []
    wordstrip = ' "|\_!§<>{}—;‘’'
    lstrip = "=-,°'“*§:/ .])"
    rstrip = "= -,°'“*§:/ (["
    found_box = []
    for candidate_dict in OCRs_and_candidates: # Agregate sequence as string and parse multi response per dict as single response (rare case)
        if candidate_dict["type"] == "found":
            found_box = candidate_dict["box"]
        if found_box != [] and candidate_dict["type"] != "found":
            candidate_dict["box"] = found_box 
            candidate_dict["type"] = "replaced_found"
        if candidate_dict["sequences"] == [] and not [] in final_text_list:
            final_text_list.append([])
            final_OCRs_and_text_dict.append(candidate_dict)
        if zone == "parasite_recherche":
            candidate_dict["choice"] = "parasite_list"
            return candidate_dict
        else : 
            # Get sequence as simple string ON A LIST (not as a list of strings)
            for sequence, index in zip(candidate_dict["sequences"], candidate_dict["indexes"]):
                res_seq, res_index = [], []
                res_dict = deepcopy(candidate_dict)
                if type(sequence) == type([]): # The only one elmnt is a list -> Make a single string
                    kept_i = [i for i in range(len(sequence)) if sequence[i] not in wordstrip] # Select non-wordstrip indices only
                    res_seq = [" ".join([sequence[i].strip(" ") for i in kept_i]).strip(wordstrip).lstrip(lstrip).rstrip(rstrip)]
                    res_index = [index[i] for i in kept_i]
                elif type(sequence) == type(""): # The only one element is a string -> Take it
                    res_seq = [sequence.strip(wordstrip)]
                    res_index = index
                else :
                    print("rare case")
                    res_seq = [sequence]
                    res_index = index
                    
                if res_seq in final_text_list: # If the sequence has already been seen return the dict
                    res_dict["choice"] = "Two times the same sequence"
                    res_dict["sequences"], res_dict["indexes"] = res_seq, res_index
                    return res_dict
                
                proba = [candidate_dict["OCR"]["conf"][i] for i in res_index]
                res_dict["sequences"], res_dict["indexes"] = res_seq, res_index
                if len(proba)>0:
                    proba_text_list.append(sum(proba)/len(proba))
                else : proba_text_list.append(0)  
                final_text_list.append(res_seq)
                final_OCRs_and_text_dict.append(res_dict)
            
    if final_text_list == [[]]: # No response
        final_OCRs_and_text_dict[0]["choice"] = "empty"
        return final_OCRs_and_text_dict[0]
    if len(final_OCRs_and_text_dict) == 1: # One non-null res
        final_OCRs_and_text_dict[0]["choice"] = "one choice"
        return final_OCRs_and_text_dict[0]
    
    sorted_by_proba = sorted(zip(proba_text_list, final_OCRs_and_text_dict), key=lambda x: x[0], reverse=True) # Return the best result according to the OCR prboability
    conf, final_dict = zip(*sorted_by_proba)
    final_dict = final_dict[0]
    final_dict["choice"] = f"Best confidance {conf}"
    return final_dict

def get_wanted_text(cropped_image, landmarks_dict, format, JSON_HELPER=OCR_HELPER, ocr_config=TESSCONFIG):
    res_dict_per_zone = {}
    for zone, key_points in JSON_HELPER[format].items():
        landmark_boxes =  landmarks_dict[zone]["landmark"]
        conditions =  key_points["conditions"]
        
        if (format, zone) == ("check", "type_lot"):
            checkbox_dict = JSON_HELPER["checkbox"][format][zone]
            candidate_OCR_list_filtered =  get_checkbox_check_format(format, checkbox_dict, cropped_image, landmark_boxes, key_points["relative_position"])
            
        else:
            if zone == "N_d_echantillon": ocr_config[2] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            else : ocr_config[2] = whitelist
            candidate_OCR_list = get_candidate_local_OCR(cropped_image, landmark_boxes, key_points["relative_position"], format, ocr_config=ocr_config)
            candidate_OCR_list_filtered = condition_filter(candidate_OCR_list, key_points["key_sentences"], conditions)
        clean_OCRs_and_candidates = common_mistake_filter(candidate_OCR_list_filtered, zone)
        
        if (format, zone) == ("table", "parasite_recherche"):
            checkbox_dict = JSON_HELPER["checkbox"][format][zone]
            clean_OCRs_and_candidates = get_checkbox_table_format(checkbox_dict, clean_OCRs_and_candidates, cropped_image)
    
        OCR_and_text_full_dict = select_text(clean_OCRs_and_candidates, zone) # Normalize and process condition text (ex : Somes are simple lists other lists of lists...)            
        
        if OCR_and_text_full_dict["sequences"] != [] and zone != "parasite_recherche":
             OCR_and_text_full_dict["sequences"] =  OCR_and_text_full_dict["sequences"][0] # extract the value
        if OCR_and_text_full_dict["indexes"] != [] :
            if type(OCR_and_text_full_dict["indexes"][0]) == type([]):
                OCR_and_text_full_dict["indexes"] = OCR_and_text_full_dict["indexes"][0]
                
        res_dict_per_zone[zone] = OCR_and_text_full_dict                    
    return res_dict_per_zone 

if __name__ == "__main__":

    print("start")
    path = r"C:\Users\CF6P\Desktop\cv_text\Data\scan7.pdf"
    images = PDF_to_images(path)
    images = images[6:]
    res_dict_per_image = {}
    for i, image in enumerate(images,1):
        print(f"\n -------------{i}----------------- \nImage {i} is starting")
        processed_image = binarized_image(image)
        format, cropped_image = crop_image_and_sort_format(processed_image, show=False)
        # plt.imshow(cropped_image)
        # plt.show()
        print(f"Image with format : {format} is cropped.")
        OCR_data, landmarks_dict = get_data_and_landmarks(format, cropped_image)
        print(f"Landmarks are found.")
        landmark_and_text_dict = get_wanted_text(cropped_image, landmarks_dict, format)
        
        non_empty_field = 0
        for _, value_dict in landmark_and_text_dict.items():
            if value_dict["sequences"] != []:
                non_empty_field+=1
                
        print(non_empty_field)
        if format == "table" and non_empty_field < 2:
            format = "hand"
            OCR_data, landmarks_dict = get_data_and_landmarks(format, cropped_image)
            landmark_and_text_dict = get_wanted_text(cropped_image, landmarks_dict, format)
            
        res_dict_per_image[i] = landmark_and_text_dict
    