import numpy as np
import pytesseract
import json
import cv2
import matplotlib.pyplot as plt
from TextExtraction import get_data_and_landmarks, get_candidate_local_OCR, condition_filter


import locale
locale.setlocale(locale.LC_TIME,'fr_FR.UTF-8')
from datetime import datetime
year = datetime.now().year

from ProcessPDF import binarized_image, HoughLines
from TextExtraction import get_candidate_local_OCR, common_mistake_filter, select_text

whitelist =  "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz(),:.-/°&=àéçëôùê''"
pytesseract.pytesseract.tesseract_cmd = r'exterior_program\Tesseract4-OCR\tesseract.exe'
LANG = 'eng+eng2'
TESSCONFIG = [1, 6, whitelist, LANG]
OCR_HELPER_JSON_PATH  = r"CONFIG\\OCR_config.json"
OCR_HELPER = json.load(open(OCR_HELPER_JSON_PATH, encoding="utf-8"))

format = "landscape"

class Column:
    """
    A class defining a template
    """
    def __init__(self, name, landmark_boxes, dict):
        """
        Args:
            name (str): path of the template image path
            absolute_position (str): the label corresponding to the template
            theorical_col (List[float]): ratio of the horiontal position of the col compared with the image shape
        """
        self.name = name
        self.landmark_boxes = landmark_boxes
        self.dict = dict
        self.status = "default"
        self.left_line = None
        self.right_line = None
    
class Point:
    """
    A class defining a template
    """
    def __init__(self, number, absolute_position, upper_line, lower_line):
        """
        Args:
            name (str): path of the template image path
            absolute_position (str): the label corresponding to the template
            theorical_col (List[int]): the color associated with the label (to plot detections)
        """
        self.number = number
        self.absolute_position = absolute_position
        self.upper_line = upper_line
        self.lower_line = lower_line
        self.res_dict = {}

def HSV_image(image):
    blurred = cv2.GaussianBlur(image, (23,23), 0)
    HSV_image = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    return HSV_image

def _postprecess_mask(mask):
    mask = cv2.erode(mask, None, iterations=4)
    mask = cv2.dilate(mask, None, iterations=4)
    return mask

def get_contours(image, conditions_dictionnary):
    
    def _area_filter(res_contours, conditions_dictionnary):
        area_threshold = conditions_dictionnary["area_threshold"]
        res_contours = [x for x in res_contours if cv2.contourArea(x)<area_threshold]
        return res_contours

    # upper mask (110-180)
    lower_red = np.array([110,50,50])
    upper_red = np.array([180,255,255])
    mask1 = cv2.inRange(image, lower_red, upper_red)

    mask = _postprecess_mask(mask1) # Mask is processed
    # plt.imshow(mask)
    # plt.show()
    contours,_ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  #Get contours
    if len(contours) == 0:
        print("No dot found")
        return [], None
    filtered_contours = _area_filter(contours, conditions_dictionnary) # Contours are selected
    return filtered_contours, mask

def get_bounding_boxes(contours): # Finally get bounding box
    rectangles = []
    centers  = []
    for contour in contours:
        rectangles.append(cv2.boundingRect(contour)) # Carry ((x,y), (width, height), rotation) with xy of the center
        M = cv2.moments(contour)
        if int(M["m00"])!=0:
            centers.append([int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])]) # More like a weighted center, tends to be more in the center of objects
        # TO FIX
    return list(sorted(zip(rectangles, centers), key=lambda x: (round(x[1][1]/100), round(x[1][0]/100)))) # Get boxes from upper left to upper right, the round by 100 is based on 3888*3888 images

def single_object_drow_boxes(processed_image, rectangles_centers_list, color=(255,255,255), object = "Detection"):
    for i, (rectangle, center) in enumerate(rectangles_centers_list, 1):
        x,y,height,width, = rectangle
        cv2.rectangle(processed_image, (x,y), (x+height, y+width), (0, 255, 255), 2)
        cv2.circle(processed_image, center, 5, (0, 255, 255), -1) # point in center
        cv2.putText(processed_image, f"{object}_{i}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 3, color, 6)
    return processed_image 

def get_dots_and_final_image(cropped_image, conditions_dict):
    """_summary_

    Args:
        cropped_image (_type_): _description_
        conditions_dict (_type_): _description_

    Returns:
        _type_: _description_
    """

    # Get dots thanks to HSV filtering
    HSV_im = HSV_image(cropped_image)
    contours, mask  = get_contours(HSV_im, conditions_dict)
    # Binarized then delete dots
    cropped_image = binarized_image(cropped_image)
    cropped_image[mask == 255] = 255
    dots = get_bounding_boxes(contours)
    if dots == []:
        print("PAS DE POINT TROUVE")
        return [], cropped_image, 1
    _, centers = zip(*dots)
    centers = list(centers)
    centers = [list(c) for c in centers]
       
    # Apply the rotation for image and dots according to the orientation
    Yc, Xc = cropped_image.shape[:2]
    k_90 = 3
    if centers[0][1]<100: # Dots indicates if the paper is from top to bottom or bottom to top
        k_90 = 1
    elif Yc<Xc:
        k_90=0

    # Apply the crop and rotate to centers
    # angle = angle if inv else -angle
    M_after_crop = cv2.getRotationMatrix2D((0,0), -90*k_90, 1) # Rotation matrix for centers       
    for i, dot in enumerate(centers):
        new_dot = np.matmul(np.array(dot)-np.array([Xc/2, Yc/2]), M_after_crop)[:2] + np.array([Yc/2, Xc/2])
        centers[i] = [max(int(new_dot[0]), 0), int(new_dot[1])] if k_90!=0 else centers[i]
    centers = sorted(centers, key=lambda x: x[1])
    final_image = np.rot90(cropped_image, k_90) # Minus for counterclockwise

    # bloc1, bloc2 = [], []
    # ref1, ref2 = centers[0][0], centers[-1][0] # Split along x axis
    # for dot in centers:
    #     processed_image[dot[1]-2:dot[1]+2, dot[0]-2:dot[0]+2] = 0
    #     if abs(dot[0]-ref1) < 100:
    #         bloc1.append(dot)
    #     if abs(dot[0]-ref2) < 100: # Prevent unwanted detection, better then a else
    #         bloc2.append(dot)
    # dot_pairs = (bloc1, bloc2)
    
    return centers, final_image, k_90

def extend_lines(lines_list, mode="vertical"):
    """
    Extend Hough line along the "mode" axis
    """
    lines_list_c = lines_list.copy()
    (cst, var) = (0,1) if mode == "vertical"  else (1,0) # The specified axis (indicated with a 0) is the constant one (is just fomrmalism)
    clean_lines = []
    merged_line = []
    # Merge discontinious lines
    for i in range(len(lines_list_c)):
        if not i in merged_line :
            line_i = lines_list_c[i]
            for j in range(i+1, len(lines_list_c)):
                if not j in merged_line:
                    line_j = lines_list_c[j]
                    if abs((line_i[0][cst]+line_i[1][cst])/2 - (line_j[0][cst]+line_j[1][cst])/2)<10:
                        res_line = [[0,0], [0,0]]
                        merged_line.append(j)
                        axis_mean = int((line_i[0][cst]+line_i[1][cst])/2)
                        if line_i[0][var] < line_i[1][var]:
                            min_naxis = min(line_i[0][var] , line_j[0][var])
                            max_naxis = max(line_i[1][var] , line_j[1][var])
                            res_line[0][var], res_line[1][var] =  min_naxis, max_naxis
                        else: 
                            min_naxis = min(line_i[1][var] , line_j[1][var])
                            max_naxis = max(line_i[0][var] , line_j[0][var])
                            res_line[1][var], res_line[0][var] =  min_naxis, max_naxis
                        res_line[0][cst], res_line[1][cst] = axis_mean, axis_mean
                        lines_list_c[i] = res_line
                        line_i = lines_list_c[i]
                else: pass
            if abs(line_i[0][var] - line_i[1][var])>150:
                clean_lines.append(lines_list_c[i])
    return clean_lines

def delete_HoughLines(image, lines, show=False):
    img = image.copy()
    value = 0 if show else 255
    for line in lines:
        [(x1,y1),(x2,y2)] = line
        cv2.line(img,(x1,y1),(x2,y2),value,6)
    
    if show:
        plt.imshow(img)
        plt.show()
        
    return img

def get_frame_lines(position, lines , mode="vertical", var_match=False):
    """
    Get the "mode" line that frame the object of position "position"

    Args:
        position (list): Accept [x,y,...] or [(x,y),(...)] ONLY
        lines (list): output of the HoughLines function ; [[(x1,y1),(x2,y2)], ...]
        mode (str, optional): The orientation of lines. Defaults to "vertical".
        var_match (bool, optional): If True, select lines that the var exis cross the position. Defaults to "False".
    """
    (cst, var) = (0,1) if mode == "vertical"  else (1,0) # vertical means the x axis (index 0) is cst
    xy_position = position[0] if type(position[0])==type((0,0)) else position[:2]
    if var_match:
        lines = [line for line in lines if min(line[0][var], line[1][var])<=xy_position[var]<=max(line[0][var], line[1][var])]
    shift = lambda x: x[0][cst]-xy_position[cst]
    linf = [line for line in lines if shift(line)<=0]
    lsup = [line for line in lines if shift(line)>=0]
    
    first_line = min(linf, key=lambda x: -shift(x)) if len(linf) !=0 else []
    second_line = min(lsup, key=lambda x: shift(x)) if len(lsup) !=0 else []
        
    return first_line, second_line    

def ProcessLandscape(image):
    # Find dots in the scan and the flip&croped image
    dots, processed_image, k_90 =  get_dots_and_final_image(image, OCR_HELPER["landscape_HSV"])
    Y,X = processed_image.shape[:2]
    if dots == []:
        print("No dot")
        return {}
    # Find lines in the scan
    vertical_lines = HoughLines(processed_image)
    # Delete lines that are not from the table
    vertical_lines = [line for line in vertical_lines if max(line[0][1], line[1][1])>2*processed_image.shape[0]/3]
    lines = {
        "vertical" : extend_lines(vertical_lines),
        "horizontal" : HoughLines(processed_image, mode="horizontal")
        }
    # Clean the image from the scan to increase OCR performances
    processed_image = delete_HoughLines(processed_image, lines["vertical"]+lines["horizontal"])
    # delete_HoughLines(processed_image, lines["vertical"]+lines["horizontal"], show=True)
    # Find columns header
    _, landmarks_dict = get_data_and_landmarks(format, processed_image)
    # Get columns frame vertical lines
    columns = []
    for zone, key_points in OCR_HELPER[format].items():
        landmark_boxes =  landmarks_dict[zone]["landmark"]
        col = Column(name=zone, landmark_boxes=landmark_boxes, dict=key_points)
        
        if "found" in [box[0] for box in landmark_boxes]:
            col.status = "found"
            f_index = [box[0] for box in landmark_boxes].index("found")
            box = landmark_boxes[f_index][1]
            xy_col = [box[0]+box[2]/4, box[1]+box[3]/4] # Frame the upper left corner  
            vert_frame_lines = get_frame_lines(xy_col, lines["vertical"])
        else : 
            vert_frame_lines = [[], []]
        for i, line in enumerate(vert_frame_lines):
            if line == []:
                x_line = int(X*key_points["theorical_col"][i])
                vert_frame_lines = list(vert_frame_lines)
                vert_frame_lines[i] = [[x_line, 0],[x_line, Y]]
            col.left_line, col.right_line = vert_frame_lines
        columns.append(col)
    
    landscape_dict_res = {}
    for i_point, point_position in enumerate(dots):
        _, y_point = point_position
        res_dict_per_zone = {}
        for col in columns:
            upper_line, lower_line = get_frame_lines(point_position, extend_lines(lines["horizontal"], mode="horizontal"), mode="horizontal")
            left_line, right_line = col.left_line, col.right_line
            if col.dict["merged"]:
                x_left, x_right = left_line[0][0], right_line[0][0]
                xy_left_mid_right = [[alpha*x_left + (1-alpha)*x_right, y_point] for alpha in [0.85, 0.75, 0.62, 0.5, 0.37, 0.25, 0.15]]
                upper_frame, lower_frame = [], []
                for position in xy_left_mid_right:
                    upper_merge, lower_merge = get_frame_lines(position, lines["horizontal"], mode="horizontal", var_match=True)
                    upper_frame.append(upper_merge)
                    lower_frame.append(lower_merge)
                upper_line = max(upper_frame, key=lambda x: x[0][1]) if upper_frame != [] else [] 
                lower_line = min(lower_frame, key=lambda x: x[0][1]) if lower_frame != [] else []
            cell_box = [left_line[0][0], upper_line[0][1], abs(right_line[0][0]-left_line[0][0]), abs(lower_line[0][1]-upper_line[0][1])] # x,y,w,h
            # delete_HoughLines(processed_image, [upper_line, lower_line, left_line, right_line], show=True)
            candidate_OCR_list = get_candidate_local_OCR(processed_image, landmark_boxes=[["found", cell_box, 1]], relative_positions=[[[0,1], [0,1]]], format="landscape", ocr_config=TESSCONFIG)[0]
            candidate_OCR_list_filtered = condition_filter([candidate_OCR_list], col.dict["key_sentences"], col.dict["conditions"])
            clean_OCRs_and_candidates = common_mistake_filter(candidate_OCR_list_filtered, col.name)
            OCR_and_text_full_dict = select_text(clean_OCRs_and_candidates, col.name)
            
            if OCR_and_text_full_dict["indexes"] != [] :
                if type(OCR_and_text_full_dict["indexes"][0]) == type([]):
                    OCR_and_text_full_dict["indexes"] = OCR_and_text_full_dict["indexes"][0]
            if OCR_and_text_full_dict["sequences"] != [] and zone != "parasite_recherche":
                OCR_and_text_full_dict["sequences"] =  OCR_and_text_full_dict["sequences"][0] # extract the value
                
            res_dict_per_zone[col.name] = OCR_and_text_full_dict
            print(col.name, " : ", OCR_and_text_full_dict["sequences"])
        res_dict_per_zone["parasite_recherche"] = {
            "OCR" : {"conf" : [1]},
            "sequences" : [['Rhizomanie']],
            "indexes" : [0]
            }
        
        landscape_dict_res[f"point_{i_point}_k90_{k_90}"] = res_dict_per_zone

    return landscape_dict_res