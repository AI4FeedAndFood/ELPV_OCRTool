import numpy as np
import json
import cv2
import matplotlib.pyplot as plt
from copy import deepcopy

import os
import sys 

import locale
locale.setlocale(locale.LC_TIME,'fr_FR.UTF-8')
from datetime import datetime
year = datetime.now().year

from Model_Fredon import get_key_matches_and_OCR, ZoneMatch
from ConditionFilter import condition_filter
from ProcessPDF import binarized_image, HoughLines, get_format_and_adjusted_image, get_rectangles

if 'AppData' in sys.executable:
    application_path = os.getcwd()
else : 
    application_path = os.path.dirname(sys.executable)

format = "SEMAE"

OCR_HELPER_JSON_PATH  = os.path.join(application_path, "CONFIG\OCR_config.json")
OCR_HELPER = json.load(open(OCR_HELPER_JSON_PATH, encoding="utf-8"))
OCR_PATHES = OCR_HELPER["PATHES"]

class Column:
    """
    A class defining a template
    """
    def __init__(self, name, landmark_box, dict):
        """
        Args:
            name (str): path of the template image path
            absolute_position (str): the label corresponding to the template
            theorical_col (List[float]): ratio of the horiontal position of the col compared with the image shape
        """
        self.name = name
        self.landmark_boxes = landmark_box
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
    blurred = cv2.GaussianBlur(image, (11,11), 0)
    HSV_image = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    return HSV_image

def _postprecess_mask(mask):
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
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

def get_bounding_boxes(contours, Yc, Xc): # Finally get bounding box
    rectangles = []
    centers  = []
    for contour in contours:
        rectangles.append(cv2.boundingRect(contour)) # Carry ((x,y), (width, height), rotation) with xy of the center
        M = cv2.moments(contour)
        if int(M["m00"])!=0:
            x = int(M["m10"] / M["m00"])
            y = int(M["m01"] / M["m00"]) # More like a weighted center, tends to be more in the center of objects
            if (y<Yc*0.05 or y>0.95*Yc) and (Xc*0.1<x<Xc*0.9):
                centers.append([x,y])

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
    # Binarized 
    bin_image = binarized_image(cropped_image)
    rectangles = get_rectangles(bin_image)

    _, cropped_image = get_format_and_adjusted_image(bin_image, rectangles, cropped_image, input_format="SEMAE", show=False)

    # Get dots thanks to HSV filtering
    HSV_im = HSV_image(cropped_image)

    contours, mask  = get_contours(HSV_im, conditions_dict)
    cropped_image = binarized_image(cropped_image)
    cropped_image[mask == 255] = 255

    # Apply the rotation for image and dots according to the orientation
    Yc, Xc = cropped_image.shape[:2]

    dots = get_bounding_boxes(contours, Yc, Xc)
    if dots == []:
        print("PAS DE POINT TROUVE")
        return [], cropped_image, 1
    else:
        print(f"-> {len(dots)} POINT(S) TROUVES <-")
    _, centers = zip(*dots)
    centers = list(centers)

    k_90 = 3
    if centers[0][1]<Yc/8: # Dots indicates if the paper is from top to bottom or bottom to top
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

    # plt.imshow(final_image)
    # plt.show()

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

def split_with_line(vertical_lines, OCR):
    """Split a text box in two boxes if a vertical line pass trough the box.
    Split the box at the lin position and the text at a space.

    Args:
        vertical_lines
        full_img_OCR (list of dict)
    """
    def _res_loop(vertical_lines, dict_seq):
        text = dict_seq["text"]
        for line in vertical_lines:
            if xmin+w/6<line[0][0]<xmin+w*5/6: # The line split the box around the middle its middle
                join_text = " ".join(text)
                frac_space = [i/(len(join_text)-1) for i, el in enumerate(join_text) if el==" "]
                frac_line = (line[0][0]-xmin)/(xmax-xmin)
                i_split = int(min(frac_space, key=lambda x: abs(x-frac_line)) * len(join_text))
                res1 = deepcopy(dict_seq)
                res1["text"] =  [join_text[:i_split+1].strip(" :,'")]
                res1["box"] = [xmin, ymin, line[0][0]-2, ymax]
                res2 = deepcopy(dict_seq)
                res2["text"] =  [join_text[i_split+1:].strip(" :,'")]
                res2["box"] = [line[0][0]+2, ymin, xmax, ymax]
                return [res1, res2]
            
        return [dict_seq]

    res_ocr = []
    for dict_seq in OCR:
        xmin, ymin, xmax, ymax = dict_seq["box"]
        w = xmax-xmin
        if len(dict_seq["text"])>1:
            res_ocr += _res_loop(vertical_lines, dict_seq)
        else:
            res_ocr.append(dict_seq)

    return res_ocr

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

def text_cell(full_OCR, cell_box, column):

    def _get_candidate(full_OCR):
        candidate_dicts = [dict_sequence for dict_sequence in full_OCR if 
                      (xmin<(dict_sequence["box"][0]+dict_sequence["box"][2])*0.5<xmax) and 
                      ((ymin<dict_sequence["box"][1]<ymax) or (ymin<dict_sequence["box"][3]<ymax))]
        return candidate_dicts
    
    xmin,ymin,xmax,ymax = cell_box
    candidate_dicts = _get_candidate(full_OCR)
    match_indices, res_seq = condition_filter(candidate_dicts, column.dict["conditions"], model=format, application_path=application_path, ocr_pathes=OCR_PATHES)
    conf = min([candidate_dicts[i]["proba"] for i in match_indices]) if match_indices else 0
    zone_match = ZoneMatch(candidate_dicts, match_indices, conf, res_seq)

    if column.name != "parasite_recherche":
        zone_match.res_seq = " ".join(zone_match.res_seq).upper().lstrip(" ._-!*:-")
    if zone_match.res_seq == "PAYS BAS":
        zone_match.res_seq = "PAYS-BAS"

    return zone_match
    
def ProcessLandscape(image,image_name):
    # Find dots in the scan and the flip&croped image
    # plt.imshow(image)
    # plt.show()
    dots, processed_image, k_90 =  get_dots_and_final_image(image, OCR_HELPER["landscape_HSV"])

    if dots == []:
        print("No dot")
        return {}
    Y,X = processed_image.shape[:2]
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
    zone_key_match_dict, full_img_OCR = get_key_matches_and_OCR(format, processed_image)
    # Now let's concider text around dots
    full_img_OCR = split_with_line(vertical_lines, full_img_OCR)
    # Get columns frame vertical lines
    columns = []
    for zone, key_points in OCR_HELPER[format].items():
        match_zone = zone_key_match_dict[zone]
        landmark_box =  match_zone.OCR["box"]
        col = Column(name=zone, landmark_box=landmark_box, dict=key_points)
        
        if match_zone.confidence>-1:
            col.status = "found"
            xy_col = [landmark_box[0], landmark_box[1]] # Frame the upper left corner  
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
    
    samples_res_dict = {}
    for i_point, point_position in enumerate(dots):
        _, y_point = point_position
        res_dict_per_zone = {}
        for col in columns:
            upper_line, lower_line = get_frame_lines(point_position, extend_lines(lines["horizontal"], mode="horizontal"), mode="horizontal")
            left_line, right_line = col.left_line, col.right_line
            if col.dict["merged"]:
                x_left, x_right = left_line[0][0], right_line[0][0]
                xy_left_mid_right = [[alpha*x_left + (1-alpha)*x_right, y_point] for alpha in [0.95, 0.75, 0.62, 0.5, 0.37, 0.25, 0.05]]
                upper_frame, lower_frame = [], []
                for position in xy_left_mid_right:
                    # print(position)
                    upper_merge, lower_merge = get_frame_lines(position, lines["horizontal"], mode="horizontal", var_match=True)
                    upper_frame.append(upper_merge)
                    lower_frame.append(lower_merge)
                # print("u :", upper_frame)
                # print("l :", lower_frame)
                try:
                    upper_line = max(upper_frame, key=lambda x: x[0][1])
                except:
                    upper_line = []
                try:
                    lower_line = min(lower_frame, key=lambda x: x[0][1])
                except:
                    lower_line = []
                    
            try:
                cell_box = [int(left_line[0][0]), int(upper_line[0][1]), int(right_line[0][0]), int(lower_line[0][1])] # x,y,x2,y2
                # delete_HoughLines(processed_image, [upper_line, lower_line, left_line, right_line], show=True)
                zone_match = text_cell(full_img_OCR, cell_box, column=col)

                res_dict_per_zone[col.name] = {
                    "sequence" : zone_match.res_seq,
                    "confidence" : float(zone_match.confidence),
                    "area" : cell_box,
                    "format" : format
                }
            except:
                res_dict_per_zone[col.name] = {
                    "sequence" : [],
                    "confidence" : 0,
                    "area" : [],
                    "format" : format
                }
            print(col.name, " : ", res_dict_per_zone[col.name]["sequence"])


        res_dict_per_zone["analyse"] ={
                "sequence" : ["Rhizomanie"],
                "confidence" : int(1),
                "area" : [],
                "format" : format
            }
        
        samples_res_dict[f"Point_{i_point+1}/{len(dots)}"] = {"IMAGE" :image_name,
                                                 "k_90": k_90,
                                                  "EXTRACTION" : res_dict_per_zone
                                                }

    return samples_res_dict

def main(scan_dict):
    pdfs_res_dict = {}

    for pdf, images_dict in scan_dict.items():
        print("###### Traitement de :", pdf, " ######")
        pdfs_res_dict[pdf] = {}
        for i_image, (image_name, sample_image) in enumerate(list(images_dict.items())):
            print("------", image_name, "------")
            pdfs_res_dict[pdf].update(ProcessLandscape(sample_image, image_name))
   # print(pdfs_res_dict)
    return pdfs_res_dict

if __name__ == "__main__":
    from LaunchTool import getAllImages

    print("start")
    path = r"C:\Users\CF6P\Desktop\ELPV\Data\test1"
    scan_dict = getAllImages(path)
    
    main(scan_dict)
            

