import numpy as np
import cv2
import json
import matplotlib.pyplot as plt

from ProcessCheckboxes import visualize, checkbox_match, non_max_suppression, Template

OCR_HELPER_JSON_PATH  = r"CONFIG\\OCR_config.json"
CONFIG_DICT = json.load(open(OCR_HELPER_JSON_PATH, encoding="utf-8"))["PATHES"]["checkbox"]

empty_checkbox_path = CONFIG_DICT["Fredon avec cases"]["type_lot"]["empty_path"]
cross_checkbox_path = CONFIG_DICT["Fredon avec cases"]["type_lot"]["cross_path"]
table_checkbox_path = CONFIG_DICT["Fredon tableau"]["cross_path"]

TRANSFORM = [lambda x: x, lambda x: cv2.flip(x,0), lambda x: cv2.flip(x,1), lambda x: cv2.resize(x, (int(x.shape[1]*1.15), x.shape[0])),
             lambda x: cv2.resize(x, (x.shape[1], int(x.shape[0]*1.15)))] # Maybe can be cleaner with a transform class

CHECK_TEMPLATES = [Template(image_path=cross_checkbox_path, label="cross", color=(0, 0, 0), matching_threshold=0.70, transform_list=TRANSFORM),
              Template(image_path=empty_checkbox_path, label="empty", color=(0, 0, 0), matching_threshold=0.6, transform_list=TRANSFORM)]

TABLE_TEMPLATE = [Template(image_path=table_checkbox_path, label="table", color=(0, 0, 0), matching_threshold=0.70, transform_list=TRANSFORM)]
    

class Match :
    """
    A class defining an match. Usefull for detection
    """
    def __init__(self, label, img_matches, match_value=None):
        """
        Args:
            label : Format label
            match_value : Mean of the 40 best matches values
        """
        self.label = label
        self.match_value = match_value
        self.color = (0, 191, 255)
        self.img_matches = img_matches

def binarized_image(image):
    gray = cv2.cvtColor(image.astype('uint8'), cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    return thresh

### FEATURE MATCHING #####

def ORB_features_extraction(image, nfeatures=2000):

  
    # Applying the function 
    orb = cv2.ORB_create(nfeatures=nfeatures) 
    key_points, descriptors = orb.detectAndCompute(image, None)
    
    # Drawing the keypoints 
    # kp_image = cv2.drawKeypoints(image, key_points, None, color=(0, 255, 0), flags=0) 
    # plt.imshow(kp_image) 
    # plt.show()

    return key_points, descriptors

def BF_FeatureMatcher(desc_img, desc_temp, thresh=1.99, n_matches=40):

    brute_force = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
    matches = brute_force.match(desc_temp, desc_img)
 
    # finding the humming distance of the matches and sorting them
    ordered_matches = sorted(matches, key=lambda x:x.distance)
    
    best_matches = []
    for index in range(len(ordered_matches)-1):
        if ordered_matches[index].distance < thresh*ordered_matches[index+1].distance:
            best_matches.append(ordered_matches[index])

    best_matches = best_matches[:n_matches] if len(best_matches)>=n_matches else best_matches

    return best_matches

def BF_FeatureKNNMatcher(desc_img, desc_temp, thresh=1, n_matches=40):

    brute_force = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=False)
    matches = brute_force.knnMatch(desc_temp, desc_img, k=2)
 
    # finding the humming distance of the matches and sorting them
    ordered_matches = sorted(matches, key=lambda x:x[0].distance)
    best_matches = []
    for m,n in ordered_matches:
        if m.distance < thresh*n.distance:
            best_matches.append(m)
    best_matches = best_matches[:n_matches] if len(best_matches)>=n_matches else best_matches

    return best_matches

##########################

def exctract_lines(bin_image):
    lines_img = 255*np.ones(bin_image.shape)

    def _HoughLines(bin_image, mode="vertical"):
        (cst, _) = (0,1) if mode == "vertical"  else (1,0) # The specified axis is the constant one
        image = bin_image.copy()
        ksize = (1,6) if mode == "vertical" else (6,1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=ksize)
        bin_image = cv2.dilate(bin_image, kernel, iterations=4)
        # plt.imshow(bin_image)
        # plt.show()
        edges = cv2.Canny(bin_image,50,150,apertureSize=3)
    
        # Apply HoughLinesP method to
        # to directly obtain line end points
        lines_list =[]
        lines = cv2.HoughLinesP(
                    edges, # Input edge image
                    1, # Distance resolution in pixels
                    np.pi/180, # Angle resolution in radians
                    threshold=60, # Min number of votes for valid line
                    minLineLength=500, # Min allowed length of line
                    maxLineGap=290 # Max allowed gap between line for joining them ; Set according to the SEMAE format
                    )
        
        for points in lines:
            # Extracted points nested in the list
            x1,y1,x2,y2=points[0]
            line  = [(x1,y1),(x2,y2)]
            if abs(line[0][cst]-line[1][cst])<40:
                lines_list.append(line)
        
        return lines_list 

    lines = {
        "vertical" : _HoughLines(bin_image),
        "horizontal" : _HoughLines(bin_image, mode="horizontal")
        }

    for line in lines["vertical"]+lines["horizontal"]:
        [(x1,y1),(x2,y2)] = line
        cv2.line(lines_img,(x1,y1),(x2,y2),0,3)

    return lines_img

def sortFormat(bin_image, rectangles, show=False):
    """
    
    """
    best_format = Match("", 100)

    if len(rectangles)==4:
        best_format.label = "Fredon non officiel"

    if len(rectangles)==3:
        best_format.label = "Fredon tableau"

    if len(rectangles)== 2:
        y_im, x_im = bin_image.shape[:2]
        detections = checkbox_match(TABLE_TEMPLATE, bin_image)
        filtered_detection = non_max_suppression(detections)

        if show :
            visualize(bin_image, filtered_detection)
        count = 0
        for checkbox in filtered_detection:
            x,y = checkbox["TOP_LEFT_X"], checkbox["TOP_LEFT_Y"] # Filter by the position of found boxes
            if x_im*(1/3)<x<x_im*(2/3) and y_im*(1/4)<y<y_im:
                count+=1
        if count>0: # Threshold choosen arbitrary
            best_format.label = "Fredon tableau"
        else:
            best_format.label = "Fredon avec cases"
    
    if len(rectangles) == 1:

        y_im, x_im = bin_image.shape[:2]
        detections = checkbox_match(CHECK_TEMPLATES, bin_image)
        filtered_detection = non_max_suppression(detections)

        if show :
            visualize(bin_image, filtered_detection)

        count = 0
        for checkbox in filtered_detection:
            x,y = checkbox["TOP_LEFT_X"], checkbox["TOP_LEFT_Y"] # Filter by the position of found boxes
            if x<x_im/2 and y_im*(1/5)<y<y_im*(4/5):
                count+=1
        if count>5: # Threshold choosen arbitrary
            best_format.label = "Fredon avec cases"
        else:
            best_format.label = "Fredon sans case"
    return best_format.label

if __name__ == "__main__":

    from ProcessPDF import PDF_to_images, get_rectangles
    print("start")
    path = r"C:\Users\CF6P\Desktop\ELPV\Data\scan3.pdf"
    images = PDF_to_images(path)
    start = 0
    images = images[1:]
    res_dict_per_image = {}
    for i, image in enumerate(images,start+1):
        print(f"\n -------------{i}----------------- \nImage {i} is starting")
        bin_image = binarized_image(image)
        bin_image = binarized_image(image)
        rectangles = get_rectangles(bin_image)
        sortFormat(bin_image, rectangles, show=True)