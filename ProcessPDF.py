import numpy as np
import cv2
import pdf2image
import matplotlib.pyplot as plt

# Link to the poppler wich open pdfs
POPPLER_PATH = r"exterior_program\poppler\poppler-23.01.0\Library\bin"
    
def PDF_to_images(path, POPPLER=POPPLER_PATH):
    """ 
    Open the pdf and return all pages as a list of array
    Args:
        path (path): python readable path
        POPPLER (path): Defaults to POPPLER_PATH.

    Returns:
        list of arrays: all pages as array
    """
    images = pdf2image.convert_from_path(path, poppler_path=POPPLER)
    return [np.array(image) for image in images]

def binarized_image(image):
    """ 
    Binarized one image thanks to OpenCV thersholding. niBlackThresholding has been tried.
    Args:
        image (np.array) : Input image

    Returns:
        np.array : binarized image
    """
    #image = image[3:-3, 3:-3]
    blur = cv2.bilateralFilter(image,5,200,200)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    return thresh

def get_iou(a, b, epsilon=1e-5):
    """ Given two boxes `a` and `b` defined as a list of four numbers:
            [x1,y1,x2,y2]
        where:
            x1,y1 represent the upper left corner
            x2,y2 represent the lower right corner
        It returns the Intersect of Union score for these two boxes.

    Args:
        a:          (list of 4 numbers) [x1,y1,x2,y2]
        b:          (list of 4 numbers) [x1,y1,x2,y2]
        epsilon:    (float) Small value to prevent division by zero

    Returns:
        (float) The Intersect of Union score.
    """
    # COORDINATES OF THE INTERSECTION BOX
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    # AREA OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)
    # handle case where there is NO overlap
    if (width<0) or (height <0):
        return 0.0
    area_overlap = width * height

    # COMBINED AREA
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_combined = area_a + area_b - area_overlap

    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined+epsilon)
    return iou

def _rect_table(processed_image, rectangles, format=""):
    
    def _process_rectangles(rectangles, format):        
        maxarea = 0
        for i, rect in enumerate(rectangles):
            if 45<rect[-1]:
                rectangles[i][1] = (rectangles[i][1][1], rectangles[i][1][0])
                if maxarea < rectangles[i][1][0]*rectangles[i][1][1]:
                    rot = 90-rect[-1]
            elif maxarea < rectangles[i][1][0]*rectangles[i][1][1]:
                rot = rect[-1] # The rot angle is chosen by taken the biggest rect angle
        
        xy_wh_rot = [[], [], []]
        for rect in rectangles:
            for comp in range(len(rect)):
                xy_wh_rot[comp].append(rect[comp])
              
        UX_LX_UY_LY = []
        for dist_i in [0,1]:
            for sens_j in [0,1]:
                if sens_j == 0 :
                    UX_LX_UY_LY.append(min([(rec[0][dist_i] - rec[1][dist_i]//2)+1 for rec in rectangles]))
                else :
                    UX_LX_UY_LY.append(max([(rec[0][dist_i] + rec[1][dist_i]//2)+1 for rec in rectangles]))
                    
        wh = (UX_LX_UY_LY[1]-UX_LX_UY_LY[0]+10, UX_LX_UY_LY[3]-UX_LX_UY_LY[2]+10)
        xy = (wh[0]//2 + UX_LX_UY_LY[0], wh[1]//2 + UX_LX_UY_LY[2])
        
        if format == "landscape":
            rot = -rot
            wh = (UX_LX_UY_LY[1]-UX_LX_UY_LY[0]+10, y)
        
        # im = processed_image.copy()
        # box = cv2.boxPoints([xy, wh, rot])
        # box = np.int0(box)
        # cv2.drawContours(im, [box], 0, (0,0,0), 20)
        # plt.imshow(im, cmap="gray")
        # plt.show()
        
        return (xy, wh, rot)
    
    y,x = processed_image.shape
    if len(rectangles)>2: # Clean if there is a black border of the scan wich is concider as a contour
        rectangles = [rect for rect in rectangles if not (0<x-rect[1][0]<10 or 0<y-rect[1][0]<10 or 0<x-rect[1][1]<0 or 0<y-rect[1][1]<10)]        
    rectangle = _process_rectangles(rectangles, format)

    return rectangle
    
def get_rectangle(processed_image, kernel_size=(3,3), def_format=""):
    """
    Extract the minimum area rectangle containg the text. 
    Thanks to that detect if the image is a TABLE format or not.
    Args:
        processed_image (np.array): The binarized images
        kernel_size (tuple, optional): . Defaults to (3,3).
        interations (int, optional): _description_. Defaults to 2.

    Returns:
        format (str) : "table" or "other"
        rectangle (cv2.MinAreaRect) : The biggest rectangle of text found in the image
    """
    y, x = processed_image.shape[:2]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    dilate = cv2.dilate(~processed_image, kernel, iterations=2)
    contours,_ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        print("No contour found : Crop is impossible, processed image is return.")
        return []

    rectangles = [cv2.minAreaRect(contour) for contour in contours]
    rectangles =  [list(rect) for rect in rectangles if rect[1][0]*rect[1][1]>x*y/12] if def_format == "landscape" else [list(rect) for rect in rectangles if rect[1][0]>x/3 and rect[1][1]>y/8]
    
    filtered_rects = []
    coord = lambda x: [int(x[0][0]-x[1][0]/2), int(x[0][1]-x[1][1]/2), int(x[0][0]+x[1][0]/2), int(x[0][1]+x[1][1]/2)]
    for rect in rectangles:
        if 45<rect[-1]:
                rect[1] = (rect[1][1], rect[1][0])
                rect[-1] = 90-rect[-1]
        overlap_found = False
        for f_rect in filtered_rects:
            coord1 = coord(rect)
            coord2 = coord(f_rect)
            iou = get_iou(coord1, coord2)
            if iou > 0.2 :
                overlap_found = True
                break
        if not overlap_found:
            filtered_rects.append(rect)
            
    if def_format == "landscape":
        format = "landscape"
        return def_format, _rect_table(processed_image, filtered_rects, format=format)
    
    if len(filtered_rects)==1:
        return "hand_or_check", filtered_rects[0]
    else:
        return "table", _rect_table(processed_image, filtered_rects)

def crop_and_rotate(processed_image, rect):
    """Crop the blank part around the found rectangle.

    Args:
        processed_image (np.array): The binarized image
        rect (cv2.MinAreaRect) : The biggest rectangle of text found in the image
    Returns:
        cropped_image (np.array) : The image cropped thanks to the rectangle
    """
    def _points_filter(points):
        """
        Get the endpoint along each axis
        """
        points[points < 0] = 0
        xpoints = sorted(points, key=lambda x:x[0])
        ypoints = sorted(points, key=lambda x:x[1])
        tpl_x0, tpl_x1 = xpoints[::len(xpoints)-1]
        tpl_y0, tpl_y1 = ypoints[::len(ypoints)-1]
        return tpl_y0[1], tpl_y1[1], tpl_x0[0], tpl_x1[0]
    
    if len(rect)==0 : 
        return processed_image

    box = np.intp(cv2.boxPoints(rect))    
    # Rotate image
    angle = rect[2]
    if 45<=angle<=90 : # Angle correction
        angle = angle-90 
    rows, cols = processed_image.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2,rows/2), angle, 1) # Rotation matrix
    img_rot = cv2.warpAffine(processed_image,M,(cols,rows))
    # rotate bounding box, then crop
    rect_points = np.intp(cv2.transform(np.array([box]), M))[0] # points of the box after rotation
    y0, y1, x0, x1 = _points_filter(rect_points) # get corners
    cropped_image = img_rot[y0:y1, x0:x1]
    return cropped_image

def delete_lines(bin_image): # Unused function wich delete (approximatly) lines on an image
    
    bin_image = np.array(bin_image).astype(np.uint8)
    copy = bin_image.copy()
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50,1))
    detected_lines = cv2.morphologyEx(bin_image, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    bin_image[detected_lines==255] = 0
    cnts = cv2.findContours(detected_lines, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(copy, [c], -1,(255, 255,255), 2)
    return copy

def HoughLines(bin_image, mode="vertical"):
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
                minLineLength=20, # Min allowed length of line
                maxLineGap=290 # Max allowed gap between line for joining them ; Set according to the SEMAE format
                )
    
    for points in lines:
        # Extracted points nested in the list
        x1,y1,x2,y2=points[0]
        line  = [(x1,y1),(x2,y2)]
        if abs(line[0][cst]-line[1][cst])<20:
            lines_list.append(line)
    return lines_list 

def agglomerate_lines(lines_list, mode = "vertical"):
    (cst, var) = (0,1) if mode == "vertical"  else (1,0) # The specified axis is the constant one
    #Delete to closed vertical lines
    threshold = 90 if mode == "vertical" else 50
    clean_lines_list = []
    agglomerate_lines = []
    for i, line in enumerate(lines_list):
        if not i in agglomerate_lines:
            new_line = line
            for j in range(i+1, len(lines_list)):
                if not j in agglomerate_lines :
                    test_line = lines_list[j]
                    condition_1 = abs((new_line[0][cst]+new_line[1][cst])/2 - (test_line[0][cst]+test_line[1][cst])/2) < threshold # Close enough
                    m, M = min(new_line[0][var], new_line[1][var]), max(new_line[0][var], new_line[1][var])
                    condition_2 = (m<max(test_line[0][var], test_line[1][var]<M) or m<min(test_line[0][var], test_line[1][var])<M) # Overlap
                    if condition_1 and condition_2:
                        agglomerate_lines.append(j)
                        cst_M = int((min(new_line[0][cst], new_line[1][cst], test_line[0][cst], test_line[1][cst]) + max(new_line[0][cst], new_line[1][cst], test_line[0][cst], test_line[1][cst]))/2)
                        var_1, var_2 = min(new_line[0][var], new_line[1][var], test_line[0][var], test_line[1][var]), max(new_line[0][var], new_line[1][var], test_line[0][var], test_line[1][var])
                        res = [[0,0], [0,0]] # New line may not support tuple assignment
                        res[0][cst], res[1][cst], res[0][var], res[1][var] = cst_M, cst_M, var_1, var_2
                        new_line = res
            clean_lines_list.append(new_line)
    clean_lines_list = sorted(clean_lines_list, key=lambda x: x[0][cst])
    return(clean_lines_list)
    
def get_preprocessed_image(image):
    """The main function to process an image from head to tail

    Args:
        image (np.array): A single image

    Returns:
        _type_: The preprocessed image
    """
    bin_image = binarized_image(image)
    cropped_image = crop_and_rotate(bin_image)
    return cropped_image

if __name__ == "__main__":

    print("start")
    path = r"C:\Users\CF6P\Desktop\cv_text\Data\scan5.pdf"
    images = PDF_to_images(path)
    images = images[0:]
    for im in images:
        processed_image = binarized_image(im)
        get_rectangle(processed_image)