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

def _split_table_landscape(processed_image, rectangles):
    
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
        # cv2.drawContours(im, [box], 0, (0,0,255), 8)
        # plt.imshow(im, cmap="gray")
        # plt.show()
        
        return (xy, wh, rot)
    
    y,x = processed_image.shape
    if len(rectangles)>2: # Clean if there is a black border of the scan wich is concider as a contour
        rectangles = [rect for rect in rectangles if not (0<x-rect[1][0]<10 or 0<y-rect[1][0]<10 or 0<x-rect[1][1]<0 or 0<y-rect[1][1]<10)]
        
    sorted_x, sorted_y = sorted(rectangles, key=lambda x: x[0][0]), sorted(rectangles, key=lambda x: x[0][1])
    Dx, Dy = (sorted_x[-1][0][0]-sorted_x[0][0][0]), (sorted_y[-1][0][1]-sorted_y[0][0][1])
    if Dx > Dy:
        format = "landscape"
    else:
        format = "table"
        
    rectangle = _process_rectangles(rectangles, format)

    return format, rectangle
    
def get_rectangle(processed_image, kernel_size=(3,3)):
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
    rectangles = [list(rect) for rect in rectangles if rect[1][0]*rect[1][1]>x*y/12]

    if len(rectangles)==1:
        return "hand_or_check", rectangles[0]
    else:
        return _split_table_landscape(processed_image, rectangles)

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
        # Delete lines in bias
        if abs(line[0][cst]-line[1][cst])<10:
            lines_list.append(line)
    lines_list = sorted(lines_list, key=lambda x: x[0][cst])

    return lines_list
    
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