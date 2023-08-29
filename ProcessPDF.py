import numpy as np
import matplotlib.pyplot as plt
import cv2
import pdf2image
import json
from copy import copy

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
    blur = cv2.bilateralFilter(image,5,200,200)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    # niblack = cv2.ximgproc.niBlackThreshold(gray, 255, cv2.THRESH_BINARY, 41, -0.1, binarizationMethod=cv2.ximgproc.BINARIZATION_NICK)
    return thresh

def get_rectangle(processed_image, kernel_size=(3,3), interations = 2):
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
    if interations == 2:
        format = "other"
    else:
        format = "table"
    im = processed_image.copy()
    plt.imshow(im, cmap="gray")
    plt.show()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    dilate = cv2.dilate(~processed_image, kernel, iterations=interations)
    plt.imshow(dilate, cmap="gray")
    plt.show()
    contours,_ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        print("No contour found : Crop is impossible, processed image is return.")
        return []
    
    y, x = processed_image.shape[:2]
    rectangles = [cv2.minAreaRect(contour) for contour in contours]
    rectangles = [rect for rect in rectangles if rect[1][0]*rect[1][1]>x*y/12]
    for rec in rectangles:
        box = cv2.boxPoints(rec)
        box = np.int0(box)
        cv2.drawContours(im, [box], 0, (0,0,255), 8)
    plt.imshow(im, cmap="gray")
    plt.show()
    if len(rectangles)==1:
        return format, rectangles[0]
    if len(rectangles)>1:
        return get_rectangle(processed_image, kernel_size=(5,5), interations = interations+2)

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
    plt.imshow(cropped_image, cmap="gray")
    plt.show()
    return cropped_image

def delete_lines(bin_image): # Unused function wich delete (approximatly) lines on an image
    imsave = bin_image.copy()
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50,1))
    remove_horizontal = cv2.morphologyEx(imsave, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    cnts = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(bin_image, [c], -1, (255,255,255), 2)
    return bin_image

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
