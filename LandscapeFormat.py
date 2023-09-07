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

def ProcessLandscape(cropped_image, image_name, ocr_config):
    landscape_dict_res = {}
    return landscape_dict_res