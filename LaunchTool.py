import matplotlib.pyplot as plt
import json
import time
import os
import dicttoxml
import numpy as np
from tqdm import tqdm
from datetime import datetime
import cv2

OCR_HELPER_JSON_PATH  = r"CONFIG\OCR_config.json"
OCR_HELPER = json.load(open(OCR_HELPER_JSON_PATH))

whitelist =  "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz(),:.-/°&=àéçëôùê''"
LANG = 'eng+eng2'
TESSCONFIG = [1, 6, whitelist, LANG]

from ProcessPDF import PDF_to_images, binarized_image, get_rectangles, get_format_and_adjusted_image
from TextExtraction import textExtraction
from LandscapeFormat import ProcessLandscape

def getAllImages(path):
    def _get_all_images(path, extList=[".tiff", ".tif", ".png"]):
        docs = os.listdir(path)
        pdf_in_folder = [file for file in docs if os.path.splitext(file)[1].lower() == ".pdf"]
        image_in_folder = [file for file in docs if os.path.splitext(file)[1].lower() in extList]
        return pdf_in_folder, image_in_folder
    
    pdf_in_folder, image_in_folder =  _get_all_images(path) # Return pathes
    res_path = os.path.join(path, "RES")
    if not os.path.exists(res_path):
        os.makedirs(res_path)
    images = []
    images_names = []

    for pdf in pdf_in_folder:
        new_images = PDF_to_images(os.path.join(path, pdf))
        images_names += [os.path.splitext(pdf)[0]+ f"_{i}" for i in range(1,len(new_images)+1)]
        images += new_images
    
    for image in image_in_folder:
        new_images = np.array(cv2.imread(os.path.join(path,image)))
        images_names += [os.path.splitext(image)[0]]
        images.append(new_images)
    return images, images_names

def saveCVTool(res_path, name, cropped_image, OCR_and_text_full_dict):
    save_im_path = os.path.join(res_path, name + ".jpg")
    save_xml_path = os.path.join(res_path, name + ".xml")
    xml = dicttoxml.dicttoxml(OCR_and_text_full_dict)

    with open(save_xml_path, 'w', encoding='utf8') as result_file:
        result_file.write(xml.decode())
    plt.imsave(save_im_path, cropped_image)
    
def TextCVTool(path, def_format="", config = ["paddle", "structure", "en"]):
    """The final tool to use with the GUI

    Args:
        path (path): a folder path
        custom_config (list, optional): tesseract config [oem, psm, whitelist, datatrain]
    """
    
    images, images_names = getAllImages(path)
    res_image, res_image_name = [], []
    res_dict_per_image = {}
    res_dict_per_image["CONFIG"] = config
    res_dict_per_image["RESPONSE"] = {}

    for i in tqdm(range(len(images))):
        print(" Start at : ", datetime.now().strftime("%H:%M:%S"))
        image = images[i]
        name = images_names[i]
        bin_image = binarized_image(image)
        rectangles = get_rectangles(bin_image)
        format, cropped_image = get_format_and_adjusted_image(bin_image, rectangles, image, input_format=def_format)

        # print(format)
        # plt.imshow(cropped_image)
        # plt.show()

        if format == "landscape":
            landscape_dict_res = ProcessLandscape(cropped_image)
            for dict_name, landscape_dict in landscape_dict_res.items():
                res_dict_per_image["RESPONSE"][name+"_"+dict_name] = landscape_dict
                res_image.append(image)
                res_image_name.append(name+"_"+dict_name)
        
        else:
            print(f"Le format de la fiche {name} est :", format)
            res_image.append(image)
            res_image_name.append(name)
            zone_matches = textExtraction(format, cropped_image, JSON_HELPER=OCR_HELPER)

            res_dict_per_image["RESPONSE"][name] = zone_matches

    return res_image_name, res_dict_per_image, res_image

if __name__ == "__main__":
    
    print("start")
    path = r"C:\Users\CF6P\Desktop\ELPV\Data\debug"
    TextCVTool(path)