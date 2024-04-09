import matplotlib.pyplot as plt
import json
import time
import os
import dicttoxml
import numpy as np
import cv2

OCR_HELPER_JSON_PATH  = r"CONFIG\OCR_config.json"
OCR_HELPER = json.load(open(OCR_HELPER_JSON_PATH))

whitelist =  "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz(),:.-/°&=àéçëôùê''"
LANG = 'eng+eng2'
TESSCONFIG = [1, 6, whitelist, LANG]

from ProcessPDF import PDF_to_images

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

    scan_dict = {}

    for pdf in pdf_in_folder:
        new_images = PDF_to_images(os.path.join(path, pdf))
        pdf_dict = {}
        for i, image in enumerate(new_images):
            pdf_dict[f"image_{i}"] = image
        scan_dict[os.path.splitext(pdf)[0]] = pdf_dict
    
    for image in image_in_folder:
        new_image = np.array(cv2.imread(os.path.join(path,image)))
        scan_dict[os.path.splitext(new_images)[0]] = {"image_0" : new_image}

    return scan_dict

def saveCVTool(res_path, name, cropped_image, OCR_and_text_full_dict):
    save_im_path = os.path.join(res_path, name + ".jpg")
    save_xml_path = os.path.join(res_path, name + ".xml")
    xml = dicttoxml.dicttoxml(OCR_and_text_full_dict)

    with open(save_xml_path, 'w', encoding='utf8') as result_file:
        result_file.write(xml.decode())
    plt.imsave(save_im_path, cropped_image)
    
def TextCVTool(path, model, config = ["paddle", "structure", "en"]):
    """The final tool to use with the GUI

    Args:
        path (path): a folder path
        custom_config (list, optional): tesseract config [oem, psm, whitelist, datatrain]
    """
    

    pdfs_res_dict = {
        "PARAMETRE" : {
            "model" : model,
            "ocr" : config
        }
    }

    scan_dict = getAllImages(path)

    if model == "SEMAE":
        from Model_SEMAE import main
        
    if model == "Fredon":
        from Model_Fredon import main

    pdfs_res_dict["RESPONSE"] = main(scan_dict)

    return scan_dict, pdfs_res_dict

if __name__ == "__main__":
    
    print("start")
    path = r"C:\Users\CF6P\Desktop\ELPV\Data\test2"
    TextCVTool(path, model="Fredon")