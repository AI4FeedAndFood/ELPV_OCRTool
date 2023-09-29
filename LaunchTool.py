import matplotlib.pyplot as plt
import json
import time
import os
import dicttoxml
import numpy as np
from tqdm import tqdm
from datetime import datetime

OCR_HELPER_JSON_PATH  = r"CONFIG\OCR_config.json"
OCR_HELPER = json.load(open(OCR_HELPER_JSON_PATH))

whitelist =  "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz(),:.-/°&=àéçëôùê''"
LANG = 'eng+eng2'
TESSCONFIG = [1, 6, whitelist, LANG]

from ProcessCheckboxes import crop_image_and_sort_format
from ProcessPDF import PDF_to_images, binarized_image
from TextExtraction import get_data_and_landmarks, get_wanted_text
from LandscapeFormat import ProcessLandscape


def TextExtractionTool_EVALUATE(path, save_path=r"C:\Users\CF6P\Desktop\cv_text\Evaluate", custom_config=TESSCONFIG):
    start_time = time.time()
    images = PDF_to_images(path)
    res_dict_per_image = {}
    res_dict_per_image["TESSERACT"] = TESSCONFIG
    res_dict_per_image["RESPONSE"] = {}
    
    for i, image in enumerate(images,1):
        print(f"\n ------------ Image {i} is starting. time : {(time.time() - start_time)} -------------------")
        processed_image = binarized_image(image)
        format, cropped_image = crop_image_and_sort_format(processed_image)
        print(f"Image is cropped, format is {format}. time : {(time.time() - start_time)}")
        OCR_data, landmarks_dict = get_data_and_landmarks(format, cropped_image, ocr_config=custom_config)
        print(f"Landmarks are found. time : {(time.time() - start_time)}")
        OCR_and_text_full_dict = get_wanted_text(cropped_image, landmarks_dict, format, ocr_config=custom_config)
        # Backup in case format is badly assigate
        non_empty_field = 0
        for _, value_dict in OCR_and_text_full_dict.items():
            if value_dict["sequences"] != []:
                non_empty_field+=1
                
        if format == "table" and non_empty_field < 2:
            format = "hand"
            OCR_data, landmarks_dict = get_data_and_landmarks(format, cropped_image, ocr_config=custom_config)
            OCR_and_text_full_dict = get_wanted_text(cropped_image, landmarks_dict, format, ocr_config=custom_config)
            
        res_dict_per_image["RESPONSE"][str(f"{i}")] = OCR_and_text_full_dict
        print(f"Job is done for this image. time : {(time.time() - start_time)}")
        save_image_EVALUATE(OCR_and_text_full_dict, cropped_image, path, i, save_path = save_path)
        print(f"Saved. time  : {(time.time() - start_time)}")
    # with open(os.path.join(save_path,"sample.json"), "w") as outfile:
    #     outfile.write(json.dumps(res_dict_per_image))
    return res_dict_per_image

def save_image_EVALUATE(landmark_text_dict, cropped_image, path, i, save_path):
    _, name = os.path.split(path)
    name, _ = os.path.splitext(name)
    save_path = os.path.join(save_path, f"{name}_{i}.jpg")
    
    fig, axs = plt.subplots(6, 2, figsize=(30,30))
    a, b = 0, 0
    for i, (zone, dict) in enumerate(landmark_text_dict.items()):
        text = dict["sequences"]
        y_min, y_max, x_min, x_max = np.array(dict["box"])
        axs[a, b].imshow(cropped_image[y_min:y_max, x_min:x_max])
        if zone == "parasite_recherche":
            t1, t2, t3 = text[:int(len(text)/3)], text[int(len(text)/3):int(2*len(text)/3):], text[int(2*len(text)/3):]
            axs[a, b].set_title(f'{zone} : \n {t1} \n {t2} \n {t3}', size = 30)
        else :
            axs[a, b].set_title(f'{zone} : \n {text}', size = 30)
        a+=1
        if i == 3 : 
            a=0
            b=1
    plt.plot()
    fig.savefig(save_path)
    plt.close()

def getAllPDFs(path):
    pdf_in_folder = [file for file in os.listdir(path) if os.path.splitext(file)[1].lower() == ".pdf"]
    return pdf_in_folder

def getAllImages(pdf_in_folder, path):
    images = []
    images_names = []
    for pdf in pdf_in_folder:
        new_images = PDF_to_images(os.path.join(path, pdf))
        images_names += [os.path.splitext(pdf)[0]+ f"_{i}" for i in range(1,len(new_images)+1)]
        images += new_images
    return images, images_names

def saveCVTool(res_path, name, cropped_image, OCR_and_text_full_dict):
    save_im_path = os.path.join(res_path, name + ".jpg")
    save_xml_path = os.path.join(res_path, name + ".xml")
    xml = dicttoxml.dicttoxml(OCR_and_text_full_dict)

    with open(save_xml_path, 'w', encoding='utf8') as result_file:
        result_file.write(xml.decode())
    plt.imsave(save_im_path, cropped_image)
    
def TextCVTool(path, custom_config=TESSCONFIG, def_format="default"):
    """The final tool to use with the GUI

    Args:
        path (path): a folder path
        custom_config (list, optional): tesseract config [oem, psm, whitelist, datatrain]
    """
    pdf_in_folder =  getAllPDFs(path)
    res_path = os.path.join(path, "RES")
    if not os.path.exists(res_path):
        os.makedirs(res_path)
    
    images, images_names = getAllImages(pdf_in_folder, path)
    res_image, res_image_name = [], []
    res_dict_per_image = {}
    res_dict_per_image["TESSERACT"] = TESSCONFIG
    res_dict_per_image["RESPONSE"] = {}

    for i in tqdm(range(len(images))):
        print(" Start at : ", datetime.now().strftime("%H:%M:%S"))
        image = images[i]
        name = images_names[i]
        processed_image = binarized_image(image)
        format, cropped_image = crop_image_and_sort_format(processed_image, original_image=image, def_format=def_format)
        print(f"Le format de la fiche {name} est :", format)
        if format == "landscape":
            landscape_dict_res = ProcessLandscape(cropped_image)
            for dict_name, landscape_dict in landscape_dict_res.items():
                res_dict_per_image["RESPONSE"][name+"_"+dict_name] = landscape_dict
                res_image.append(image)
                res_image_name.append(name+"_"+dict_name)
        else :
            res_image.append(image)
            res_image_name.append(name)
            OCR_data, landmarks_dict = get_data_and_landmarks(format, cropped_image, ocr_config=custom_config)
            OCR_and_text_full_dict = get_wanted_text(cropped_image, landmarks_dict, format, ocr_config=custom_config)
            
            # Backup in case format is wrongly assigate
            non_empty_field = 0
            for _, value_dict in OCR_and_text_full_dict.items():
                if value_dict["sequences"] != []:
                    non_empty_field+=1
            if format == "table" and non_empty_field < 2: 
                format = "hand"
                OCR_data, landmarks_dict = get_data_and_landmarks(format, cropped_image, ocr_config=custom_config)
                OCR_and_text_full_dict = get_wanted_text(cropped_image, landmarks_dict, format, ocr_config=custom_config)
            res_dict_per_image["RESPONSE"][name] = OCR_and_text_full_dict
            
    return res_image_name, res_dict_per_image, res_image

if __name__ == "__main__":
    
    print("start")
    path = r"C:\Users\CF6P\Desktop\cv_text\Data\landscape"
    TextCVTool(path, custom_config=TESSCONFIG, def_format="landscape")
        