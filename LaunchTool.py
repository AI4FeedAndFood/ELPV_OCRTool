import matplotlib.pyplot as plt
import json
import os
import dicttoxml
import numpy as np
import cv2
import win32com.client

OCR_HELPER_JSON_PATH  = r"CONFIG\OCR_config.json"
OCR_HELPER = json.load(open(OCR_HELPER_JSON_PATH))

whitelist =  "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz(),:.-/°&=àéçëôùê''"
LANG = 'eng+eng2'
TESSCONFIG = [1, 6, whitelist, LANG]

from ProcessPDF import PDF_to_images

def extractFromMailBox(savePath, SenderEmailAddress, mailbox_name="", n_message_stop=10):
    """
    Extracts and saves attachments from emails in a specified Outlook mailbox folder sent by a specific sender.
    
    Parameters:
    - savePath (str): The directory path where attachments will be saved.
    - SenderEmailAddress (str): The email address of the sender whose emails should be processed.
    - mailbox_name (str, optional): The name of the specific mailbox to connect to. If not provided, uses the default mailbox.
    - n_message_stop (int): The number of messages to process before stopping.
    """
    
    # Create the save directory if it doesn't exist
    os.makedirs(savePath, exist_ok=True)
    
    # Connect to Outlook
    outlook = win32com.client.Dispatch("Outlook.Application").GetNamespace("MAPI")
    
    # Access the specified mailbox or the default mailbox
    if mailbox_name:
        recipient = outlook.Folders[mailbox_name]
        inbox = recipient.Folders["Boîte de réception"]
    else:
        inbox = outlook.GetDefaultFolder(6)  # 6 refers to the default Inbox folder
    
    # Retrieve all messages in the inbox
    messages = inbox.Items
    messages.Sort("ReceivedTime", True)  # Sort messages by received time in descending order
    
    n_message = 0
    while n_message <= n_message_stop:
        try:
            message = messages[n_message]
            n_message += 1
            
            # Check if the message is from the specified sender and is unread
            if message.SenderEmailAddress == SenderEmailAddress and message.Unread:
                # Iterate through attachments in the message
                for attachment in message.Attachments:
                    # Save each attachment to the specified directory
                    attachment.SaveAsFile(os.path.join(savePath, str(attachment.FileName)))
                    # Mark the message as read after saving the attachment
                    message.Unread = False
                    # Process only the first attachment for this message
                    break
        except Exception as e:
            # Print the error message for debugging purposes
            print(f"An error occurred: {e}")
            continue


def getAllImages(path):
    def _get_all_images(path, extList=[".tiff", ".tif", ".png", ".jpg"]):
        docs = os.listdir(path)
        pdf_in_folder = [file for file in docs if os.path.splitext(file)[1].lower() == ".pdf"]
        image_in_folder = [file for file in docs if os.path.splitext(file)[1].lower() in extList]
        pdf_in_folder.reverse(), image_in_folder.reverse()
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
        scan_dict[pdf] = pdf_dict
    
    for image in image_in_folder:
        new_image = np.array(cv2.imread(os.path.join(path,image)))
        scan_dict[image] = {"image_0" : new_image}

    return scan_dict

def saveCVTool(res_path, name, cropped_image, OCR_and_text_full_dict):
    save_im_path = os.path.join(res_path, name + ".jpg")
    save_xml_path = os.path.join(res_path, name + ".xml")
    xml = dicttoxml.dicttoxml(OCR_and_text_full_dict)

    with open(save_xml_path, 'w', encoding='utf8') as result_file:
        result_file.write(xml.decode())
    plt.imsave(save_im_path, cropped_image)
    
def TextCVTool(path, model, config = ["paddle", "structure", "en"], email_sender=""):
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

    if email_sender:
        try:
            extractFromMailBox(path, SenderEmailAddress=email_sender, n_message_stop=10)
            print("PDF are loaded from mails")
        except:
            print("Error : PDF from mails ARE NOT downloaded, please do it manually")


    scan_dict = getAllImages(path)

    if model == "SEMAE":
        from Model_SEMAE import main
        
    if model == "Fredon":
        from Model_Fredon import main

    pdfs_res_dict["RESPONSE"] = main(scan_dict)

    return scan_dict, pdfs_res_dict

if __name__ == "__main__":
    
    # print("start")
    # path = r"C:\Users\CF6P\Desktop\ELPV\Data\test2"
    # TextCVTool(path, model="Fredon")

    p = r"C:\Users\CF6P\Desktop\test"
    extractFromMailBox(p, SenderEmailAddress="Pierre.Rouyer@ftfr.eurofins.com")