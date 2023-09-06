import PySimpleGUI as sg
import os
import shutil
import io
import dicttoxml
import json
import pandas as pd

from screeninfo import get_monitors
from PIL import Image

from LaunchingTool import getAllPDFs, TextCVTool, getAllImages

def is_valid_path(folderpath):
    if folderpath and os.path.exists(folderpath):
        return True
    sg.popup_error("Le dossier n'existe pas")
    return False

def fit_the_screen(X_loc):
    ordered_monitors = sorted([monitor for monitor in get_monitors()], key=lambda mtr: mtr.x)
    for monitor in ordered_monitors:
        x, width, height = monitor.x, monitor.width, monitor.height
        if X_loc<x+width:
            return int(0.95*width), int(0.95*height)

def use_the_tool(folderPath):
    images_names, res_dict_per_image, images = TextCVTool(folderPath)
    return images_names, res_dict_per_image, images

def checkboxs_for_parasite(parasite_list, found_parasites):
    found_parasites = [para[0] for para in found_parasites]
    not_found = [para for para in parasite_list if para not in found_parasites]
    found_col = []
    not_found_col = []
    for i, fpara in enumerate(found_parasites,1):
        found_col.append([sg.Checkbox(str(fpara), default=True, key=f"para_{str(fpara)}")])
    for i, fpara in enumerate(not_found, len(found_parasites)+1):
        not_found_col.append([sg.Checkbox(str(fpara), key=f"para_{str(fpara)}")])
    return found_col, not_found_col

def _getFieldsLayout(image_dict, X_dim, Y_dim):
    conversion_dict = LIMSsettings["CLEAN_ZONE"]
    conf_threshold = int(GUIsettings["TOOL"]["confidence_threshold"])    
    lineLayout = []
        
    for zone, landmark_text_dict in image_dict.items():
        if not zone in ["client_name", "contract_name"]:
            clean_zone_name = conversion_dict[zone]
            text = f"{clean_zone_name} : "
            sequence = landmark_text_dict["sequences"]
            conf= 100 # If empty must not be red
            if landmark_text_dict["indexes"] != []:
                conf = min([landmark_text_dict["OCR"]["conf"][i] for i in landmark_text_dict["indexes"]])           
            if conf < conf_threshold : 
                back_color = 'red'
            else : 
                back_color = sg.theme_input_background_color()
                
            if zone == "parasite_recherche":
                n_list = len(sequence)
                seq = ""
                found_col, not_found_col = checkboxs_for_parasite(PARASITES_LIST, sequence)
                text = f"{clean_zone_name} : {n_list} proposé.s"
                
                lineLayout.append([sg.Text(text, s=(25,1)), sg.Column(found_col, size=(int(X_dim*0.15), int(Y_dim*0.035*len(found_col))), scrollable=True)])
                # lineLayout.append([sg.HorizontalSeparator(key='sep')])
                lineLayout.append([sg.Text("Ajouter si besoin", s=(25,1)), sg.Column(not_found_col, scrollable=True, size=(int(X_dim*0.15),
                                                                                                                            int(Y_dim*0.035*len(not_found_col))))])
            
            elif len(sequence)>=INPUT_LENGTH:
                lineLayout.append([sg.Text(text, s=(25,1)),
                                sg.Multiline(sequence, background_color=back_color,
                                key=f"-{zone}-", expand_y=True, expand_x=False, size=(INPUT_LENGTH, (len(sequence)//INPUT_LENGTH)+1), justification='center')])
            else :
                lineLayout.append([sg.Text(text, s=(25,1)),
                                sg.I(sequence, background_color=back_color,
                                key=f"-{zone}-", expand_y=True, expand_x=False, size=(INPUT_LENGTH, 1), justification='center')])
                                # sg.Image(data=bio.getvalue(), key=f'image_{zone}')])
        
    return lineLayout
    
def _getImageLayout(image):
    
    searching_area = Image.fromarray(image).resize((int(image.shape[1]*0.58), int(image.shape[0]*0.58)))
    bio = io.BytesIO()
    searching_area.save(bio, format="PNG")
    imageLayout = [[sg.Image(data=bio.getvalue(), key=f'scan')]]
    
    return imageLayout

def _getClientContractLayout(image_dict):
    
    ClientContractLayout = []
    
    client, contract = "", ""
    if "client_name" in list(image_dict.keys()):
        client = image_dict["client_name"]["sequences"]
    if "contract_name" in list(image_dict.keys()):
        contract = image_dict["contract_name"]["sequences"]
        
    ClientContractLayout.append([sg.Text('N° de client', size=(25, 1)) ,sg.Input(client, size=(INPUT_LENGTH, 1), enable_events=True, key='-client_name-')])
    ClientContractLayout.append([sg.Text('N° de contrat', size=(25, 1)) ,sg.Input(contract, size=(INPUT_LENGTH, 1), enable_events=True, key='-contract_name-', background_color='light gray')])
    ClientContractLayout.append([sg.HorizontalSeparator(key='sep')])
    
    return ClientContractLayout

def ClientSuggestionWindow(values, mainWindow):
    x = mainWindow['-client_name-'].Widget.winfo_rootx()
    y = mainWindow['-client_name-'].Widget.winfo_rooty() + 25
    if not values:
        return
    layout = [
        [sg.Listbox(values, size=(50, 5), enable_events=True, key='-AUTO_CLIENT-',
            select_mode=sg.LISTBOX_SELECT_MODE_SINGLE)]]
    return sg.Window('Title', layout, no_titlebar=True, keep_on_top=True,
        location=(x, y), margins=(0, 0), finalize=True)

def ContractSuggestionWindow(values, mainWindow):
    x = mainWindow['-contract_name-'].Widget.winfo_rootx()
    y = mainWindow['-contract_name-'].Widget.winfo_rooty() + 25
    if not values:
        return
    layout = [
        [sg.Listbox(values, size=(50, 5), enable_events=True, key='-AUTO_CONTRACT-',
            select_mode=sg.LISTBOX_SELECT_MODE_SINGLE)]]
    return sg.Window('Title', layout, no_titlebar=True, keep_on_top=True,
        location=(x, y), margins=(0, 0), finalize=True)

def getMainLayout(image_dict, image, X_dim, Y_dim):
    FiledsLayout, ImageLayout, ClientContractLayout = _getFieldsLayout(image_dict, X_dim, Y_dim), _getImageLayout(image), _getClientContractLayout(image_dict)
    
    MainLayout = [
        [sg.Text("Attention : Veuillez bien vérifier les champs proposés"), sg.Push(), sg.Button("Consignes d'utilisation", k="-consignes-")],
        [sg.Push(), sg.B("<- Retour", s=10), sg.B("Valider ->", s=10), sg.Push()]
    ]
    
    MainLayout.append([sg.Column(ClientContractLayout + FiledsLayout , justification="r"), 
            sg.Column(ImageLayout, scrollable=True, justification="l", size=(int(X_dim*0.9), int(0.9*Y_dim)))])
    return MainLayout

def choice_overwrite_continue_popup(general_text, red_text, green_text):
    layout = [
        [sg.Text(general_text)],
        [sg.B(red_text, k="overwrite", button_color="tomato"), sg.B(green_text, k="continue", button_color="medium sea green")] 

    ]
    window = sg.Window("Predict Risk", layout, use_default_focus=False, finalize=True, modal=True)
    event, values = window.read()
    window.close()
    return event

def convertDictToLIMS(verified_dict, scan_name, CLIENT_CONTRACT_DF): 
    
    def _get_parasite_code(para_list):
        unit_code_list = []
        test_code_list = []
        for u_code, u_para in LIMSsettings["UNIT_TEST_CODE"].items():
            if u_para in para_list:
                unit_code_list.append(u_code)
        unit_in_package = []
        for u_code, u_para in LIMSsettings["PACKAGE_TEST_CODE"].items(): 
           # Unique case of a package that include two others            
            if u_code in ["PIP0W", "PIPIG"] and "PIPT3" in test_code_list: pass
            
            if set(unit_code_list).intersection(set(u_para)) == set(u_para):
                test_code_list.append(u_code)
                unit_in_package += u_para
        test_code_list += list(set(unit_in_package) ^ set(unit_code_list)) # Add non-packed unit test
        
        return test_code_list

    # Clean fields
    scan_converted_dict = {
                           }
    scan_clean_dict = {}
    para = []
    for key, value  in verified_dict.items():
        if "para_" in key and value == True:
            para.append(key.split("para_")[-1])
        if key[0] == "-":
            scan_clean_dict[key] = value
    scan_clean_dict["-parasite_recherche-"] = para
    scan_clean_dict["-parasite_package-"] =_get_parasite_code(para)
        
    # Convert client and contract for the LIMS
    
    clientName, contractName = verified_dict["-client_name-"], verified_dict["-contract_name-"]
    corresponding_row = CLIENT_CONTRACT_DF[(CLIENT_CONTRACT_DF["clientname"]==clientName) & (CLIENT_CONTRACT_DF["contractname"]==contractName)]
    if len(corresponding_row) == 1:
        scan_converted_dict["CustomerCode"] = list(corresponding_row["clientCode"])[0] #Found way to avoid an object return
        scan_converted_dict["ContractCode"] = list(corresponding_row["contractcode"])[0]
        scan_converted_dict["Devis"] = list(corresponding_row["Devis"])[0]
        scan_converted_dict["EngamentJuridique"] = list(corresponding_row["EngamentJuridique"])[0]
    else :
        scan_converted_dict["CustomerCode"] = ""
        scan_converted_dict["ContractCode"] = ""
        scan_converted_dict["Devis"] = ""
        scan_converted_dict["EngamentJuridique"] = ""
    
    
    scan_converted_dict.update({"Sample" :  {}})
    additionalFiel_dict = {}
    for key, code in LIMSsettings["LIMS_CONVERTER"].items():
        if f"-{key}-" in list(scan_clean_dict.keys()):
            if key == "N_d_echantillon":
                scan_converted_dict["Sample"][code] = scan_clean_dict[f"-{key}-"]
            else :
                additionalFiel_dict[code] = scan_clean_dict[f"-{key}-"]
    scan_converted_dict["Sample"]["AdditionalField"] = additionalFiel_dict
    return scan_converted_dict

def runningSave(save_path_json, verified_imageDict, image_name, res_dict):
    for key, items in verified_imageDict.items() :
        if key[0] == "-":
            res_dict["RESPONSE"][image_name][key.strip("-")]["sequences"] = items
    with open(save_path_json, 'w', encoding='utf-8') as f:
        json.dump(res_dict, f,  ensure_ascii=False)
        
def finalSaveDict(verified_dict, CLIENT_CONTRACT_DF, xml_save_path, res_path, out_path="", xml_name="verified_XML"):
    for scan_name, scan_dict in verified_dict.items(): 
        clean_dict = convertDictToLIMS(scan_dict, scan_name, CLIENT_CONTRACT_DF)
        xml = dicttoxml.dicttoxml(clean_dict)
        with open(os.path.join(xml_save_path, f"{scan_name}.xml"), 'w', encoding='utf8') as result_file:
            result_file.write(xml.decode())
    
    if out_path:
        new_xml =  os.path.join(out_path, xml_name)
        i_test = 1
        while os.path.exists(new_xml):
            new_xml = os.path.join(out_path, xml_name+ f"_{i_test}")
            i_test+=1
        os.rename(xml_save_path, new_xml)
        # shutil.rmtree(res_path)
    
def main():
    welcomeLayout = [
        [sg.Text("Dossier contenant les PDFs"), sg.Input(LIMSsettings["TOOL_PATH"]["input_folder"], key="-PATH-"), 
         sg.FolderBrowse(button_color="cornflower blue")],
        [sg.Push(), sg.Exit(button_color="tomato"), sg.Push(), sg.Button("Lancer l'algorithme", button_color="medium sea green"), sg.Push()]
    ]
    window_title = GUIsettings["GUI"]["title"]
    welcomWindow = sg.Window(window_title, welcomeLayout, use_custom_titlebar=True)
    ClientSuggestionW, contractSuggestionW = None, None
    while True:
        event, values = welcomWindow.read()
        if event in (sg.WINDOW_CLOSED, "Exit"):
            break
        if event == "Lancer l'algorithme":
            givenPath = values["-PATH-"]
            if is_valid_path(givenPath):
                pdfs = getAllPDFs(givenPath)
                if pdfs == []:
                    sg.popup_ok("Aucun PDF n'est trouvé dans le dossier")
                else :
                    res_path = os.path.join(givenPath, "RES")
                    save_path_json = os.path.join(res_path, "res.json")
                    xml_res_path = os.path.join(res_path, "verified_XML")
                    start = False # Set to true when scans are processed ; condition to start the verification process
                    end = False # Force the verification process to be done or abandonned (no infinit loop)
                    if os.path.exists(save_path_json):
                        continue_or_smash = choice_overwrite_continue_popup("Il semblerait que l'analyse ai déjà été effectuée.\nEcraser la précédente analyse ?"
                                                                   , "Ecraser", "Reprendre la précédente analyse") #Change conditon
                    else : continue_or_smash="overwrite"
                    if continue_or_smash==None : pass
                    if continue_or_smash == "continue":
                        json_file  = open(save_path_json, encoding='utf-8')
                        res_dict_per_image = json.load(json_file)
                        images_names_dict = list(res_dict_per_image["RESPONSE"].keys())
                        images, images_names = getAllImages(pdfs, givenPath)
                        
                        welcomWindow.close()
                        start = True
                        if images_names_dict != images_names:
                            sg.popup_ok(f"Les images {str(list(set(images_names_dict+images_names)))} ne se trouvent pas dans l'analyse précédente.")
                    if continue_or_smash == "overwrite":
                        sg.popup_auto_close("L'agorithme va démarrer !\nSuivez l'évolution dans le terminal", auto_close_duration=5)
                        welcomWindow.close()
                        print("------ START -----")
                        print("Attendez la barre de chargement \nAppuyez sur ctrl+c dans le terminal pour interrompre")
                        images_names, res_dict_per_image, images = use_the_tool(givenPath)
                        print("------ DONE -----")
                        with open(save_path_json, 'w', encoding='utf-8') as json_file:
                            json.dump(res_dict_per_image, json_file,  ensure_ascii=False) # Save the extraction json on RES
                        if os.path.exists(xml_res_path): # Create or overwrite the verified_XML folder in RES
                            shutil.rmtree(xml_res_path)
                        start = True
                    if not os.path.exists(xml_res_path):
                        os.makedirs(xml_res_path) 
                    while start and not end:
                        end = True
                        if images_names == []:
                            sg.popup_ok("Aucune image n'a été trouvé dans les PDFs")
                        else :
                            verified_dict = {}
                            n_image = 0
                            n_displayed= -1
                            X_dim, Y_dim = fit_the_screen(1)
                            X_loc, Y_loc = (10,10)
                            start = False
                            CONTRACT_LIST_0 = list(CLIENT_CONTRACT_DF["contractname"].unique())
                            CONTRACT_LIST = CONTRACT_LIST_0
                            while n_image < len(images_names):
                                if n_image != n_displayed:
                                    client_warning=None
                                    if start == True :
                                        X_loc, Y_loc = VerificationWindow.current_location()
                                        X_dim, Y_dim = fit_the_screen(10)
                                    image_name = images_names[n_image]
                                    image = images[n_image]
                                    image_dict = res_dict_per_image["RESPONSE"][image_name]
                                    VerificactionLayout = getMainLayout(image_dict, image, X_dim, Y_dim)
                                    if start == True :
                                        VerificationWindow.close()
                                    VerificationWindow = sg.Window(f"Fiche {image_name} - ({n_image+1}/{len(images_names)})", 
                                                                VerificactionLayout, use_custom_titlebar=True, location=(X_loc, Y_loc), 
                                                                size=(X_dim, Y_dim), resizable=True, finalize=True)
                                    n_displayed = n_image
                                start = True
                                verif_window, verif_event, verif_values = sg.read_all_windows()
                                if verif_event == sg.WINDOW_CLOSED:
                                    break
                                if verif_event == "-consignes-":
                                    sg.popup_scrolled(GUIsettings["UTILISATION"]["texte"], title="Consignes d'utilisation", size=(50,10))                              
                                if verif_event == "-client_name-":
                                    client_text = verif_values["-client_name-"]
                                    client_sugg = sorted([client for client in CLIENT_LIST if client_text.lower() in client.lower()])
                                    if client_text and client_sugg:
                                        if ClientSuggestionW :
                                            ClientSuggestionW["-AUTO_CLIENT-"].update(values=client_sugg)
                                            ClientSuggestionW.refresh()
                                        elif len(client_text)>2 :
                                            ClientSuggestionW = ClientSuggestionWindow(client_sugg, VerificationWindow)                
                                                                                     
                                if verif_event == '-AUTO_CLIENT-':
                                    ClientSuggestionW.close()
                                    ClientSuggestionW = False
                                    text = verif_values['-AUTO_CLIENT-'][0]
                                    VerificationWindow['-client_name-'].update(value=text)
                                    
                                    CONTRACT_LIST = sorted(list(CLIENT_CONTRACT_DF[CLIENT_CONTRACT_DF["clientname"]== text]["contractname"].unique()))
                                    VerificationWindow['-contract_name-'].update(disabled=False, background_color=sg.theme_input_background_color())
                                    VerificationWindow['-client_name-'].set_focus()
                                    
                                if verif_event == "-contract_name-":
                                    contract_text = verif_values["-contract_name-"]
                                    contract_sugg = [contract for contract in CONTRACT_LIST if contract_text.lower() in contract.lower()]
                                    if contract_text and contract_sugg:
                                        if contractSuggestionW :
                                            contractSuggestionW["-AUTO_CONTRACT-"].update(values=contract_sugg)
                                            contractSuggestionW.refresh()
                                        else :
                                            contractSuggestionW = ContractSuggestionWindow(contract_sugg, VerificationWindow)                                                                                   
                                if verif_event == '-AUTO_CONTRACT-':
                                    contractSuggestionW.close()
                                    contractSuggestionW = False
                                    text = verif_values['-AUTO_CONTRACT-'][0]
                                    VerificationWindow['-contract_name-'].update(value=text)
                                    VerificationWindow['-contract_name-'].set_focus()    
                                
                                if verif_event == "Valider ->":
                                    if contractSuggestionW : contractSuggestionW.close()
                                    if ClientSuggestionW : ClientSuggestionW.close()
                                    if (not verif_values["-client_name-"] in CLIENT_LIST or not verif_values["-contract_name-"] in CONTRACT_LIST_0) and not client_warning:
                                            sg.popup_ok("ATTENTION : VEUILLEZ REMPLIR UN CODE CIENT ET UN CODE CONTRAT VALIDE", button_color="dark green", 
                                                        location = (X_loc+200, Y_loc+200))
                                            client_warning=True
                                    # If last image                                              
                                    elif n_image == len(images_names)-1:
                                        verified_dict[image_name] = verif_values
                                        runningSave(save_path_json, verif_values, image_name, res_dict_per_image)
                                        choice = sg.popup_ok("Il n'y a pas d'image suivante. Finir l'analyse ?", button_color="dark green")
                                        if choice == "OK":
                                            json_file.close() # Close the file
                                            finalSaveDict(verified_dict, CLIENT_CONTRACT_DF, xml_res_path, res_path, LIMSsettings["TOOL_PATH"]["output_folder"])
                                            VerificationWindow.close()
                                            break
                                    else: 
                                        # Register the response and go to the following
                                        verified_dict[image_name] = verif_values
                                        runningSave(save_path_json, verif_values, image_name, res_dict_per_image)
                                        n_image+=1
                                if verif_event == "<- Retour":
                                    runningSave(save_path_json, verif_values, image_name, res_dict_per_image)
                                    n_image-=1
                                    if contractSuggestionW : contractSuggestionW.close()
                                    if ClientSuggestionW : ClientSuggestionW.close()
                            VerificationWindow.close()                 
    
    welcomWindow.close()               
                         
if __name__ == "__main__":
    print("Attendez quelques instants, une page va s'ouvrir")
    
    SETTINGS_PATH = os.getcwd()
    GUIsettings = sg.UserSettings(path=os.path.join(SETTINGS_PATH, "CONFIG"), filename="GUI_config.ini", use_config_file=True, convert_bools_and_none=True)
    theme = GUIsettings["GUI"]["theme"]
    font_family = GUIsettings["GUI"]["font_family"]
    font_size = int(GUIsettings["GUI"]["font_size"])
    help_text = GUIsettings["UTILISATION"]["text"]    
    sg.theme(theme)
    sg.set_options(font=(font_family, font_size))
    
    # sys.path.append(r"CONFIG")
    OCR_HELPER = json.load(open(r"CONFIG\OCR_config.json"))
    LIMSsettings = json.load(open(r"CONFIG\LIMS_config.json"))
    INPUT_LENGTH = 45

    PARASITES_LIST = OCR_HELPER["lists"]["parasite"]

    LIMSCodeXlsxPath = os.path.join(r"CONFIG\client_contract.xlsx")
    CLIENT_CONTRACT_DF = pd.read_excel(LIMSCodeXlsxPath, dtype=str).fillna("")
    clientName, contractName = CLIENT_CONTRACT_DF.iloc[0]["clientname"], CLIENT_CONTRACT_DF.iloc[0]["contractname"]
    row = CLIENT_CONTRACT_DF[(CLIENT_CONTRACT_DF["clientname"]==clientName) & (CLIENT_CONTRACT_DF["contractname"]==contractName)]
    CLIENT_LIST = list(CLIENT_CONTRACT_DF["clientname"].unique())
    main()
