import PySimpleGUI as sg
import os
import shutil
import numpy as np
import io
import dicttoxml
import json
import pandas as pd
from copy import deepcopy

from screeninfo import get_monitors
from PIL import Image

OCR_HELPER_JSON_PATH  = r"CONFIG\OCR_config.json"
OCR_HELPER = json.load(open(OCR_HELPER_JSON_PATH))

from LaunchTool import getAllImages, TextCVTool, getAllImages

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

def use_the_tool(folderPath, def_format=""):
    images_names, res_dict_per_image, images = TextCVTool(folderPath, def_format=def_format)
    return images_names, res_dict_per_image, images

def checkboxs_for_parasite(parasite_list, found_parasites):
    not_found = [para for para in parasite_list if para not in found_parasites]
    found_col = []
    not_found_col = []
    for i, fpara in enumerate(found_parasites,1):
        found_col.append([sg.Checkbox(str(fpara), default=True, key=f"para_{str(fpara)}")])
    for i, fpara in enumerate(not_found, len(found_parasites)+1):
        not_found_col.append([sg.Checkbox(str(fpara), key=f"para_{str(fpara)}")])
    return found_col, not_found_col

def _getFieldsLayout(image_dict, X_dim, Y_dim, landscape=False):
    conversion_dict = LIMSsettings["CLEAN_ZONE"]
    conf_threshold = int(GUIsettings["TOOL"]["confidence_threshold"])/100   
    lineLayout = []
    # Returned the tool's response for each field
    for zone, landmark_text_dict in image_dict.items():
        if not zone in ["client_name", "contract_name", "n_copy", "ref_echantillon"]:
            clean_zone_name = conversion_dict[zone]
            text = f"{clean_zone_name} : "
            sequence = landmark_text_dict["sequence"]
            conf= landmark_text_dict["confidence"] # If empty must not be red
            if conf < conf_threshold or sequence=="":
                back_color = 'red'
            else: 
                back_color = sg.theme_input_background_color()
                
            if zone == "parasite_recherche":
                n_list = len(sequence)
                found_col, not_found_col = checkboxs_for_parasite(PARASITES_LIST, sequence)
                text = f"{clean_zone_name} : {n_list} proposé.s"
                
                lineLayout.append([sg.Text(text, s=(25,1)), sg.Column(found_col, size=(int(X_dim*0.15), int(Y_dim*0.035*len(found_col))), scrollable=True)])
                # lineLayout.append([sg.HorizontalSeparator(key='sep')])
                lineLayout.append([sg.Text("Ajouter si besoin", s=(25,1)), sg.Column(not_found_col, scrollable=True, size=(int(X_dim*0.15),
                                                                                                                            int(Y_dim*0.035*len(not_found_col))))])
            elif zone == "type_lot":
                type_list = list(LIMSsettings["TYPE_LOT"].keys())
                default = "Surveillance"
                for type in type_list:
                    if sequence in LIMSsettings["TYPE_LOT"][type]:
                        default=type
                        break
                lineLayout.append([sg.Text(text, s=(25,1)),
                                sg.Combo(type_list, default_value=default, background_color=back_color,
                                key=f"-{zone}-", expand_y=True, expand_x=False, size=(INPUT_LENGTH, 1))])

            elif len(sequence)>=INPUT_LENGTH:
                lineLayout.append([sg.Text(text, s=(25,1)),
                                sg.Multiline(sequence, background_color=back_color,
                                key=f"-{zone}-", expand_y=True, expand_x=False, size=(INPUT_LENGTH, (len(sequence)//INPUT_LENGTH)+1), justification='left')])
            else :
                lineLayout.append([sg.Text(text, s=(25,1)),
                                sg.I(sequence, background_color=back_color,
                                key=f"-{zone}-", expand_y=True, expand_x=False, size=(INPUT_LENGTH, 1), justification='left')])
                                # sg.Image(data=bio.getvalue(), key=f'image_{zone}')])
    # Combo for Reference Echantillon
    if (image_dict["variete"]["sequence"] == []) or landscape:
        default_ref = "Sol"  
    else: 
        default_ref = "Tubercule"
        
    if "ref_echantillon" in list(image_dict.keys()):
        default_ref = image_dict["ref_echantillon"]["sequence"]
    line = [sg.Text("Référence échantillon : ", s=(25,1)),
            sg.Combo(["Sol", "Tubercule", "Plant de pomme de terre", "Betterave"], default_value=default_ref,
            key=f"-ref_echantillon-", expand_y=True, expand_x=False, size=(INPUT_LENGTH, 1))]
    lineLayout = lineLayout[:-2] + [line] + lineLayout[-2:] # Ref should be above parasites
    
    return lineLayout
    
def _getImageLayout(image):
    
    searching_area = Image.fromarray(image).resize((int(image.shape[1]*0.4), int(image.shape[0]*0.4)))
    bio = io.BytesIO()
    searching_area.save(bio, format="PNG")
    imageLayout = [[sg.Image(data=bio.getvalue(), key=f'scan')]]
    
    return imageLayout

def _getClientContractLayout(image_dict, last_info):
    
    ClientContractLayout = []
    
    client, contract = "", ""
    if "client_name" in list(image_dict.keys()):
        client = image_dict["client_name"]["sequence"]
    if "contract_name" in list(image_dict.keys()):
        contract = image_dict["contract_name"]["sequence"]
    
    if not client:
        client = last_info[0]
    if not contract:
        contract = last_info[1]
        
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

def getMainLayout(image_dict, image, X_dim, Y_dim, last_info, landscape=False, add=False):
    FiledsLayout, ImageLayout, ClientContractLayout = _getFieldsLayout(image_dict, X_dim, Y_dim, landscape=landscape), _getImageLayout(image), _getClientContractLayout(image_dict, last_info)
    n_copy = image_dict["n_copy"] if "n_copy" in list(image_dict.keys()) else 1
    MainLayout = [
        [sg.Text("Attention : Veuillez bien vérifier les champs proposés"), sg.Push(), sg.Button("Consignes d'utilisation", k="-consignes-")],
        [sg.Push(), 
         sg.B("<- Retour", s=10), sg.B("Valider ->", s=10), 
         sg.T("Nombre de copie total : "), sg.I(str(n_copy), key='n_copy', size=(3,1), justification='right'),
         sg.B("Ajouter une commande manuellement", s=10, size=(35, 1)),
         sg.Push()]
    ]

    if add:
        MainLayout = [
        [sg.Push(), sg.Text("AJOUT MANUEL", font=("Helvetica", 36, "bold")), sg.Push()],
        [sg.Text("Attention : Veuillez bien vérifier les champs proposés"), sg.Push()],
        [sg.Push(), 
         sg.B("Valider ->", s=10), 
         sg.T("Nombre de copie total : "), sg.I(str(n_copy), key='n_copy', size=(3,1), justification='right'),
         sg.Push()]
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

def process_contract_client_action(verif_event, verif_values, CONTRACT_LIST, VerificationWindow, ClientSuggestionW, contractSuggestionW):
    if verif_event == "-client_name-":
        client_text = verif_values["-client_name-"]
        client_sugg = sorted([client for client in CLIENT_LIST if client_text.lower() in client.lower()])
        if client_text and client_sugg:
            if ClientSuggestionW :
                ClientSuggestionW.BringToFront()
                ClientSuggestionW["-AUTO_CLIENT-"].update(values=client_sugg)
                ClientSuggestionW.refresh()
            elif len(client_text)>2 :
                ClientSuggestionW = ClientSuggestionWindow(client_sugg, VerificationWindow)
                ClientSuggestionW.BringToFront()              
                                                                
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
                contractSuggestionW.BringToFront()
                contractSuggestionW["-AUTO_CONTRACT-"].update(values=contract_sugg)
                contractSuggestionW.refresh()
            else :
                contractSuggestionW = ContractSuggestionWindow(contract_sugg, VerificationWindow)
                contractSuggestionW.BringToFront()

    if verif_event == '-AUTO_CONTRACT-':
        contractSuggestionW.close()
        contractSuggestionW = False
        text = verif_values['-AUTO_CONTRACT-'][0]
        VerificationWindow['-contract_name-'].update(value=text)
        VerificationWindow['-contract_name-'].set_focus()
    
    return  CONTRACT_LIST, verif_event, verif_values, VerificationWindow, ClientSuggestionW, contractSuggestionW

def check_n_copy(verif_values, X_loc, Y_loc):
    try:
        n_copy = int(verif_values["n_copy"])
        return n_copy, True
    except ValueError:
        sg.popup_ok("Veuillez renseigner un nombre entier de copie", button_color="dark green", location=(X_loc+200, Y_loc+200))
        return None, False

def convertDictToLIMS(verified_dict, CLIENT_CONTRACT_DF): 
    
    def _get_parasite_code(para_list):
        unit_code_list = []
        test_code_list = []
        for u_code, u_para in LIMSsettings["UNIT_TEST_CODE"].items():
            if u_para in para_list:
                unit_code_list.append(u_code)
                
        unit_in_package = [] # store integrated unit test
        package_sorted = []
        
        # Sort package from the biggest to the smaller to prevent the case where small package are included in big to be miss assignated
        for p_code, p_para in LIMSsettings["PACKAGE_TEST_CODE"].items():
            package_sorted.append((p_code, p_para))
        package_sorted = sorted(package_sorted, key = lambda x : len(x[1]), reverse=True)

        for (p_code, p_para) in package_sorted:
            if set(unit_code_list).intersection(set(p_para)) == set(p_para):
                test_code_list.append(p_code)
                unit_in_package += p_para
                unit_code_list = list(set(unit_code_list) - set(unit_in_package))

        return test_code_list, unit_code_list
    
    # Clean fields
    scan_clean_dict = {}
    para = []
    for key, value  in verified_dict.items():
        if "para_" in key and value == True:
            para.append(key.split("para_")[-1])
        if key[0] == "-":
            scan_clean_dict[key] = value
    scan_clean_dict["-parasite_recherche-"] = para
    scan_clean_dict["-parasite_package-"], scan_clean_dict["-parasite_test-"] =_get_parasite_code(para)
        
    # Convert client and contract for the LIMS
    stack_dict = {}
    stack_dict.update({"root" : {}})
    stack_dict.update({"Sample" : {}})
    stack_dict.update({"AdditionalField" : {}})
    
    clientName, contractName = verified_dict["-client_name-"], verified_dict["-contract_name-"]
    corresponding_row = CLIENT_CONTRACT_DF[(CLIENT_CONTRACT_DF["clientname"]==clientName) & (CLIENT_CONTRACT_DF["contractname"]==contractName)]
    if len(corresponding_row) == 1:
        scan_clean_dict["-CustomerCode-"] = list(corresponding_row["clientCode"])[0] #Found way to avoid an object return
        scan_clean_dict["-ContractCode-"] = list(corresponding_row["contractcode"])[0]
        scan_clean_dict["-Devis-"] = list(corresponding_row["Devis"])[0]
        scan_clean_dict["-EngamentJuridique-"] = list(corresponding_row["EngamentJuridique"])[0]
    else :
        scan_clean_dict["-CustomerCode-"] = ""
        scan_clean_dict["-ContractCode-"] = ""
        scan_clean_dict["-Devis-"] = ""
        scan_clean_dict["-EngamentJuridique-"] = ""
    
    for key, name_code in LIMSsettings["LIMS_CONVERTER"].items():
        if f"-{key}-" in list(scan_clean_dict.keys()):
            dict_name, code = name_code
            stack_dict[dict_name][code] = scan_clean_dict[f"-{key}-"]
    stack_dict["Sample"]["AdditionalField"] = stack_dict["AdditionalField"]
    stack_dict["root"]["Sample"] = stack_dict["Sample"]
    return stack_dict["root"]

def runningSave(save_path_json, verified_imageDict, image_name, res_dict, n_copy):
    res_dict["RESPONSE"][image_name].update({"n_copy" : n_copy})
    res_dict["RESPONSE"][image_name].update({"ref_echantillon" : {"sequence" : "" }})
    res_dict["RESPONSE"][image_name].update({"client_name" : {"sequence" : "" }})
    res_dict["RESPONSE"][image_name].update({"contract_name" : {"sequence" : "" }})
    for key, items in verified_imageDict.items():
        if key[0] == "-":
            res_dict["RESPONSE"][image_name][key.strip("-")]["sequence"] = items

    with open(save_path_json, 'w', encoding='utf-8') as f:
        json.dump(res_dict, f,  ensure_ascii=False)
        
def finalSaveDict(verified_dict, CLIENT_CONTRACT_DF, xml_save_path, out_path="", xml_name="verified_XML"):

    def _copy_dict(verified_dict):
        res_dict = {}
        for scan_name, scan_dict in verified_dict.items():
            n_copy = int(scan_dict["n_copy"])
            clean_dict = convertDictToLIMS(scan_dict, CLIENT_CONTRACT_DF)
            if n_copy>1:
                customer_reference = clean_dict["Sample"]["CustomerReference"]
                sample_dict = clean_dict.pop("Sample")
                for i_copy in range(n_copy):
                    # clean_dict[f"Sample_{i_copy+1}"] = deepcopy(sample_dict)
                    # clean_dict[f"Sample_{i_copy+1}"].update({"CustomerReference" : customer_reference + f"_{i_copy+1}"})

                    clean_dict["Sample"] = deepcopy(sample_dict)
                    clean_dict["Sample"].update({"CustomerReference" : customer_reference + f"_{i_copy+1}"})
                    res_dict[scan_name+f"_{i_copy+1}"] = deepcopy(clean_dict)
            else: res_dict[scan_name] = deepcopy(clean_dict)
            # res_dict[scan_name] = clean_dict

        return res_dict
    
    if out_path:
        new_xml = os.path.join(out_path, xml_name)
        if not os.path.exists(new_xml):
            os.makedirs(new_xml)
    
    cleaned_verified_dict = _copy_dict(verified_dict)
    for scan_name, scan_dict in cleaned_verified_dict.items():
        xml = dicttoxml.dicttoxml(scan_dict)
        with open(os.path.join(xml_save_path, f"{scan_name}.xml"), 'w', encoding='utf8') as result_file:
            result_file.write(xml.decode())
        if out_path:
            with open(os.path.join(new_xml, f"{scan_name}.xml"), 'w', encoding='utf8') as result_file:
                result_file.write(xml.decode())

def get_images_and_adapt_landscape(images_names_dict, images, images_names):
    res_images, res_names = [], []
    for i, name in enumerate(images_names):
        loc_im = [images[i]]
        loc_name = [name]
        point_names = [point for point in images_names_dict if (name in point) and ("point" in point)]
        if point_names != []:
            loc_name = point_names
            loc_im = [images[i] for k in point_names]
        res_images += loc_im
        res_names += loc_name
        
    return res_images, res_names          

def manually_add_order(MainLayout, verified_dict, image_name, CONTRACT_LIST, CONTRACT_LIST_0, X_loc, Y_loc, X_dim, Y_dim):
    ClientSuggestionW_add, contractSuggestionW_add = None, None
    client_warning=False
    image_name_add = image_name+"_ajout_main_"
    num = 0
    VerificactionLayout_add = MainLayout
    VerificationWindow_add = sg.Window(f"Fiche {image_name_add}", 
                                VerificactionLayout_add, use_custom_titlebar=True, location=(X_loc+30, Y_loc), 
                                size=(X_dim, Y_dim), resizable=True, finalize=True, )
    while True:
        add_windows, verif_event_add, verif_values_add = sg.read_all_windows()
        if verif_event_add == sg.WINDOW_CLOSED:
            return None, None, None

        if verif_event_add in ["-client_name-", '-AUTO_CLIENT-', "-contract_name-", '-AUTO_CONTRACT-']:
            processed_add = process_contract_client_action(verif_event_add, verif_values_add, CONTRACT_LIST, VerificationWindow_add, ClientSuggestionW_add, contractSuggestionW_add)
            CONTRACT_LIST, verif_event_add, verif_values_add, VerificationWindow_add, ClientSuggestionW_add, contractSuggestionW_add = processed_add

        if verif_event_add == "Valider ->":
            if contractSuggestionW_add : contractSuggestionW_add.close()
            if ClientSuggestionW_add : ClientSuggestionW_add.close()
            if (not verif_values_add["-client_name-"] in CLIENT_LIST or not verif_values_add["-contract_name-"] in CONTRACT_LIST_0) and not client_warning:
                verif_event_add=None
                sg.popup_ok("ATTENTION : VEUILLEZ REMPLIR UN CODE CIENT ET UN CODE CONTRAT VALIDE", button_color="dark green", 
                            location = (X_loc+200, Y_loc+200))
                client_warning=True
                copy_status = False
            # Check "n_copie" format
            else : n_copy, copy_status = check_n_copy(verif_values_add, X_loc+20, Y_loc)

            if copy_status:
                # Register the response and go to the following
                image_name_add_num = image_name_add + str(num)
                while image_name_add_num in list(verified_dict.keys()):
                    image_name_add_num = image_name_add + str(num)
                    num+=1
                VerificationWindow_add.close()

                return verif_values_add, image_name_add_num, n_copy
                # If last image

def main():
    welcomeLayout = [
        [sg.Text("Dossier contenant les PDFs"), sg.Input(LIMSsettings["TOOL_PATH"]["input_folder"], key="-PATH-"), 
         sg.FolderBrowse(button_color="cornflower blue")],
        [sg.Push(), sg.Exit(button_color="tomato"), sg.Push(), sg.Button("Lancer l'algorithme", button_color="medium sea green"), sg.Checkbox(' Format SEMAE', default=False, key=f"landscape"),
          sg.Push()]
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
                images, images_names = getAllImages(givenPath)
                if images == []:
                    sg.popup_ok("Aucun PDF n'est trouvé dans le dossier")
                else :
                    res_path = os.path.join(givenPath, "RES")
                    save_path_json = os.path.join(res_path, "res.json")
                    xml_res_path = os.path.join(res_path, "verified_XML")
                    start = False # Set to true when scans are processed ; condition to start the verification process
                    end = False # Force the verification process to be done or abandonned (no infinit loop)
                    if os.path.exists(save_path_json):
                        continue_or_smash = choice_overwrite_continue_popup("Il semblerait que l'analyse ai déjà été effectuée.\nEcraser la précédente analyse ?"
                                                                   , "Ecraser", "Reprendre la précédente analyse") # Change conditon
                    else : continue_or_smash="overwrite"
                    if continue_or_smash==None : pass
                    if continue_or_smash == "continue":
                        json_file  = open(save_path_json, encoding='utf-8')
                        res_dict_per_image = json.load(json_file)
                        images_names_dict = list(res_dict_per_image["RESPONSE"].keys())
                        images, images_names =  get_images_and_adapt_landscape(images_names_dict, images, images_names)
                        welcomWindow.close()
                        start = True
                        if images_names_dict != images_names:
                            sg.popup_ok(f"Attention : Les images précédemment analysées et celles dans le dossier ne sont pas identiques")
                    if continue_or_smash == "overwrite":
                        sg.popup_auto_close("L'agorithme va démarrer !\nSuivez l'évolution dans le terminal", auto_close_duration=2)
                        welcomWindow.close()
                        print("------ START -----")
                        print("Attendez la barre de chargement \nAppuyez sur ctrl+c dans le terminal pour interrompre")
                        def_format = "landscape" if values["landscape"] else ""
                        images_names, res_dict_per_image, images = use_the_tool(givenPath, def_format=def_format)
                        images_names_dict = list(res_dict_per_image["RESPONSE"].keys())
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
                            last_info = ["", ""]
                            while n_image < len(images_names):
                                if n_image != n_displayed:
                                    client_warning=False
                                    if start == True :
                                        X_loc, Y_loc = VerificationWindow.current_location()
                                        X_dim, Y_dim = fit_the_screen(10)
                                    image_name = images_names_dict[n_image]
                                    image = images[n_image]
                                    image = np.rot90(image, int(image_name[image_name.index("k90_")+len("k90_")])) if values["landscape"] else image # Landscape rota get by the end
                                    image_dict = res_dict_per_image["RESPONSE"][image_name]
                                    VerificactionLayout = getMainLayout(image_dict, image, X_dim, Y_dim, last_info=last_info, landscape=values["landscape"])
                                    if start == True:
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

                                if verif_event in ["-client_name-", '-AUTO_CLIENT-', "-contract_name-", '-AUTO_CONTRACT-']:
                                    processed = process_contract_client_action(verif_event, verif_values, CONTRACT_LIST, VerificationWindow, ClientSuggestionW, contractSuggestionW)
                                    CONTRACT_LIST, verif_event, verif_values, VerificationWindow, ClientSuggestionW, contractSuggestionW = processed

                                if verif_event == "Valider ->":
                                    if contractSuggestionW : contractSuggestionW.close()
                                    if ClientSuggestionW : ClientSuggestionW.close()
                                    if (not verif_values["-client_name-"] in CLIENT_LIST or not verif_values["-contract_name-"] in CONTRACT_LIST_0) and not client_warning:
                                            verif_event=None
                                            sg.popup_ok("ATTENTION : VEUILLEZ REMPLIR UN CODE CIENT ET UN CODE CONTRAT VALIDE", button_color="dark green", 
                                                        location = (X_loc+200, Y_loc+200))
                                            client_warning=True
                                            copy_status = False
                                    # Check "n_copie" format
                                    else : n_copy, copy_status = check_n_copy(verif_values, X_loc, Y_loc)

                                    if copy_status:
                                        # If not last image
                                        last_info = [verif_values["-client_name-"], verif_values['-contract_name-']]
                                        if not n_image == len(images_names)-1:
                                            # Register the response and go to the following
                                            verified_dict[image_name] = verif_values
                                            runningSave(save_path_json, verif_values, image_name, res_dict_per_image, n_copy)
                                            n_image+=1
                                        # If last image
                                        else:
                                            verified_dict[image_name] = verif_values
                                            runningSave(save_path_json, verif_values, image_name, res_dict_per_image, n_copy)
                                            choice = sg.popup_ok(f"Il n'y a pas d'image suivante.\n\n{len(verified_dict)} fichiers vont être générés.\n\nFinir l'analyse ?", 
                                                                 button_color="dark green")
                                            if choice == "OK":
                                                json_file.close() # Close the file
                                                finalSaveDict(verified_dict, CLIENT_CONTRACT_DF, xml_res_path, out_path=LIMSsettings["TOOL_PATH"]["output_folder"])
                                                VerificationWindow.close()
                                                break

                                if verif_event == "Ajouter une commande manuellement":
                                    VerificationWindow.Disable()
                                    VerificationWindow.SetAlpha(0.5)
                                    new_CONTRACT_LIST = CONTRACT_LIST_0
                                    addLayout = getMainLayout(image_dict, image, X_dim, Y_dim, last_info=last_info, landscape=values["landscape"], add=True)
                                    verif_values_add, image_name_add, n_copy = manually_add_order(addLayout, verified_dict, image_name, new_CONTRACT_LIST, CONTRACT_LIST_0, X_loc, Y_loc, X_dim, Y_dim)
                                    if verif_values_add:
                                        verified_dict[image_name_add] = verif_values_add
                                        res_dict_per_image["RESPONSE"][image_name_add] = deepcopy(image_dict)
                                        runningSave(save_path_json, verif_values_add, image_name_add, res_dict_per_image, n_copy)
                                    VerificationWindow.Enable()
                                    VerificationWindow.SetAlpha(1)
                                    

                                if verif_event == "<- Retour":
                                    if n_image>0:
                                        runningSave(save_path_json, verif_values, image_name, res_dict_per_image, n_copy)
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

    lists_df = pd.read_excel(r"CONFIG\\lists.xlsx")
    PARASITES_LIST = list(lists_df["parasite"].dropna())

    LIMSCodeXlsxPath = os.path.join(r"CONFIG\client_contract.xlsx")
    CLIENT_CONTRACT_DF = pd.read_excel(LIMSCodeXlsxPath, dtype=str).fillna("")
    clientName, contractName = CLIENT_CONTRACT_DF.iloc[0]["clientname"], CLIENT_CONTRACT_DF.iloc[0]["contractname"]
    row = CLIENT_CONTRACT_DF[(CLIENT_CONTRACT_DF["clientname"]==clientName) & (CLIENT_CONTRACT_DF["contractname"]==contractName)]
    CLIENT_LIST = list(CLIENT_CONTRACT_DF["clientname"].unique())
    main()
