import PySimpleGUI as sg
import sys, os
import shutil
import numpy as np
import io
import json
import pandas as pd
from copy import deepcopy

from screeninfo import get_monitors
from PIL import Image

from LaunchTool import getAllImages, TextCVTool
from SaveDict import runningSave, finalSaveDict, saveToCopyFolder

def is_valid(folderpath, model):

    if len(model) != 1:
        sg.popup_error("Veuillez cocher un model")

    if not (folderpath and os.path.exists(folderpath)):
        sg.popup_error("Le dossier n'existe pas")
        return False
    
    scan_dict = getAllImages(folderpath)

    if scan_dict == {}:
        sg.popup_ok("Aucun PDF n'est trouvé dans le dossier")
        return False
    
    return scan_dict

def fit_the_screen(X_loc):
    ordered_monitors = sorted([monitor for monitor in get_monitors()], key=lambda mtr: mtr.x)
    for monitor in ordered_monitors:
        x, width, height = monitor.x, monitor.width, monitor.height
        if X_loc<x+width:
            return int(0.95*width), int(0.95*height)

def use_the_tool(folderPath, model="Fredon"):
    scan_dict, pdfs_res_dict = TextCVTool(folderPath, model=model)
    return scan_dict, pdfs_res_dict

def checkboxes_for_analysis(analysis_list, found_analysis):
    not_found = list(set(analysis_list)-set(found_analysis))
    found_col = [[sg.Checkbox(str(fanalyse), default=True, key=("ana", fanalyse))] for i, fanalyse in enumerate(found_analysis,1)]
    not_found_col = [[sg.Checkbox(str(nfanalyse), key=("ana", nfanalyse))] for i, nfanalyse in enumerate(not_found, len(found_analysis)+1)]
    
    return found_col, not_found_col

def _getFieldsLayout(image_dict, X_dim, Y_dim, analysis_list, model="Fredon"):
    conversion_dict = LIMS_HELPER["CLEAN_ZONE_NAMES"]
    product_codes = LIMS_HELPER["PRODUCTCODE_DICT"]
    conf_threshold = int(GUIsettings["TOOL"]["confidence_threshold"])/100   
    lineLayout = []

    added_field = LIMS_HELPER["ADDED_FIELDS"]

    # Combo for Reference Echantillon
    default_ref = "Sol" if ((image_dict["variete"]["sequence"] == []) or model=="SEMAE") else "Tubercule"

    pructcode_combo = [sg.Text("Référence échantillon : ", s=(25,1)),sg.Combo(list(product_codes.keys()), default_value=default_ref,
            key="code_produit", expand_y=True, expand_x=False, size=(INPUT_LENGTH, 1))]
            
    lineLayout.append(pructcode_combo)

    # Returned the tool's response for each field
    for zone, landmark_text_dict in image_dict.items():
        if not zone in added_field:
            clean_zone_name = conversion_dict[zone]
            text = f"{clean_zone_name} : "
            sequence = landmark_text_dict["sequence"]
            conf= landmark_text_dict["confidence"] # If empty must not be red
            if conf < conf_threshold or sequence=="":
                back_color = 'red'
            else: 
                back_color = sg.theme_input_background_color()
                
            if zone == "analyse":
                n_list = len(sequence)
                found_col, not_found_col = checkboxes_for_analysis(analysis_list, sequence)
                text = f"{clean_zone_name} : {n_list} proposé.s"
                lineLayout.append([sg.Text(text, s=(25,1)), sg.Column(found_col, size=(int(X_dim*0.15), int(Y_dim*0.035*len(found_col))), scrollable=True)])
                # lineLayout.append([sg.HorizontalSeparator(key='sep')])
                lineLayout.append([sg.Text("Ajouter si besoin", s=(25,1)), sg.Column(not_found_col, scrollable=True, size=(int(X_dim*0.15),
                                                                                                                            int(Y_dim*0.035*len(not_found_col))))])
            elif zone == "type_lot":
                type_list = list(LIMS_HELPER["TYPE_LOT"].keys())
                default = "Surveillance"
                for type in type_list:
                    if sequence in LIMS_HELPER["TYPE_LOT"][type]:
                        default=type
                        break
                lineLayout.append([sg.Text(text, s=(25,1)),
                                sg.Combo(type_list, default_value=default, background_color=back_color,
                                key=("zone", zone), expand_y=True, expand_x=False, size=(INPUT_LENGTH, 1))])

            elif len(sequence)>=INPUT_LENGTH:
                lineLayout.append([sg.Text(text, s=(25,1)),
                                sg.Multiline(sequence, background_color=back_color,
                                key=("zone", zone), expand_y=True, expand_x=False, size=(INPUT_LENGTH, (len(sequence)//INPUT_LENGTH)+1), justification='left')])
            else :
                lineLayout.append([sg.Text(text, s=(25,1)),
                                sg.I(sequence, background_color=back_color,
                                key=("zone", zone), expand_y=True, expand_x=False, size=(INPUT_LENGTH, 1), justification='left')])
                                # sg.Image(data=bio.getvalue(), key=f'image_{zone}')])
    
    return lineLayout
    
def _getImageLayout(image):
    
    searching_area = Image.fromarray(image).resize((int(image.shape[1]*0.4), int(image.shape[0]*0.4)))
    bio = io.BytesIO()
    searching_area.save(bio, format="PNG")
    imageLayout = [[sg.Image(data=bio.getvalue(), key=f'scan')]]
    
    return imageLayout

def _getClientContractLayout(image_dict, last_client_contract):
    
    ClientContractLayout = []
    
    client, contract = "", ""
    if "client_name" in list(image_dict.keys()):
        client = image_dict["client_name"]
    if  "contract_name" in list(image_dict.keys()):
        contract = image_dict["contract_name"]
    
    if not client:
        client = last_client_contract[0]
    if not contract:
        contract = last_client_contract[1]
        
    ClientContractLayout.append([sg.Text('N° de client', size=(25, 1)) ,sg.Input(client, size=(INPUT_LENGTH, 1), enable_events=True, key=("to_save", "client_name"))])
    ClientContractLayout.append([sg.Text('N° de contrat', size=(25, 1)) ,sg.Input(contract, size=(INPUT_LENGTH, 1), enable_events=True, key=("to_save", "contract_name"), background_color='light gray')])
    ClientContractLayout.append([sg.HorizontalSeparator(key='sep')])
    
    return ClientContractLayout

def ClientSuggestionWindow(values, mainWindow, verif_event):
    x = mainWindow[verif_event].Widget.winfo_rootx()
    y = mainWindow[verif_event].Widget.winfo_rooty() + 25
    if not values:
        return
    layout = [
        [sg.Listbox(values, size=(50, 5), enable_events=True, key='-AUTO_CLIENT-',
            select_mode=sg.LISTBOX_SELECT_MODE_SINGLE)]]
    return sg.Window('Title', layout, no_titlebar=True, keep_on_top=True,
        location=(x, y), margins=(0, 0), finalize=True)

def ContractSuggestionWindow(values, mainWindow,  verif_event):
    x = mainWindow[verif_event].Widget.winfo_rootx()
    y = mainWindow[verif_event].Widget.winfo_rooty() + 25
    if not values:
        return
    layout = [
        [sg.Listbox(values, size=(50, 5), enable_events=True, key='-AUTO_CONTRACT-',
            select_mode=sg.LISTBOX_SELECT_MODE_SINGLE)]]
    return sg.Window('Title', layout, no_titlebar=True, keep_on_top=True,
        location=(x, y), margins=(0, 0), finalize=True)

def getMainLayout(image_dict, image, X_dim, Y_dim, last_client_contract, analysis_list, model, add=False):
    FiledsLayout, ImageLayout, ClientContractLayout = _getFieldsLayout(image_dict, X_dim, Y_dim, analysis_list, model=model), _getImageLayout(image), _getClientContractLayout(image_dict, last_client_contract)
    n_start = image_dict['n_start'] if 'n_start' in list(image_dict.keys()) else ""
    n_end = image_dict['n_end'] if 'n_end' in list(image_dict.keys()) else ""
    with_0 = image_dict["with_0"] if 'with_0' in list(image_dict.keys()) else False

    MainLayout = [
        [sg.Text("Attention : Veuillez bien vérifier les champs proposés"), sg.Push(), sg.Button("Consignes d'utilisation", k="-consignes-")],

        [sg.Push(), 
         sg.B("<- Retour", s=10), sg.B("Valider ->", s=10), 
         sg.Push(), sg.B("Ajouter une commande manuellement", s=10, size=(35, 1)), sg.Push()],

        [sg.Push(), sg.T("Pour ajouter des copies : Index de début"), sg.I(str(n_start), key=("to_save", 'n_start'), size=(3,1), justification='right', ),
         sg.T("Index de fin"), sg.I(str(n_end), key=("to_save", 'n_end'), size=(3,1), justification='right'),
         sg.Checkbox("Incrémenter avec 0", key=("to_save", "with_0"), default=with_0),
         sg.Push()]
    ]

    if add:
        MainLayout = [
        [sg.Push(), sg.Text("AJOUT MANUEL", font=("Helvetica", 36, "bold")), sg.Push()],
        [sg.Text("Attention : Veuillez bien vérifier les champs proposés"), sg.Push()],
        [sg.Push(), 
         sg.B("Valider ->", s=10),
         sg.Push()],
        [sg.Push(), sg.T("Pour ajouter des copies : Index de début"), sg.I(str(n_start), key=("to_save", "n_start"), size=(3,1), justification='right'),
         sg.T("Index de fin"), sg.I(str(n_end), key=("to_save", "n_end"), size=(3,1), justification='right'),
         sg.Checkbox("Incrémenter avec 0",  key=("to_save", "with_0")),
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
    if verif_event[1] == "client_name":
        client_text = verif_values[("to_save", "client_name")]
        client_sugg = sorted([client for client in CLIENT_LIST if client_text.lower() in client.lower()])
        if client_text and client_sugg:
            try:
                if ClientSuggestionW:
                    ClientSuggestionW.BringToFront()
                    ClientSuggestionW["-AUTO_CLIENT-"].update(values=client_sugg)
                    ClientSuggestionW.refresh()
                elif len(client_text)>2 :
                    ClientSuggestionW = ClientSuggestionWindow(client_sugg, VerificationWindow, verif_event)
                    ClientSuggestionW.BringToFront()             
            except:
                pass 
                                                                
    if verif_event == '-AUTO_CLIENT-':
        ClientSuggestionW.close()
        ClientSuggestionW = False
        text = verif_values['-AUTO_CLIENT-'][0]
        VerificationWindow[("to_save", "client_name")].update(value=text)
        
        CONTRACT_LIST = sorted(list(CLIENT_CONTRACT[CLIENT_CONTRACT["ClientName"]== text]["ContractName"].unique()))
        VerificationWindow[("to_save", "contract_name")].update(disabled=False, background_color=sg.theme_input_background_color())
        VerificationWindow[("to_save", "client_name")].set_focus()
        
    if verif_event[1] == "contract_name":
        contract_text = verif_values[("to_save", "contract_name")]
        contract_sugg = [contract for contract in CONTRACT_LIST if contract_text.lower() in contract.lower()]
        if contract_text and contract_sugg:
            if contractSuggestionW :
                contractSuggestionW.BringToFront()
                contractSuggestionW["-AUTO_CONTRACT-"].update(values=contract_sugg)
                contractSuggestionW.refresh()
            else :
                contractSuggestionW = ContractSuggestionWindow(contract_sugg, VerificationWindow, verif_event)
                contractSuggestionW.BringToFront()

    if verif_event == '-AUTO_CONTRACT-':
        contractSuggestionW.close()
        contractSuggestionW = False
        text = verif_values['-AUTO_CONTRACT-'][0]
        VerificationWindow[("to_save", "contract_name")].update(value=text)
        VerificationWindow[("to_save", "contract_name")].set_focus()
    
    return  CONTRACT_LIST, verif_event, verif_values, VerificationWindow, ClientSuggestionW, contractSuggestionW

def get_image_and_adapt_semae(scan_dict, pdf_name, sample_image_extract, model):
    l_image = []
    for image in sample_image_extract["IMAGE"].split("+"):
        if model=="SEMAE":
            image = np.rot90(scan_dict[pdf_name][image], sample_image_extract["k_90"]) # SEMAE rota get by the end
        else:
            image = scan_dict[pdf_name][image]

        l_image.append(image)

    return np.concatenate(l_image)

    # res_images, res_names = [], []
    # for i, name in enumerate(images_names):
    #     loc_im = [images[i]]
    #     loc_name = [name]
    #     point_names = [point for point in images_names_dict if (name in point) and ("point" in point)]
    #     if point_names != []:
    #         loc_name = point_names
    #         loc_im = [images[i] for k in point_names]
    #     res_images += loc_im
    #     res_names += loc_name
        
    # return res_images, res_names          

def manually_add_order(MainLayout, verified_dict, image_name, CONTRACT_LIST, CONTRACT_LIST_0, X_loc, Y_loc, X_dim, Y_dim):
    ClientSuggestionW_add, contractSuggestionW_add = None, None
    client_warning=False
    num=0
    image_name_add = image_name+"_ajout_main_"
    VerificactionLayout_add = MainLayout
    VerificationWindow_add = sg.Window(f"Fiche {image_name_add}", 
                                VerificactionLayout_add, use_custom_titlebar=True, location=(X_loc+30, Y_loc), 
                                size=(X_dim, Y_dim), resizable=True, finalize=True, )
    while True:
        add_windows, verif_event_add, verif_values_add = sg.read_all_windows()
        if verif_event_add == sg.WINDOW_CLOSED:
            VerificationWindow_add.close()
            return None, None, None, None, None

        if verif_event_add in [("to_save", "client_name"), '-AUTO_CLIENT-', ("to_save", "contract_name"), '-AUTO_CONTRACT-']:
            processed_add = process_contract_client_action(verif_event_add, verif_values_add, CONTRACT_LIST, VerificationWindow_add, ClientSuggestionW_add, contractSuggestionW_add)
            CONTRACT_LIST, verif_event_add, verif_values_add, VerificationWindow_add, ClientSuggestionW_add, contractSuggestionW_add = processed_add

        if verif_event_add == "Valider ->":
            if contractSuggestionW_add : contractSuggestionW_add.close()
            if ClientSuggestionW_add : ClientSuggestionW_add.close()
            if (not verif_values_add[("to_save", "client_name")] in CLIENT_LIST or not verif_values_add[("to_save", "contract_name")] in CONTRACT_LIST_0) and not client_warning:
                verif_event_add=None
                sg.popup_ok("ATTENTION : VEUILLEZ REMPLIR UN CODE CIENT ET UN CODE CONTRAT VALIDE", button_color="dark green", 
                            location = (X_loc+200, Y_loc+200))
                client_warning=True

            # Register the response and go to the following
            image_name_add_num = image_name_add + str(num)
            while image_name_add_num in list(verified_dict.keys()):
                image_name_add_num = image_name_add + str(num)
                num+=1
            VerificationWindow_add.close()

            return verif_values_add, image_name_add_num

def welcomeLayout():

    welcomeLayout = [
        [sg.Text("Dossier contenant les PDFs"), sg.Input(LIMS_HELPER["TOOL_PATH"]["input_folder"], key="-PATH-"), 
         sg.FolderBrowse(button_color="cornflower blue")],

        [sg.Push(), sg.Text("Client :")] + [sg.Radio(model, group_id="model", key=model) for model in LIMS_HELPER["MODELS"]] + [sg.Push()],
        
        [sg.Push(), sg.Exit(button_color="tomato"), sg.Button("Lancer l'algorithme", button_color="medium sea green"), sg.Push()]
    ]
    
    return welcomeLayout

def main():

    window_title = GUIsettings["GUI"]["title"]
    welcomWindow = sg.Window(window_title, welcomeLayout(), use_custom_titlebar=True, finalize=True)
    ClientSuggestionW, contractSuggestionW = None, None
    while True:
        event, values = welcomWindow.read()
        welcomWindow.BringToFront()
        if event in (sg.WINDOW_CLOSED, "Exit"):
            break
        if event == "Lancer l'algorithme":

            model = [model for model in LIMS_HELPER["MODELS"] if values[model]]
            givenPath = values["-PATH-"]
            
            # If everything is valid return the scan dict
            scan_dict = is_valid(givenPath, model)

            if scan_dict:

                # SPECIFIC
                MODEL = model[0]
                # Set the path to load the res.json file
                res_path = os.path.join(givenPath, "RES")
                res_save_path = os.path.join(res_path, "res.json")

                # Set the path to place the XML, if it's not exist then creat it
                xml_save_path = LIMS_HELPER["TOOL_PATH"]["output_folder"] if LIMS_HELPER["TOOL_PATH"]["output_folder"] else os.path.join(res_path, "verified_XML")
                if not os.path.exists(xml_save_path):
                    os.makedirs(xml_save_path)

                # Ask if the tool is going to overwrite the found result of tak eit back
                continue_or_smash = "overwrite"
                if os.path.exists(res_save_path):
                    continue_or_smash = choice_overwrite_continue_popup("Il semblerait que l'analyse ai déjà été effectuée.\nEcraser la précédente analyse ?"
                                                                , "Ecraser", "Reprendre la précédente analyse") #Change conditon

                # If the tool take last results back
                if continue_or_smash == "continue":
                    json_file  = open(res_save_path, encoding='utf-8')
                    res_dict = json.load(json_file)
                    welcomWindow.close()
                    if set(list(res_dict["RESPONSE"].keys())) != set(list(scan_dict.keys())):
                        sg.popup_ok(f"Attention : Les images précédemment analysées et celles dans le dossier ne sont pas identiques")
                
                # If the tool have to launch the extraction process
                if continue_or_smash == "overwrite":
                    sg.popup_auto_close("L'agorithme va démarrer !\nSuivez l'évolution dans le terminal", auto_close_duration=2)
                    welcomWindow.close()

                    print("_________ START _________\nAttendez la barre de chargement \nAppuyez sur ctrl+c dans le terminal pour interrompre")
                
                    scan_dict, res_dict = use_the_tool(givenPath, MODEL)

                    print("_________ DONE _________")

                # Save the extraction json on RES
                with open(res_save_path, 'w', encoding='utf-8') as json_file:
                    json.dump(res_dict, json_file,  ensure_ascii=False)

                pdfs_res_dict = res_dict["RESPONSE"]

                SuggestionW = None

                is_loop_started = False # Force the verification process to be done or abandonned (no infinit loop)

                while not is_loop_started:
                    is_loop_started = True 
                    is_first_step = True

                    # Reset suggestions
                    CONTRACT_LIST_0 = list(CLIENT_CONTRACT["ContractName"].unique())
                    CONTRACT_LIST = CONTRACT_LIST_0
                    last_client_contract = ["", ""]
                    MODEL_ANALYSIS = pd.read_excel(OCR_HELPER["PATHES"]["contract_analysis_path"], sheet_name="analyse")
                    analysis_list = MODEL_ANALYSIS["Denomination"].dropna().unique().tolist()

                    n_pdf = 0
                    n_sample = 0
                    n_displayed= (-1, -1)
                    X_dim, Y_dim = fit_the_screen(1)
                    X_loc, Y_loc = (10,10)

                    # Last PDF
                    n_pdf_end = len(list(pdfs_res_dict.keys()))
                    
                    # Last pdf, last sample
                    last_sample = (n_pdf_end-1, len(list(pdfs_res_dict[list(pdfs_res_dict.keys())[-1]].keys()))-1)

                    while n_pdf < n_pdf_end:
                        pdf_name, samples_res_dict = list(pdfs_res_dict.items())[n_pdf]
                        # last sample of the pdf
                        n_sample_end = len(samples_res_dict.keys())
                        
                        while n_sample < n_sample_end:
                            sample_name, sample_image_extract = list(samples_res_dict.items())[n_sample]
                            # Identify the current sample
                            n_place = (n_pdf, n_sample)

                            # If new displayed sample is aked
                            if n_place != n_displayed:
                                
                                # Close the window if a new one is going to be display
                                if is_first_step == False :
                                    X_loc, Y_loc = VerificationWindow.current_location()
                                    X_dim, Y_dim = fit_the_screen(10)
                                    VerificationWindow.close()
                                # Set client information as default
                                client_warning=False
                                
                                # Get all information to display
                                image =  get_image_and_adapt_semae(scan_dict, pdf_name, sample_image_extract, MODEL)
                                extract_dict = sample_image_extract["EXTRACTION"]
                                # Create the new window
                                VerificactionLayout = getMainLayout(extract_dict, image, X_dim, Y_dim, last_client_contract=last_client_contract, analysis_list=analysis_list, model=MODEL)

                                VerificationWindow = sg.Window(f"PDF {pdf_name} - Commande ({sample_name})", 
                                                            VerificactionLayout, use_custom_titlebar=True, location=(X_loc, Y_loc), 
                                                            size=(X_dim, Y_dim), resizable=True, finalize=True)
                                # Set displayed sample index
                                n_displayed = n_place
                            
                            is_first_step = False

                            while n_place == n_displayed:

                                window, verif_event, verif_values = sg.read_all_windows()

                                if verif_event == sg.WINDOW_CLOSED:
                                    return

                                if verif_event == "-consignes-":
                                    sg.popup_scrolled(GUIsettings["UTILISATION"]["texte"], title="Consignes d'utilisation", size=(50,10))

                                # Client and contract case
                                if verif_event in [("to_save", "client_name"), '-AUTO_CLIENT-', ("to_save", "contract_name"), '-AUTO_CONTRACT-']:

                                    processed = process_contract_client_action(verif_event, verif_values, CONTRACT_LIST, VerificationWindow, ClientSuggestionW, contractSuggestionW)
                                    CONTRACT_LIST, verif_event, verif_values, VerificationWindow, ClientSuggestionW, contractSuggestionW = processed
                                
                                if verif_event == "Ajouter une commande manuellement":
                                    # Disable main sugg windows
                                    VerificationWindow.Disable()
                                    VerificationWindow.SetAlpha(0.5)

                                    # Generate the add layout
                                    VerfifLayout_add = getMainLayout(extract_dict, image, X_dim, Y_dim, last_client_contract=last_client_contract, analysis_list=analysis_list, model=MODEL, add=True)

                                    new_CONTRACT_LIST = CONTRACT_LIST_0
                                    verif_values_add, sample_name_add = manually_add_order(VerfifLayout_add, res_dict, sample_name, new_CONTRACT_LIST, CONTRACT_LIST_0, X_loc, Y_loc, X_dim, Y_dim)
                                    if verif_values_add:
                                        res_dict["RESPONSE"][pdf_name][sample_name_add] = deepcopy(res_dict["RESPONSE"][pdf_name][sample_name])
                                        runningSave(res_dict, res_save_path, verif_values_add, pdf_name, sample_name_add)
                                    
                                    VerificationWindow.Enable()
                                    VerificationWindow.SetAlpha(1)
                                    
                                if verif_event == "<- Retour":
                                    if n_pdf>0 or n_sample>0:
                                        runningSave(res_dict, res_save_path, verif_values, pdf_name, sample_name)
                                        # Go to the past sample
                                        n_sample-=1
                                        # If it's from the past pdf change the pdf number
                                        if n_pdf>0 and n_sample==-1:
                                            n_pdf-=1
                                            n_sample = 0
                                        n_place = (n_pdf, n_sample)  

                                if verif_event == "Valider ->":
                                    
                                    if contractSuggestionW : contractSuggestionW.close()
                                    if ClientSuggestionW : ClientSuggestionW.close()
                                    if (not verif_values[("to_save", "client_name")] in CLIENT_LIST or not verif_values[("to_save", "contract_name")] in CONTRACT_LIST_0) and not client_warning:
                                            verif_event=None
                                            sg.popup_ok("ATTENTION : VEUILLEZ REMPLIR UN CODE CIENT ET UN CODE CONTRAT VALIDE", button_color="dark green", 
                                                        location = (X_loc+200, Y_loc+200))
                                            client_warning=True

                                    last_client_contract = [verif_values[("to_save", "client_name")], verif_values[("to_save", "contract_name")]]
                                    # To fit the wanted "numero d'echantillon"

                                    # If not last image
                                    if not n_place == last_sample:
                                        runningSave(res_dict, res_save_path, verif_values, pdf_name, sample_name)
                                        n_sample+=1
                                        n_place = (n_pdf, n_sample)

                                    # If last image
                                    else:
                                        final_dict = runningSave(res_dict, res_save_path, verif_values, pdf_name, sample_name)
                                        
                                        choice = sg.popup_ok(f"Il n'y a pas d'image suivante. Finir l'analyse ?", 
                                                                button_color="dark green")
                                        if choice == "OK":
                                            json_file.close() # Close the file
                                            finalSaveDict(final_dict["RESPONSE"], xml_save_path, analysis_lims=MODEL_ANALYSIS, model=MODEL, lims_helper=LIMS_HELPER,
                                                            client_contract=CLIENT_CONTRACT)
                                            if LIMS_HELPER["TOOL_PATH"]["copy_folder"]:
                                                saveToCopyFolder(LIMS_HELPER["TOOL_PATH"]["copy_folder"], os.path.join(givenPath, pdf_name+".pdf"), rename=pdf_name+"AA")
                                            VerificationWindow.close()
                                            return
                                        
                        # If n_sample exceed the n_sample_end, go to the next one from the folowing pdf
                        n_sample = 0
                        n_pdf+=1

                    if VerificationWindow :  VerificationWindow.close()
                    if SuggestionW : SuggestionW.close()

        # welcomWindow.close()
                
if __name__ == "__main__":
    # Get the base path if executable or launch directly
    if 'AppData' in sys.executable:
        application_path = os.getcwd()

    else : 
        application_path = os.path.dirname(sys.executable)

    # Load helper
    OCR_HELPER_JSON_PATH  = os.path.join(application_path, "CONFIG\OCR_CONFIG.json")
    OCR_HELPER = json.load(open(OCR_HELPER_JSON_PATH))

    LIMS_HELPER_JSON_PATH  = os.path.join(application_path, "CONFIG\LIMS_CONFIG.json")
    LIMS_HELPER= json.load(open(LIMS_HELPER_JSON_PATH))

    CLIENT_CONTRACT = pd.read_excel(os.path.join(application_path, OCR_HELPER["PATHES"]["contract_analysis_path"]), sheet_name='client_contract').fillna("")
    clientName, contractName = CLIENT_CONTRACT.iloc[0]["ClientName"], CLIENT_CONTRACT.iloc[0]["ContractName"]
    row = CLIENT_CONTRACT[(CLIENT_CONTRACT["ClientName"]==clientName) & (CLIENT_CONTRACT["ContractName"]==contractName)]
    CLIENT_LIST = list(CLIENT_CONTRACT["ClientName"].unique())
    
    # Set interface's graphical settings
    GUIsettings = sg.UserSettings(path=os.path.join(application_path, "CONFIG"), filename="GUI_CONFIG.ini", use_config_file=True, convert_bools_and_none=True)

    theme = GUIsettings["GUI"]["theme"]
    font_family = GUIsettings["GUI"]["font_family"]
    font_size = int(GUIsettings["GUI"]["font_size"])
    help_text = GUIsettings["UTILISATION"]["text"]    
    sg.theme(theme)
    sg.set_options(font=(font_family, font_size))
    INPUT_LENGTH = 45

    main()
