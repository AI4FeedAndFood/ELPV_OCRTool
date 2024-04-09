import numpy as np
import matplotlib.pyplot as plt
import cv2
import json

OCR_HELPER_JSON_PATH  = r"CONFIG\\OCR_config.json"
CONFIG_CHECK_DICT = json.load(open(OCR_HELPER_JSON_PATH, encoding="utf-8"))["PATHES"]["checkbox"]

cross_checkbox_path = CONFIG_CHECK_DICT["Fredon avec cases"]["type_lot"]["cross_path"]
empty_checkbox_path = CONFIG_CHECK_DICT["Fredon avec cases"]["type_lot"]["empty_path"]

def preprocessed_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    return thresh

class Template:
    """
    A class defining a template
    """
    def __init__(self, image_path, label, color, matching_threshold=0.5, transform_list=[lambda x:x]):
        """
        Args:
            image_path (str): path of the template image path
            label (str): the label corresponding to the template
            color (List[int]): the color associated with the label (to plot detections)
            matching_threshold (float): the minimum similarity score to consider an object is detected by template
                matching
        """
        self.image_path = image_path
        self.label = label
        self.color = color
        self.transform_list = transform_list
        self.template = preprocessed_image(cv2.imread(image_path))
        self.template_height, self.template_width = self.template.shape[:2]
        self.matching_threshold = matching_threshold
        
        self.transformed_template = self.transform(self.template, self.transform_list)
        
    def transform(cls, template, transform):
        return [trans(template) for trans in transform]

TRANSFORM = [lambda x: x, lambda x: cv2.flip(x,0), lambda x: cv2.flip(x,1), lambda x: cv2.resize(x, (int(x.shape[1]*1.15), x.shape[0])),
             lambda x: cv2.resize(x, (x.shape[1], int(x.shape[0]*1.15)))] # Maybe can be cleaner with a transform class

CHECK_TEMPLATES = [Template(image_path=cross_checkbox_path, label="cross", color=(0, 0, 0), matching_threshold=0.70, transform_list=TRANSFORM),
              Template(image_path=empty_checkbox_path, label="empty", color=(0, 0, 0), matching_threshold=0.70, transform_list=TRANSFORM)]

def checkbox_match(templates, cropped_image):
    detections = []
    for i, template in enumerate(templates):
        w, h = template.template_width, template.template_height
        for transformed in template.transformed_template:
            template_matching = cv2.matchTemplate(transformed, cropped_image, cv2.TM_CCOEFF_NORMED)
            match_locations = np.where(template_matching >= template.matching_threshold)
        
            for (x, y) in zip(match_locations[1], match_locations[0]):
                match = {
                    "TOP_LEFT_X": x,
                    "TOP_LEFT_Y": y,
                    "BOTTOM_RIGHT_X": x + w,
                    "BOTTOM_RIGHT_Y": y + h,
                    "MATCH_VALUE": template_matching[y, x],
                    "LABEL" : template.label,
                    "COLOR": (0, 191, 255)
                }
                detections.append(match)
    return detections

def get_iou(a, b, epsilon=1e-5):
    """ Given two boxes `a` and `b` defined as a list of four numbers:
            [x1,y1,x2,y2]
        where:
            x1,y1 represent the upper left corner
            x2,y2 represent the lower right corner
        It returns the Intersect of Union score for these two boxes.

    Args:
        a:          (list of 4 numbers) [x1,y1,x2,y2]
        b:          (list of 4 numbers) [x1,y1,x2,y2]
        epsilon:    (float) Small value to prevent division by zero

    Returns:
        (float) The Intersect of Union score.
    """
    # COORDINATES OF THE INTERSECTION BOX
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    # AREA OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)
    # handle case where there is NO overlap
    if (width<0) or (height <0):
        return 0.0
    area_overlap = width * height

    # COMBINED AREA
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_combined = area_a + area_b - area_overlap

    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined+epsilon)
    return iou

def non_max_suppression(objects, non_max_suppression_threshold=0.2, score_key="MATCH_VALUE"):
    """
    Filter objects overlapping with IoU over threshold by keeping only the one with maximum score.
    Args:
        objects (List[dict]): a list of objects dictionaries, with:
            {score_key} (float): the object score
            {top_left_x} (float): the top-left x-axis coordinate of the object bounding box
            {top_left_y} (float): the top-left y-axis coordinate of the object bounding box
            {bottom_right_x} (float): the bottom-right x-axis coordinate of the object bounding box
            {bottom_right_y} (float): the bottom-right y-axis coordinate of the object bounding box
        non_max_suppression_threshold (float): the minimum IoU value used to filter overlapping boxes when
            conducting non max suppression.
        score_key (str): score key in objects dicts
    Returns:
        List[dict]: the filtered list of dictionaries.
    """
    sorted_objects = sorted(objects, key=lambda obj: obj[score_key], reverse=True)
    filtered_objects = []
    for object_ in sorted_objects:
        overlap_found = False
        for filtered_object in filtered_objects:
            coord1 = [object_["TOP_LEFT_X"],object_["TOP_LEFT_Y"], object_["BOTTOM_RIGHT_X"], object_["BOTTOM_RIGHT_Y"]]
            coord2 = [filtered_object["TOP_LEFT_X"],filtered_object["TOP_LEFT_Y"],filtered_object ["BOTTOM_RIGHT_X"], filtered_object["BOTTOM_RIGHT_Y"]]
            iou = get_iou(coord1, coord2)
            if iou > non_max_suppression_threshold:
                overlap_found = True
                break
        if not overlap_found:
            filtered_objects.append(object_)
            
    return filtered_objects   

def visualize(cropped_image, filtered_objects):
    image_with_detections = cropped_image.copy()
    for detection in filtered_objects:
        cv2.rectangle(
            image_with_detections,
            (detection["TOP_LEFT_X"], detection["TOP_LEFT_Y"]),
            (detection["BOTTOM_RIGHT_X"], detection["BOTTOM_RIGHT_Y"]),
            detection["COLOR"],2)
    plt.imshow(image_with_detections, cmap='gray')
    plt.show(block=True)

def get_checkboxes(cropped_image, TEMPLATES=CHECK_TEMPLATES, show=False):

    detections = checkbox_match(TEMPLATES, cropped_image)
    filtered_detection = non_max_suppression(detections)
    if show : 
        visualize(cropped_image, filtered_detection)
    
    return filtered_detection

if __name__ == "__main__":
    from ProcessPDF import PDF_to_images, binarized_image
    print("start")
    path = r"C:\Users\CF6P\Desktop\ELPV\Data\scan5.pdf"
    images = PDF_to_images(path)
    images = images[0:]
    res_dict_per_image = {}
    for i, image in enumerate(images,1):
        print(f"\nImage {i} is starting")
        processed_image = binarized_image(image)
        format, cropped_image, rect = crop_image_and_sort_format(processed_image, original_image=image, show=True)
        # plt.imsave("t.png", cropped_image, cmap="gray")
