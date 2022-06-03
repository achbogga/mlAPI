"""
File IO utils
created on June 3, 2022
@author : aboggaram@iunu.com
"""

import json
import os

import numpy as np
from pascal_voc_writer import Writer
from PIL import ImageDraw


class NpEncoder(json.JSONEncoder):
    """
    Helper class to save np_arrays as json objects

    Parameters
    ---------
    np.ndarray obj

    Returns
    ------
    jsonEncoder obj
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def save_voc_anns(
    pil_image,
    output_image_path,
    image_width,
    image_height,
    output_annotation_folder_path,
    bboxes_dict,
    overlay_image_path="/temp/overlay",
    overlay_class_name="pgf_daylight_unhealthy",
):
    """
    Function to save the bboxes in the PASCAL VOC 2007 format
    Overlays bboxes if needed

    Parameters
    ---------
    pil_image : Image
    output_image_path : str
    image_width : int
    image_height : int
    output_annotation_folder_path : str
    bboxes_dict : dict
    overlay_image_path : str
    overlay_class_name : str

    Returns
    ------
    None
    """
    if not os.path.exists(output_image_path):
        pil_image.save(output_image_path)
    writer = Writer(output_image_path, image_width, image_height)
    for class_name in bboxes_dict.keys():
        overlay_flag = class_name == overlay_class_name
        if overlay_flag:
            overlay_image = ImageDraw.Draw(pil_image)
        for bbox in bboxes_dict[class_name]:
            xmin, ymin, xmax, ymax = bbox
            if overlay_flag:
                overlay_image.rectangle(bbox, fill=None, outline="red", width=2)
            writer.addObject(class_name, xmin, ymin, xmax, ymax)
        if overlay_flag:
            pil_image.save(overlay_image_path)
    ann_name = output_image_path.split("/")[-1].replace("jpg", "xml")
    writer.save(output_annotation_folder_path + "/" + ann_name)
