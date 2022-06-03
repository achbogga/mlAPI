"""
Grid View PDDS for LUNA system using classifier
Uses NVIDIA triton inference server instead of iunuml2.
Uses FastAPI instead of rill graph
Depends on triton image client which communicates with the server via http
created on June 3, 2022
@author : aboggaram@iunu.com
"""
import itertools
import json
import logging
import os
import traceback
import urllib
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from io import BytesIO
from multiprocessing import Pool, cpu_count

import numpy as np
import psycopg2
from PIL import Image
from tqdm import tqdm

from triton.triton_image_client import classify_with_triton, get_slice_generator
from utils.cluster_utils import cluster_using_db_scan
from utils.io_utils import NpEncoder, save_voc_anns

log = logging.getLogger("pdds.grid_view_pdds")


def chunk(image_list, n):
    # loop over the list in n-sized chunks
    for i in range(0, len(image_list), n):
        # yield the current n-sized chunk to the calling function
        yield image_list[i : i + n]


def process_images(payload):
    """
     Multiprocess copy of the grid_view_pdds,
     takes in the payload per process and returns results
     The arguments are same as the parent fn

    Args:
        payload (dict): dict of the function kwargs

    Returns:
        images_info_list (dict): dict of the results per process
    """
    label_map = payload["label_map"]
    input_preprocessing_enum = payload["input_preprocessing_enum"]
    model_name = payload["model_name"]
    model_version = payload["model_version"]
    batch_size = payload["batch_size"]
    server_url = payload["server_url"]
    cluster_eps = payload["cluster_eps"]
    cluster_min_samples = payload["cluster_min_samples"]
    slice_width = payload["slice_width"]
    slice_height = payload["slice_height"]
    debug = payload["debug"]
    debug_folder = payload["debug_folder"]
    images = payload["input_paths"]
    run_clustering = payload["run_clustering"]

    # display the process ID for debugging
    log.info("[INFO] starting process {}".format(payload["id"]))
    images_info_list = []
    for image_entry in tqdm(images):
        image_info = {}
        image_info["id"] = image_entry[0]
        image_info["x"] = image_entry[1]
        image_info["y"] = image_entry[2]
        image_info["url"] = image_entry[3]
        image_info["name"] = os.path.basename(image_info["url"])
        images_path = debug_folder + "/VOCdevkit/VOC2007/JPEGImages"
        # If the image already exists in the local mahcine, don't download it
        if debug:
            if not os.path.exists(images_path):
                os.makedirs(images_path)
        debug_image_path = images_path + "/" + image_info["name"]
        if not os.path.exists(debug_image_path):
            try:
                with urllib.request.urlopen(image_info["url"]) as resp:
                    pil_image = Image.open(BytesIO(resp.read()))
            except urllib.error.URLError as url_e:
                logging.info(f"{url_e} Skipping image: {image_info['name']}")
                logging.error(traceback.format_exc())
                continue
            except Exception as exception:
                logging.info(f"{exception} Skipping image: {image_info['name']}")
                logging.error(traceback.format_exc())
                continue
        else:
            pil_image = Image.open(debug_image_path)

        # Slice it according to the slice width and height and yield a slice generator
        # Identify the slice centers by using the absolute co-ordinates
        # These slices are resized in the triton image client before passed into the model
        np_image = np.asarray(pil_image, dtype=np.uint8)
        slice_centers = []
        slice_bboxes = []
        for x in range(0, np_image.shape[0], slice_width):
            for y in range(0, np_image.shape[1], slice_height):
                slice_center_x = (x + slice_width) // 2
                slice_center_y = (y + slice_height) // 2
                slice_centers.append((slice_center_x, slice_center_y))
                slice_xmin = max(0, slice_center_x - (slice_width) // 2)
                slice_ymin = max(0, slice_center_y - (slice_height) // 2)
                slice_xmax = min(np_image.shape[0], slice_center_x + (slice_width) // 2)
                slice_ymax = min(
                    np_image.shape[1], slice_center_y + (slice_height) // 2
                )
                slice_bbox = [slice_xmin, slice_ymin, slice_xmax, slice_ymax]
                slice_bboxes.append(slice_bbox)
        image_info["slice_centers"] = slice_centers
        slices_generator = get_slice_generator(np_image, slice_width, slice_height)
        no_of_slices = len(slice_centers)

        # classify the slices to find out the unhealthy patches
        image_info["slice_predictions"] = classify_with_triton(
            model_name,
            slices_generator,
            no_of_slices,
            model_version=model_version,
            no_of_classes=len(label_map.keys()),
            batch_size=batch_size,
            server_url=server_url,
            scaling=input_preprocessing_enum,
            protocol="HTTP",
            verbose=False,
            async_flag=False,
            streaming=False,
        )
        slice_predictions = image_info["slice_predictions"]
        all_pred_labels = list(np.array(slice_predictions)[:, 0])
        all_classes_in_preds = list(set(all_pred_labels))
        if debug:
            # if debug is enabled, write the bbox overlays for unhealthy patches
            #  and store in voc format
            detections_dict = {
                v: [] for k, v in label_map.items() if v in all_classes_in_preds
            }
            for index in range(len(all_pred_labels)):
                pred_label = all_pred_labels[index]
                detections_dict[pred_label].append(slice_bboxes[index])
            anns_path = debug_folder + "/VOCdevkit/VOC2007/Annotations"
            overlay_folder = debug_folder + "/unhealthy_bbox_overlays"
            # Save VOC format images
            if not os.path.exists(anns_path):
                os.makedirs(anns_path)
            if not os.path.exists(overlay_folder):
                os.makedirs(overlay_folder)
            overlay_image_path = overlay_folder + "/" + image_info["name"]
            save_voc_anns(
                pil_image,
                debug_image_path,
                np_image.shape[0],
                np_image.shape[1],
                anns_path,
                detections_dict,
                overlay_image_path=overlay_image_path,
                overlay_class_name="pgf_daylight_unhealthy",
            )
        # Run clustering if requested
        if run_clustering:
            # Run clustering only if unhealthy patches are identified by the model
            run_clustering_on_image = False
            for cls in all_classes_in_preds:
                if cls == "pgf_daylight_unhealthy":
                    run_clustering_on_image = True
            if run_clustering_on_image:
                if debug:
                    unhealthy_clusters_path = debug_folder + "/unhealthy_clusters"
                    plot_image_path = (
                        unhealthy_clusters_path
                        + "/"
                        + image_info["name"].replace(".jpg", "_unhealthy_clusters.jpg")
                    )
                    if not os.path.exists(unhealthy_clusters_path):
                        os.makedirs(unhealthy_clusters_path)
                else:
                    plot_image_path = ""
                # store the unhealthy slice centers in the image info dict
                unhealhthy_slice_centers = [
                    slice_centers[i]
                    for i in range(len(slice_centers))
                    if slice_predictions[i][0] == "pgf_daylight_unhealthy"
                ]
                # Cluster unhealthy centers using db_scan and store the cluster centroids
                X = np.array(unhealhthy_slice_centers, dtype=np.int16)
                image_info["unhealthy_cluster_centers"] = cluster_using_db_scan(
                    X,
                    np_image,
                    cluster_eps=cluster_eps,
                    min_samples=cluster_min_samples,
                    debug=debug,
                    plot_image_path=plot_image_path,
                )
            else:
                image_info["unhealthy_cluster_centers"] = []
        if debug:
            debug_info_path = debug_folder + "/debug_info"
            if not os.path.exists(debug_info_path):
                os.makedirs(debug_info_path)
            output_json_file_path = (
                debug_info_path + "/" + image_info["name"].replace(".jpg", ".json")
            )
            with open(output_json_file_path, "w") as fp:
                json.dump(image_info, fp, indent=4, cls=NpEncoder)
        images_info_list.append(image_info)
    return images_info_list


def grid_view_pdds(
    space_id="75195",
    date_str="07-08-2021",
    label_map={
        0: "pgf_daylight_healthy",
        1: "pgf_daylight_unhealthy",
        2: "empty",
        3: "purple",
    },
    input_preprocessing_enum="CONVNEXT",
    model_name="convnext_onnx",
    model_version="",
    batch_size=32,
    server_url="localhost:8000",
    run_clustering=False,
    cluster_eps=1000,
    cluster_min_samples=5,
    slice_width=1024,
    slice_height=1024,
    debug=False,
    debug_folder="/temp/debug_images_output",
):
    """
    Full space Pest and Disease Detection System (PDDS) on Grid images
    -> given a space id, and a date_str in the format '%m-%d-%Y',
    returns a list of unhealthy clusters with resepctive confidence values in the full space
    respective to each grid image in the space

    Args:
        space_id (str): _description_. Defaults to "75195".
        date_str (str): _description_. Defaults to "07-08-2021".
        label_map (dict, optional): _description_. Defaults to { 0: "pgf_daylight_healthy", 1: "pgf_daylight_unhealthy", 2: "empty", 3: "purple", }.
        input_preprocessing_enum (str, optional): _description_. Defaults to "CONVNEXT".
        model_name (str, optional): _description_. Defaults to "convnext_onnx".
        model_version (str, optional): _description_. Defaults to "".
        batch_size (int, optional): _description_. Defaults to 32.
        server_url (str, optional): _description_. Defaults to "localhost:8000".
        run_clustering (bool, optional): _description_. Defaults to False.
        cluster_eps (int, optional): _description_. Defaults to 1000.
        cluster_min_samples (int, optional): _description_. Defaults to 5.
        slice_width (int, optional): _description_. Defaults to 1024.
        slice_height (int, optional): _description_. Defaults to 1024.
        debug (bool, optional): _description_. Defaults to False.
        debug_folder (str, optional): _description_. Defaults to "/temp/debug_images_output".

    Returns:
        _type_: _description_
    """

    # get the args copied into a dict
    function_params = locals()

    log.info(f"Starting full space {space_id} from {date_str} grid view PDDS")

    log.info(f"All classes in the model are {label_map.values()}")
    log.info(f"Batch size value set to {batch_size}")
    # Connect to the database and download grid view bob images
    try:
        conn = psycopg2.connect(
            "dbname='luna' \
            user='data_pipeline_dev' host='10.168.0.2'"
        )
    except Exception as e:
        logging.info(f"{e} Can't connect to the PSQL database")
        logging.error(traceback.format_exc())
        exit(0)

    if debug and not os.path.exists(debug_folder):
        os.makedirs(debug_folder)

    # Construct the ROI query
    date_time_str = date_str + " 00:00:00"
    date_time_obj = datetime.strptime(date_time_str, "%m-%d-%Y %H:%M:%S")
    next_date_time_obj = date_time_obj + timedelta(days=1)
    query = "select id, position_x, position_y, \
                gs_url from log_arc_image_stream where \
                space_id = (%s) and created_on > (%s) and created_on < (%s) \
                and passnum = 1;"
    # Create the cursor
    cur = conn.cursor()
    # Execute the query
    cur.execute(query, (space_id, date_time_obj, next_date_time_obj))
    images = cur.fetchall()

    # Get the number of available cpu cores
    procs = cpu_count()
    procIDs = list(range(0, procs))
    numImagesPerProc = len(images) / float(procs)
    numImagesPerProc = int(np.ceil(numImagesPerProc))
    # chunk the image paths into N (approximately) equal sets, one
    # set of image paths for each individual process
    chunkedPaths = list(chunk(images, numImagesPerProc))

    # initialize the list of payloads
    payloads = []
    # loop over the set chunked image paths
    for (i, imagePaths) in enumerate(chunkedPaths):
        # construct a dictionary of data for the payload, then add it
        # to the payloads list
        data = {"id": i, "input_paths": imagePaths}
        data.update(function_params)
        payloads.append(data)
    # construct and launch the processing pool
    log.info("[INFO] launching pool using {} processes...".format(procs))
    pool = Pool(processes=procs)
    results = pool.map(process_images, payloads)

    # close the pool and wait for all processes to finish
    log.info("[INFO] Combining the results...")
    images_info_list = itertools.chain(results)

    # close the pool and wait for all processes to finish
    log.info("[INFO] waiting for processes to finish...")
    pool.close()
    pool.join()
    log.info("[INFO] multiprocessing complete")

    # send out the data as a list of dicts corresponding to each bob image
    return images_info_list
