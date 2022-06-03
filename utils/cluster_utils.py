"""
DB Scan Cluster utils
created on May, 2022
@author : aboggaram@iunu.com
"""

import math

import numpy as np
from PIL import Image, ImageDraw
from sklearn.cluster import DBSCAN


def centeroidnp(arr):
    """
    Computes the cluster centers given all the points
    belonging to a cluster of same labels

    Parameters
    ---------
    array of points belonging to a certain cluster
    Returns
    ------
    cluster centroid

    """
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return sum_x / length, sum_y / length


def cluster_using_db_scan(
    X,
    np_image,
    cluster_eps=256,
    min_samples=8,
    debug=True,
    plot_image_path="unhealthy_cluster_visualization.jpg",
    r=50,
):
    """
    Estimates the number of clusters given a
    bunch of points in the 2D space. Uses DB Scan
    clustering algorithm. Takes in numpy image and
    draws cluster centers as circle given a display radius r

    Parameters
    ---------
    X : np.array
    np_image : np.array
    cluster_eps : float
    min_samples : int
    debug : bool
    plot_image_path : str
    r : int
    Returns
    ------
    cluster centrs : list

    """
    # Compute DBSCAN
    db = DBSCAN(eps=cluster_eps, min_samples=min_samples).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    if debug and n_clusters_:
        print("Estimated number of clusters: %d" % n_clusters_)
        print("Estimated number of noise points: %d" % n_noise_)
        pil_image = Image.fromarray(np_image)
    cluster_centers = []
    unique_labels = set(labels)
    for k in unique_labels:
        if k != -1:
            class_member_mask = labels == k
            xy = X[class_member_mask & core_samples_mask]
            cluster_center = centeroidnp(xy)
            if not math.isnan(cluster_center[0]) and not math.isnan(cluster_center[1]):
                cluster_center = [int(cluster_center[0]), int(cluster_center[1])]
                # If debug, overlay cluster centroids
                if debug and n_clusters_:
                    draw = ImageDraw.Draw(pil_image)
                    draw.ellipse(
                        (
                            cluster_center[0] - r,
                            cluster_center[1] - r,
                            cluster_center[0] + r,
                            cluster_center[1] + r,
                        ),
                        fill=(255, 0, 0, 0),
                    )
                cluster_centers.append(cluster_center)
    if debug and n_clusters_:
        pil_image.save(plot_image_path)
    return cluster_centers
