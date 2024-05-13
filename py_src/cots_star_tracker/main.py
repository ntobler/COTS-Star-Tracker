"""
main.py

This script is intended to contain the primary function(s)
required to get an attitude estimate from an image
of stars
"""
import numpy as np
import cv2

import cots_star_tracker.rpi_core as rpi_core
import cots_star_tracker.cam_matrix as cam
import cots_star_tracker.support_functions as support_functions
import cots_star_tracker.array_transformations as xforms

from typing import Optional, Tuple, Union

CameraParams = Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]


# @support_functions.timing_decorator
def star_tracker(
    img: np.ndarray,
    cam_config_file_name: Union[str, CameraParams],
    m=None,
    q=None,
    x_cat=None,
    k=None,
    indexed_star_pairs=None,
    darkframe: Optional[np.ndarray] = None,
    undist_img_bool=True,
    n_stars=30,
    low_thresh_pxl_intensity=None,
    hi_thresh_pxl_intensity=None,
    min_star_area=4,
    max_star_area=36,
    isa_thresh=0.0008,
    nmatch=6,
    watchdog=10.,
    graphics=False,
    verbose=False,
):
    assert img.ndim == 2
    assert img.dtype == np.uint8

    if isinstance(cam_config_file_name, str):
        camera_matrix, cam_resolution, dist_coefs = cam.read_cam_json(cam_config_file_name)
    elif isinstance(cam_config_file_name, tuple) and len(cam_config_file_name) == 3:
        camera_matrix, cam_resolution, dist_coefs = cam_config_file_name
    else:
        raise ValueError("Bad argument: cam_config_file_name")

    dist_coefs = np.array(dist_coefs)
    im_resolution = np.array([img.shape[1], img.shape[0]], dtype=int)  # pixels

    assert np.all(im_resolution == cam_resolution)

    if darkframe is not None:
        assert img.ndim == darkframe.ndim
        assert img.dtype == darkframe.dtype
        img = cv2.subtract(img, darkframe)  # also clips the image

    # # undistort points #TODO determine location of this vs CV2 and the functionality difference
    # if undist_img_bool is True:
    #     img = rpi_core.undistort_image(
    #         img, idxUndistorted, idxDistorted, interpWeights)

    if min_star_area >= max_star_area:
        raise ValueError("min_star_area must be less than max_star_area")

    centroids, intensities = support_functions.find_candidate_stars(
        img,
        min_star_area=min_star_area,
        max_star_area=max_star_area,
        low_thresh=low_thresh_pxl_intensity,
        hi_thresh=hi_thresh_pxl_intensity,
        graphics=graphics,
    )

    # if fewer than 3 centroids are found, don't bother.  Gums things up downstream.
    if len(centroids) < 3:
        raise rpi_core.StartrackerError("Found too few star candidates (< 3) to continue.")

    if graphics:
        import matplotlib.pyplot as plt

        fig = plt.figure()
        plt.imshow(img, cmap="Greys_r")
        plt.plot(centroids[..., 0], centroids[..., 1], "x")
        plt.title("Greyscale image with all centroids")
        fig.tight_layout()
        plt.show()

    intensities = intensities.squeeze()
    n_stars = min(len(intensities), n_stars)
    bright_star_idx = intensities.argsort()[-n_stars:]
    centroids = centroids[bright_star_idx, :]
    if verbose:
        print("Using " + str(len(centroids)) + " centroids")

    if graphics:
        import matplotlib.pyplot as plt

        fig = plt.figure()
        plt.imshow(img, cmap="Greys_r")
        plt.plot(centroids[..., 0], centroids[..., 1], "x")
        plt.title("Greyscale image with selected centroids")
        fig.tight_layout()
        plt.show()

    # Correct centroids with lens distortion
    if undist_img_bool is True:
        centroids = cv2.undistortPoints(
            centroids[:, None], camera_matrix, dist_coefs, P=camera_matrix
        )[:, 0]

    # Corrected centroids in unit vectors
    cinv = cam.cam_matrix_inv(camera_matrix)
    x_obs = xforms.pixel2vector(cinv, centroids.T)
    if verbose:
        print(f"Number of star observations: {x_obs.shape[1]}")

    if np.isnan(x_obs).any():
        raise rpi_core.StartrackerError("Failure to calculate x_obs")

    if verbose:
        print("Starting star ID")

    k_vector_interp = (m, q)

    q_est, idmatch, nmatches = rpi_core.triangle_isa_id(
        x_obs,
        x_cat,
        indexed_star_pairs,
        isa_thresh,
        nmatch,
        k,
        k_vector_interp,
        watchdog=watchdog,
        verbose=verbose,
    )

    if graphics and nmatches > 0:
        support_functions.reproject(
            img, camera_matrix, idmatch, q_est, x_obs, x_cat
        )
        # matched candidates: pixel position of the blob that we think is a star
        # invalid candidates: pixel position of a blob we found, but that we don't think are stars
        # matched catalog stars: this is where the star should be assuming the quat is true
        # unmatched catalog stars: this is where a catalog star should be, but we couldn't find it (or maybe it wasn't bright enough to be considered)

    # Return x_obs and the img
    return q_est, idmatch, nmatches, x_obs, img
