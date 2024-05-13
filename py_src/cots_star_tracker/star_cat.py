import pathlib
import time

import numpy as np

import cots_star_tracker.ground as ground
import cots_star_tracker.cam_matrix as cam_matrix

from typing import Union, Optional, Callable, Tuple, Union

CameraParams = Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]


def create_catalog(
    cam_config_file: Union[str, pathlib.Path, CameraParams],
    save_dir: Union[str, pathlib.Path],
    b_thresh: float = 6.0,
    save_vals: bool = True,
    verbose: bool = False,
):
    """
    Create a trimmed star catalog that is optimized for a given camera calibration.

    Args:
        cam_config_file: The name of the camera config file or a tuple with camera matrix, image size and distortion coefficients
        save_dir: Directory to save the compiled data to
        b_thresh: Brightness threshold, recommend 5-6 to start with
        save_vals: Option to save values or just return them from function
        verbose: Print status.
    """
    # Excess rows to remove from starcat_file
    excess_rows = [53, 54]
    # column (0-indexing) containing the Hipparcos ID number
    index_col = 2

    starcat_file = ground.get_star_cat_file()

    if isinstance(cam_config_file, str):
        camera_matrix, cam_resolution, _ = cam_matrix.read_cam_json(cam_config_file)
    if isinstance(cam_config_file, pathlib.Path):
        camera_matrix, cam_resolution, _ = cam_matrix.read_cam_json(cam_config_file)
    elif isinstance(cam_config_file, tuple) and len(cam_config_file) == 3:
        camera_matrix, cam_resolution, _ = cam_config_file
    else:
        raise ValueError("Bad argument: cam_config_file_name")


    ncol, nrow = cam_resolution[:2]
    fov = cam_matrix.cam2fov(cam_matrix.cam_matrix_inv(camera_matrix), nrow, ncol)

    if verbose:
        print(
            f"Creating star pair catalog including stars up to mag {b_thresh}, saving to: {save_dir} ..."
        )
    start_time = time.time()
    ground.create_star_catalog(
        starcat_file=starcat_file,
        brightness_thresh=b_thresh,
        excess_rows=excess_rows,
        index_col=index_col,
        fov=fov,
        save_vals=save_vals,
        save_dir=save_dir,
    )
    if verbose:
        print(
            f"...catalog creation complete in {time.time() - start_time:.3f} seconds."
        )

