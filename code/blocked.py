import cv2
import numpy as np

def is_camera_blocked(frame, threshold=10):
    """
    Detects if the camera is blocked by checking the brightness variance of the frame.
    :param frame: The current frame from the video feed.
    :param threshold: Variance threshold to detect if the camera is blocked (lower values indicate blockage).
    :return: True if the camera is blocked, False otherwise.
    """
    # Convert the frame to grayscale (to simplify brightness checking)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate the variance of pixel intensities in the grayscale image
    variance = np.var(gray)

    # If the variance is below the threshold, we assume the camera is blocked
    if variance < threshold:
        return True
    return False


