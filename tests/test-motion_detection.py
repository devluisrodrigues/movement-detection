from motion_detection import *
import numpy as np

def test_draw_contours():
    # Create a dummy frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Create dummy contours
    contour1 = np.array([[100, 100], [200, 100], [200, 200], [100, 200]])
    contour2 = np.array([[300, 300], [400, 300], [400, 400], [300, 400]])
    contours = [contour1, contour2]

    # Call the draw_contours function
    movement_detected = draw_contours(frame, contours)

    # Check if movement is detected
    assert movement_detected == True

    print(frame[102, 102])
    # Check if contours are drawn on the frame
    for f in frame[100:200, 100]:
        assert np.array_equal(f, np.array([0, 255, 0], dtype=np.uint8)), f"Expected [0, 255, 0] but got {f}"
    for f in frame[100:200, 200]:
        assert np.array_equal(f, np.array([0, 255, 0], dtype=np.uint8)), f"Expected [0, 255, 0] but got {f}"
    for f in frame[100, 100:200]:
        assert np.array_equal(f, np.array([0, 255, 0], dtype=np.uint8)), f"Expected [0, 255, 0] but got {f}"
    for f in frame[200, 100:200]:
        assert np.array_equal(f, np.array([0, 255, 0], dtype=np.uint8)), f"Expected [0, 255, 0] but got {f}"