import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from motion_detection import *
import numpy as np
from unittest.mock import patch, MagicMock
from motion_detection import initialize_video_capture, process_frame, detect_movement

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

def test_initialize_video_capture():
    with patch('cv2.VideoCapture') as MockCapture:
        mock_capture = MagicMock()
        MockCapture.return_value = mock_capture
        
        cap = initialize_video_capture()
        
        MockCapture.assert_called_once_with(0)
        assert cap == mock_capture

def test_process_frame():
    # cria um frame de teste (uma imagem RGB de 100x100 pixels) e processa
    frame = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    processed_frame = process_frame(frame)

    # verifica o tipo e a forma do frame processado e se ele está em escala de cinza
    assert isinstance(processed_frame, np.ndarray)
    assert processed_frame.shape == (100, 100)
    assert len(processed_frame.shape) == 2

def test_detect_movement():
    # cria dois frames de teste (imagens em escala de cinza de 100x100 pixels) e
    # adiciona condição para ser verificada
    prev_frame = np.zeros((100, 100), dtype=np.uint8)
    gray_frame = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(gray_frame, (20, 20), (40, 40), 255, -1)
    
    # testa detecção de movimento e verifica se houveram contornos e se estão no formato correto
    contours = detect_movement(prev_frame, gray_frame)

    assert len(contours) > 0
    assert isinstance(contours[0], np.ndarray)
