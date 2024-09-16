import pytest
import numpy as np
import cv2
from unittest.mock import patch, MagicMock
from motion_detection import initialize_video_capture, process_frame, detect_movement

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
