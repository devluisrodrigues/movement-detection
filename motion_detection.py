"""
Este módulo implementa um sistema de detecção de movimento usando OpenCV.
Quando um movimento é detectado, o sistema grava um vídeo de 17 segundos.
"""

import time
import cv2

# Parâmetros para a gravação
MIN_CONTOUR_AREA = 15  # Área mínima do contorno para considerar como movimento
VIDEO_DURATION = 17  # Tempo de vídeo que será gravado após detectar o movimento

def initialize_video_capture():
    """
    Inicializa a captura de vídeo da câmera padrão.

    Returns:
        cv2.VideoCapture: Objeto de captura de vídeo.
    """
    return cv2.VideoCapture(0)  # 0 é geralmente a câmera padrão

def process_frame(frame):
    """
    Converte o frame para escala de cinza e aplica um desfoque gaussiano.

    Args:
        frame (numpy.ndarray): Frame de vídeo original.

    Returns:
        numpy.ndarray: Frame processado em escala de cinza e desfocado.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    return gray

def detect_movement(prev_frame, gray_frame):
    """
    Detecta movimento comparando o frame atual com o frame anterior.

    Args:
        prev_frame (numpy.ndarray): Frame anterior em escala de cinza.
        gray_frame (numpy.ndarray): Frame atual em escala de cinza.

    Returns:
        list: Lista de contornos detectados.
    """
    frame_delta = cv2.absdiff(prev_frame, gray_frame)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def draw_contours(frame, contours):
    """
    Desenha contornos no frame e verifica se há movimento.

    Args:
        frame (numpy.ndarray): Frame de vídeo original.
        contours (list): Lista de contornos detectados.

    Returns:
        bool: True se movimento for detectado, caso contrário False.
    """
    movement_detected = False
    for contour in contours:
        if cv2.contourArea(contour) > MIN_CONTOUR_AREA:
            movement_detected = True
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return movement_detected

def start_recording(cap):
    """
    Inicia a gravação de vídeo.

    Args:
        cap (cv2.VideoCapture): Objeto de captura de vídeo.

    Returns:
        tuple: Objeto de gravação de vídeo e nome do arquivo de vídeo.
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    video_filename = f"movement_{timestamp}.avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(
        video_filename, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
    print(f"Movimento detectado. Gravando vídeo: {video_filename}")
    return video_writer, video_filename

def main():
    """
    Função principal que executa o loop de detecção de movimento e gravação de vídeo.
    """
    cap = initialize_video_capture()
    is_recording = False
    recording_start_time = 0
    video_writer = None
    video_filename = None
    prev_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = process_frame(frame)

        if prev_frame is None:
            prev_frame = gray
            continue

        contours = detect_movement(prev_frame, gray)
        movement_detected = draw_contours(frame, contours)

        if movement_detected:
            if not is_recording:
                video_writer, video_filename = start_recording(cap)
                is_recording = True
                recording_start_time = time.time()
            else:
                recording_start_time = time.time()

        if is_recording:
            video_writer.write(frame)
            if time.time() - recording_start_time > VIDEO_DURATION:
                is_recording = False
                video_writer.release()
                print(f"Gravação concluída: {video_filename}")

        cv2.imshow('Video', frame)
        prev_frame = gray

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
