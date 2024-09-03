import cv2
import numpy as np
import time

# Defina a captura de vídeo
cap = cv2.VideoCapture(0)  # 0 é geralmente a câmera padrão

# Variáveis para controle de gravação
IS_RECORDING = False
RECORDING_START_TIME = 0
VIDEO_WRITER = None

# Parâmetros para a gravacao
# Aqui você pode ajustar conforme necessário
MIN_CONTOUR_AREA = 15  # Área mínima do contorno para considerar como movimento
VIDEO_DURATION = 17 # Tempo de video que sera gravado apos dectectar o movimento

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Converter o frame para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Inicializar o primeiro frame
    if 'prev_frame' not in locals():
        prev_frame = gray
        continue

    # Calcular a diferença entre os frames
    frame_delta = cv2.absdiff(prev_frame, gray)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

    # Encontrar contornos no frame de threshold
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    MOVEMENT_DETECTED = False

    # Loop sobre os contornos encontrados
    for contour in contours:
        if cv2.contourArea(contour) > MIN_CONTOUR_AREA:
            MOVEMENT_DETECTED = True
            # break
                        
            # Calcular a caixa delimitadora mínima ao redor do contorno
            (x, y, w, h) = cv2.boundingRect(contour)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if MOVEMENT_DETECTED:
        if not IS_RECORDING:
            # Começar a gravar
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            video_filename = f"movement_{timestamp}.avi"
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            VIDEO_WRITER = cv2.VideoWriter(video_filename, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
            IS_RECORDING = True
            RECORDING_START_TIME = time.time()
            print(f"Movimento detectado. Gravando vídeo: {video_filename}")
        else:
            RECORDING_START_TIME = time.time()
            # print(f"Movimento detectado enquanto gravava.")


    if IS_RECORDING:
        # Gravar o frame no vídeo
        VIDEO_WRITER.write(frame)

        # Verificar se 15 segundos se passaram
        if time.time() - RECORDING_START_TIME > VIDEO_DURATION:
            IS_RECORDING = False
            VIDEO_WRITER.release()
            print(f"Gravação concluída: {video_filename}")

    # Mostrar o vídeo ao vivo
    cv2.imshow('Video', frame)

    # Atualizar o frame anterior
    prev_frame = gray

    # Fechar o vídeo ao pressionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if VIDEO_WRITER is not None:
    VIDEO_WRITER.release()
cv2.destroyAllWindows()
