import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
from sonolência_tempo import DetectorSonolenciaInteligente

# Inicialização do MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [263, 387, 385, 362, 380, 373]

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

EAR_THRESHOLD = 0.20
EAR_EYES_OPEN = 0.25
WINDOW_SIZE = 60
MIN_CLOSURE_DURATION = 10

frame_buffer = deque(maxlen=WINDOW_SIZE)
current_closure_duration = 0
is_eyes_closed = False

COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_BLUE = (255, 0, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_CYAN = (255, 255, 0)

# Detector de sonolência inteligente
detector = DetectorSonolenciaInteligente()

cap = cv2.VideoCapture(0)
print("Pressione 'q' para sair.")
print("PERCLOS REAL + Detector de tempo de piscada")

start_time = time.time()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    h, w, _ = frame.shape
    current_ear = 0.0
    eyes_status = "Detectando..."
    closure_type = "Aberto"
    perclos_real = 0.0

    timestamp = time.time()  # Timestamp para detecção de piscada

    if results.multi_face_landmarks:
        mesh = results.multi_face_landmarks[0]
        
        right_eye = np.array([[mesh.landmark[i].x * w, mesh.landmark[i].y * h] for i in RIGHT_EYE])
        left_eye = np.array([[mesh.landmark[i].x * w, mesh.landmark[i].y * h] for i in LEFT_EYE])
        
        right_ear = eye_aspect_ratio(right_eye)
        left_ear = eye_aspect_ratio(left_eye)
        current_ear = (right_ear + left_ear) / 2.0

        for point in right_eye:
            x, y = point.astype(int)
            cv2.circle(frame, (x, y), 2, COLOR_GREEN, -1)
        for point in left_eye:
            x, y = point.astype(int)
            cv2.circle(frame, (x, y), 2, COLOR_GREEN, -1)

        # 🎯 PERCLOS REAL
        if current_ear < EAR_THRESHOLD:
            current_closure_duration += 1
            eyes_status = f"FECHADO {current_closure_duration}frames"
            if current_closure_duration >= MIN_CLOSURE_DURATION:
                closure_type = "FECHAMENTO LENTO"
                frame_buffer.append(1)
            else:
                closure_type = "PISCADA"
                frame_buffer.append(0)
        else:
            current_closure_duration = 0
            eyes_status = "ABERTO"
            closure_type = "Aberto"
            frame_buffer.append(0)

        if len(frame_buffer) == WINDOW_SIZE:
            perclos_real = (sum(frame_buffer) / WINDOW_SIZE) * 100

        # Integração com detector de sonolência temporal
        detector.detectar_piscada(current_ear, timestamp)

    else:
        eyes_status = "FACE NAO DETECTADA"
        closure_type = "Indeterminado"
        frame_buffer.append(0)

    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0

    # Interface visual
    cv2.putText(frame, f"Status: {eyes_status}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_GREEN, 2)
    cv2.putText(frame, f"Tipo: {closure_type}", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_YELLOW, 2)
    cv2.putText(frame, f"EAR: {current_ear:.3f}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_YELLOW, 2)
    cv2.putText(frame, f"PERCLOS REAL: {perclos_real:.1f}%", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_CYAN, 2)
    
    bar_width = 300
    bar_height = 20
    perclos_bar = int((perclos_real / 100) * bar_width)
    cv2.rectangle(frame, (30, 160), (30 + bar_width, 160 + bar_height), COLOR_WHITE, 1)
    cv2.rectangle(frame, (30, 160), (30 + perclos_bar, 160 + bar_height), COLOR_CYAN, -1)
    
    # Alertas do PERCLOS
    if perclos_real > 15:
        cv2.putText(frame, "ALERTA: Sonolencia Leve", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_YELLOW, 2)
    if perclos_real > 30:
        cv2.putText(frame, "ALERTA: SONOLENCIA GRAVE!", (30, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_RED, 2)

    # Alertas do detector temporal
    if detector.alertas_ativos:
        cv2.putText(frame, detector.mensagem_alerta, (30, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)
        detector.resetar_alerta()

    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 150, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)
    cv2.putText(frame, f"Buffer: {len(frame_buffer)}/{WINDOW_SIZE}", (w - 200, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)
    cv2.putText(frame, f"Fechamento: {current_closure_duration} frames", (w - 250, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)

    cv2.imshow('PERCLOS REAL - Detecção de Sonolencia', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()