import cv2
import mediapipe as mp
import numpy as np

# Inicialização do MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Pontos melhorados para detecção de olhos
RIGHT_EYE_IMPROVED = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]
LEFT_EYE_IMPROVED = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# Pontos originais (6 pontos) para comparação
RIGHT_EYE_SIMPLE = [33, 160, 158, 133, 153, 144]
LEFT_EYE_SIMPLE = [263, 387, 385, 362, 380, 373]

def draw_enhanced_eyes(frame, mesh, h, w):
    """Desenha os olhos com contornos completos"""
    
    # Olho direito - pontos melhorados
    right_eye_points = []
    for i in RIGHT_EYE_IMPROVED:
        x = int(mesh.landmark[i].x * w)  # Converte para pixels
        y = int(mesh.landmark[i].y * h)  # Converte para pixels
        right_eye_points.append([x, y])
        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)  # Pontos verdes
    
    # Conecta os pontos do olho direito (contorno azul)
    right_eye_np = np.array(right_eye_points, np.int32)
    cv2.polylines(frame, [right_eye_np], True, (255, 0, 0), 1)
    
    # Olho esquerdo - pontos melhorados
    left_eye_points = []
    for i in LEFT_EYE_IMPROVED:
        x = int(mesh.landmark[i].x * w)  # Converte para pixels
        y = int(mesh.landmark[i].y * h)  # Converte para pixels
        left_eye_points.append([x, y])
        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)  # Pontos verdes
    
    # Conecta os pontos do olho esquerdo (contorno azul)
    left_eye_np = np.array(left_eye_points, np.int32)
    cv2.polylines(frame, [left_eye_np], True, (255, 0, 0), 1)
    
    # Desenha pontos originais (6 pontos) em vermelho para comparação
    for i in RIGHT_EYE_SIMPLE:
        x = int(mesh.landmark[i].x * w)
        y = int(mesh.landmark[i].y * h)
        cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)  # Pontos vermelhos
    
    for i in LEFT_EYE_SIMPLE:
        x = int(mesh.landmark[i].x * w)
        y = int(mesh.landmark[i].y * h)
        cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)  # Pontos vermelhos

def eye_aspect_ratio(eye):
    """Calcula o Eye Aspect Ratio (EAR)"""
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Configurações
EAR_THRESHOLD = 0.20

cap = cv2.VideoCapture(0)
print("Pressione 'q' para sair.")
print("VERDE: Pontos melhorados (16 pontos)")
print("VERMELHO: Pontos originais (6 pontos)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip horizontal para efeito espelho
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    # Obtém altura (h) e largura (w) do frame
    h, w, _ = frame.shape

    if results.multi_face_landmarks:
        mesh = results.multi_face_landmarks[0]
        
        # Chama a função para desenhar os olhos melhorados
        draw_enhanced_eyes(frame, mesh, h, w)
        
        # Calcula EAR com pontos originais (para manter consistência)
        right_eye = np.array([[mesh.landmark[i].x * w, mesh.landmark[i].y * h] for i in RIGHT_EYE_SIMPLE])
        left_eye = np.array([[mesh.landmark[i].x * w, mesh.landmark[i].y * h] for i in LEFT_EYE_SIMPLE])
        
        right_ear = eye_aspect_ratio(right_eye)
        left_ear = eye_aspect_ratio(left_eye)
        ear = (right_ear + left_ear) / 2.0

        # Exibe status
        if ear < EAR_THRESHOLD:
            status = "OLHOS FECHADOS"
            color = (0, 0, 255)
        else:
            status = "Olhos abertos"
            color = (0, 255, 0)
        
        cv2.putText(frame, status, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"EAR: {ear:.3f}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Frame: {w}x{h}", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    else:
        cv2.putText(frame, "Face não detectada", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Visualização Melhorada dos Olhos - MediaPipe', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()