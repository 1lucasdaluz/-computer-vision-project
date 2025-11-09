import cv2
import mediapipe as mp
import numpy as np

# Inicialização do MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)

# Índices dos pontos dos olhos (MediaPipe Face Mesh)
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [263, 387, 385, 362, 380, 373]


def eye_aspect_ratio(eye):
    # EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


# Threshold para detectar olhos fechados
EAR_THRESHOLD = 0.20

cap = cv2.VideoCapture(0)  # iniciando opencv
print("Pressione 'q' para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        mesh = results.multi_face_landmarks[0]
        h, w, _ = frame.shape

        # Extrai os pontos dos olhos
        right_eye = np.array(
            [[mesh.landmark[i].x * w, mesh.landmark[i].y * h] for i in RIGHT_EYE])
        left_eye = np.array(
            [[mesh.landmark[i].x * w, mesh.landmark[i].y * h] for i in LEFT_EYE])

        right_ear = eye_aspect_ratio(right_eye)
        left_ear = eye_aspect_ratio(left_eye)
        ear = (right_ear + left_ear) / 2.0

        if ear < EAR_THRESHOLD:
            cv2.putText(frame, "Olhos fechados", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Olhos abertos", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(frame, f"EAR: {ear:.2f}", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    else:
        cv2.putText(frame, "Face não detectada", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Detecção de Olhos - MediaPipe', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

