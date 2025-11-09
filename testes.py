import cv2  # opencv
# importa a bibliotéca mediapipe e "da o nome" de mp pra ela dentro do código
import mediapipe as mp

# inicializar opencv
webcam = cv2.VideoCapture(0)
# ele precisa ser conectado a uma câmera

# inicializar mediapipe
solucao_rostos = mp.solutions.face_detection
reconhece_rosto = solucao_rostos.FaceDetection()
desenho = mp.solutions.drawing_utils

while True:
    # ler infos da webcam
    # primeira info 'frame' só verifica se conseguiu ler ou não
    imagem, frame = webcam.read()
    # a segunda, se conseguiu ler, é onde fica armazenada a informação

    if not imagem:
        break

    # reconhecer rostos dentro da imagem
    # processa frame e ve se há algum rosto
    nrostos = reconhece_rosto.process(frame)

    if nrostos.detections:  # testa se há algum rosto na imagem
        for rosto in nrostos.detections:
            # fazer o desenho ao redor dos rostos
            desenho.draw_detection(frame, rosto)

    cv2.imshow("Rosto na câmera", frame)

    # parar o loop com alguma tecla
    if cv2.waitKey(10) == 27:
        # waitkey é uma funçao qe espera tu apertar uma tecla do teclado
        # nesse caso é o esc. Se não souber o código da tecla pode usar ord('tecla')
        break

webcam.release()
cv2.destroyAllWindows()
