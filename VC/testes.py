# Importando as bibliotecas necessárias
import cv2                      # Para captura e exibição de vídeo
import mediapipe as mp          # Para detecção de mãos

# Inicializando os módulos do MediaPipe
mp_hands = mp.solutions.hands                          # Módulo de mãos
mp_drawing = mp.solutions.drawing_utils                # Para desenhar os pontos na tela
mp_styles = mp.solutions.drawing_styles                # Estilos padrões do MediaPipe

# Criando o objeto de detecção de mãos
# Usamos o 'with' para garantir que o objeto será corretamente encerrado
with mp_hands.Hands(
    static_image_mode=False,               # False: para vídeo contínuo
    max_num_hands=2,                       # Detectar até 2 mãos
    min_detection_confidence=0.5,          # Confiança mínima para detectar
    min_tracking_confidence=0.5            # Confiança mínima para rastrear
) as hands:

    # Inicializando a captura de vídeo (webcam padrão)
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        # Captura um frame da webcam
        success, frame = cap.read()
        if not success:
            print("Não foi possível ler o frame da câmera.")
            break

        # Espelha o frame (opcional para uma visualização mais natural)
        frame = cv2.flip(frame, 1)

        # Converte a imagem de BGR (OpenCV) para RGB (MediaPipe exige isso)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Desativa gravação da imagem (para ganhar performance)
        frame_rgb.flags.writeable = False

        # Processa a imagem para detectar as mãos
        results = hands.process(frame_rgb)

        # Reativa gravação da imagem
        frame_rgb.flags.writeable = True

        # Se alguma mão for detectada...
        if results.multi_hand_landmarks:
            # Para cada mão detectada
            for hand_landmarks in results.multi_hand_landmarks:

                # Desenha os pontos (landmarks) e conexões da mão
                mp_drawing.draw_landmarks(
                    frame,                              # Imagem de destino
                    hand_landmarks,                     # Posição dos pontos da mão
                    mp_hands.HAND_CONNECTIONS,          # Conexões entre os pontos
                    mp_styles.get_default_hand_landmarks_style(),       # Estilo dos pontos
                    mp_styles.get_default_hand_connections_style()      # Estilo das conexões
                )

                # Mostrando as coordenadas dos landmarks no terminal
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    h, w, _ = frame.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    print(f"Landmark {idx}: x={cx}, y={cy}")

        # Mostra o resultado em uma janela
        cv2.imshow('MediaPipe - Mãos', frame)

        # Sai do loop se pressionar a tecla ESC (código 27)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Libera os recursos (importante)
    cap.release()
    cv2.destroyAllWindows()
