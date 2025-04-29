import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Pontos dos lábios superiores e inferiores (referência: face mesh da Mediapipe)
UPPER_LIP_IDX = 13
LOWER_LIP_IDX = 14

# Inicializa captura de vídeo
cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                ih, iw, _ = frame.shape

                # Obtém coordenadas dos lábios
                upper_lip = face_landmarks.landmark[UPPER_LIP_IDX]
                lower_lip = face_landmarks.landmark[LOWER_LIP_IDX]

                y1 = int(upper_lip.y * ih)
                y2 = int(lower_lip.y * ih)

                lip_distance = abs(y2 - y1)

                # Desenhar pontos do rosto (opcional)
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
                )

                # Heurística simples: se a boca está mais aberta que um limiar, considera sorriso
                if lip_distance > 15:
                    cv2.putText(frame, "Feliz! :D", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                else:
                    cv2.putText(frame, "Neutro", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        cv2.imshow("Detector de Sorriso - MediaPipe", frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
