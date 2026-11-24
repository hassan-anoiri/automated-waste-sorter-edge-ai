import time
import cv2
import numpy as np

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

MODEL_PATH = "model_tipe.tflite"
CLASSES = ['bouteille', 'canette', 'pain', 'pot_plastique']
SEUIL = 0.70

# INITIALISATION IA
print("Chargement du modèle...")
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = 224
width = 224

# INITIALISATION CAMERA
cap = cv2.VideoCapture(0)
print("Système prêt")

while True:
    # 1. Capture de l'image
    ret, frame = cap.read()
    if not ret:
        break

    # 2. Prétraitement pour l'IA (Redimensionner en 224x224)
    image_display = frame.copy()
    # L'IA veut du RGB (OpenCV donne du BGR)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (width, height))
    input_data = np.expand_dims(img_resized, axis=0)

    # NOTE IMPORTANTE : MobileNet a besoin de float32
    # La normalisation (-1,1) est déjà dans le modèle grâce à la
    # couche Rescaling de Colab !
    # Donc on envoie juste les pixels bruts mais en float.
    input_data = input_data.astype(np.float32)

    # 3. Inférence (Le cerveau réfléchit)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # output_data est une liste de probabilités ex: [[0.02, 0.95, 0.01, 0.02]]
    predictions = output_data[0]
    score_max = np.max(predictions)
    index_gagnant = np.argmax(predictions)
    label_gagnant = CLASSES[index_gagnant]

    # 4. Décision & Affichage
    if score_max > SEUIL:
        text = f"{label_gagnant.upper()} ({score_max:.2f})"
        color = (0, 255, 0)  # Vert
        # ICI ON AJOUTE LE CODE DE COMMANDE DU SERVO MOTEUR
    else:
        text = f"ORGANIQUE ({score_max:.2f})"
        color = (0, 0, 255)

    # Affichage sur l'écran du PC
    cv2.putText(image_display, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.rectangle(image_display, (10, 60), (int(score_max * 200) + 10, 80), color, -1)
    cv2.imshow('TIPE - Vision Artificielle', image_display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
