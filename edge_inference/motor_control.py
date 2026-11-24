import RPi.GPIO as GPIO
import time

BROCHE_SERVO = 18
GPIO.setmode(GPIO.BCM)
GPIO.setup(BROCHE_SERVO, GPIO.OUT)

# Initialisation du signal PWM à 50 Hz (Standard pour le S3003)
pwm_bras = GPIO.PWM(BROCHE_SERVO, 50)
pwm_bras.start(0)  # Démarre avec un rapport cyclique de 0 (moteur au repos)

# --- 2. FONCTION DE CONVERSION (Angle vers Rapport Cyclique) ---
def angle_vers_rapport_cyclique(angle):
    """0° correspond à une impulsion de ~1ms (Rapport cyclique de 2.5%)
    et 180° correspond à une impulsion de ~2ms (Rapport cyclique de 12.5%)"""
    rapport_cyclique = 2.5 + (angle / 180.0) * 10.0
    return rapport_cyclique

def trier_dechet(classe_detectee):
    # Oriente le bras du servomoteur en fonction de la classe du déchet.
    if classe_detectee == "organique" or classe_detectee == "inconnu":
        # Position initiale (laisse le déchet continuer tout droit)
        angle = 0
    elif classe_detectee == "canette":
        # Orientation vers le bac n°2
        angle = 45
    elif classe_detectee == "bouteille":
        # Orientation vers le bac n°1
        angle = 135
    else:
        angle = 0

    # Calcul et envoi du signal physique
    rapport = angle_vers_rapport_cyclique(angle)
    pwm_bras.ChangeDutyCycle(rapport)
    # Délai pour attendre le déplacement du bras
    time.sleep(0.5)
    # On coupe le signal pour éviter que le servomoteur "tremble" et chauffe
    pwm_bras.ChangeDutyCycle(0)

# EXEMPLE D'APPEL qu'on peut faire :
# quand l'IA a validé 5 frames :
# trier_dechet("canette")
