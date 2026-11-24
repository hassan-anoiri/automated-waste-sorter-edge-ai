import collections
from collections import deque
import time

# --- INITIALISATION ---
# Création d'une mémoire tampon (buffer) qui ne garde QUE les 5 derniers éléments
historique_predictions = deque(maxlen=5)

# --- DANS TA BOUCLE VIDÉO PRINCIPALE (while True:) ---
# label_gagnant = CLASSES[index_gagnant] (ex: "canette", "bouteille", "organique")
# 1. Ajout de la prédiction instantanée dans la mémoire glissante
historique_predictions.append(label_gagnant)

# 2. On attend que la mémoire soit pleine (5 images) pour prendre une décision
if len(historique_predictions) == 5:

    # On compte le nombre d'apparitions de chaque déchet dans les 5 cases
    compteur = collections.Counter(historique_predictions)

    # On extrait la classe la plus fréquente et son nombre d'apparitions
    classe_majoritaire = compteur.most_common(1)[0][0]
    nombre_apparitions = compteur.most_common(1)[0][1]

    # 3. LE FILTRE STRICT : Au moins 4 images sur 5
    if nombre_apparitions >= 4:

        # On ignore le "vide" ou "organique" (le tapis tourne à vide)
        if classe_majoritaire != "inconnu" and classe_majoritaire != "organique":

            print(f"OBJET CONFIRMÉ : {classe_majoritaire.upper()} (Fiabilité temporelle : {nombre_apparitions}/5)")

            # --- DÉCLENCHEMENT DE L'ACTIONNEUR ---
            # Appel de la fonction du servomoteur (vue dans l'annexe précédente)
            trier_dechet(classe_majoritaire)
            # On vide complètement la mémoire après une action !
            historique_predictions.clear()
