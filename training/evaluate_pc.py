import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from google.colab import drive, files

drive.mount('/content/drive')

IMG_SIZE = 224
SEUIL_CONFIANCE = 0.80
class_names = ['bouteille', 'canette', 'pain', 'pot_plastique']

# CHARGEMENT DU MODÈLE
print("Récupération du cerveau sauvegardé")
try:
    model = tf.keras.models.load_model("/content/drive/MyDrive/mon_modele_final.keras")
    print("Modèle chargé avec succès, Prêt pour le service.")
except Exception as e:
    print(f"Erreur : Je ne trouve pas le fichier ! Vérifie ton Drive ({e})")

# TA FONCTION DE TEST (Pour les graphes)
def test_image_avec_seuil(filename):
    try:
        img = tf.keras.utils.load_img(filename, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        predictions = model.predict(img_array, verbose=0)[0]
        score_max = np.max(predictions)
        index_gagnant = np.argmax(predictions)
        classe_gagnante = class_names[index_gagnant]

        if score_max > SEUIL_CONFIANCE:
            decision_texte = f"{classe_gagnante.upper()}\n(Confiance: {score_max:.2f})"
            couleur = 'green'
        else:
            decision_texte = f"ORGANIQUE / \n(Max: {score_max:.2f})"
            couleur = 'red'

        # Affichage
        plt.figure(figsize=(10, 3))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(decision_texte, color=couleur, fontweight='bold', fontsize=10)

        plt.subplot(1, 2, 2)
        couleurs = ['lightgray'] * len(class_names)
        couleurs[index_gagnant] = 'orange' if couleur == 'red' else 'green'
        barres = plt.bar(class_names, predictions, color=couleurs)
        plt.axhline(y=SEUIL_CONFIANCE, color='r', linestyle='--', label='Seuil')
        plt.ylim(0, 1.1)

        # Ajout des valeurs
        for rect in barres:
            height = rect.get_height()
            if height > 0.01:
                plt.text(rect.get_x() + rect.get_width()/2.0, height, f'{height:.2f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.show()
        print("-" * 40)
    except Exception as e:
        print(f"Erreur image {filename}: {e}")

# 4. LANCEMENT DU TEST
print("Cliquez sur le bouton ci-dessous pour tester les images :")
uploaded = files.upload()
for nom in uploaded.keys():
    test_image_avec_seuil(nom)
