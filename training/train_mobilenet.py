import tensorflow as tf
from tensorflow.keras import layers, models
from google.colab import drive

# 1. Montage du disque
drive.mount('/content/drive')

# --- PARAMÈTRES HYPER-HEURISTIQUES ---
#le DATASETS était dans mon drive et j'ai pas mis dans mon github
CHEMIN_BASE_DONNEES = "/content/drive/MyDrive/DATASET"
TAILLE_IMAGE_ENTREE = 224  # Imposé par l'architecture MobileNet
TAILLE_LOT = 32  # Batch Size : Compromis mémoire/stabilité du gradient

# 2. Ségrégation des données (80% Apprentissage / 20% Validation)
print("Structuration des tenseurs de données...")
donnees_apprentissage = tf.keras.utils.image_dataset_from_directory(
    CHEMIN_BASE_DONNEES,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(TAILLE_IMAGE_ENTREE, TAILLE_IMAGE_ENTREE),
    batch_size=TAILLE_LOT
)

donnees_validation = tf.keras.utils.image_dataset_from_directory(
    CHEMIN_BASE_DONNEES,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(TAILLE_IMAGE_ENTREE, TAILLE_IMAGE_ENTREE),
    batch_size=TAILLE_LOT
)

categories_dechets = donnees_apprentissage.class_names
print(f"Classes identifiées : {categories_dechets}")

# 3. Architecture du Réseau (Transfer Learning)
# On importe le réseau pré-entraîné (Extracteur de caractéristiques)
extracteur_caracteristiques = tf.keras.applications.MobileNetV2(
    input_shape=(TAILLE_IMAGE_ENTREE, TAILLE_IMAGE_ENTREE, 3),
    include_top=False,  # On enlève la tête de classification d'origine
    weights='imagenet'
)
extracteur_caracteristiques.trainable = False  # Gel des poids synaptiques

# Construction du modèle séquentiel final
modele_tri_dechets = models.Sequential([
    # Couche de normalisation mathématique [-1, 1]
    layers.Rescaling(1./127.5, offset=-1, input_shape=(TAILLE_IMAGE_ENTREE, TAILLE_IMAGE_ENTREE, 3)),

    extracteur_caracteristiques,

    # Couche de réduction de dimension (Aplatissement)
    layers.GlobalAveragePooling2D(),

    # Couche de régularisation (Prévention du surapprentissage)
    layers.Dropout(0.2),

    # Couche de décision finale (Autant de neurones que de types de déchets)
    layers.Dense(len(categories_dechets), activation='softmax')
])

# Paramétrage de l'algorithme d'optimisation
modele_tri_dechets.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 4. Phase d'entraînement
print("Début de l'optimisation des poids...")
historique = modele_tri_dechets.fit(
    donnees_apprentissage,
    validation_data=donnees_validation,
    epochs=10
)

# 5. Quantification et Export pour système embarqué (Raspberry Pi)
convertisseur = tf.lite.TFLiteConverter.from_keras_model(modele_tri_dechets)
modele_embarque_tflite = convertisseur.convert()

with open('modele_tri_optimise.tflite', 'wb') as fichier:
    fichier.write(modele_embarque_tflite)

print("Fichier .tflite embarqué généré !")
