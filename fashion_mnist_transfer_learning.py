# a. Set up environment  (pip install tensorflow)
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

D = "/Users/aayushagarwal/Downloads/LP IV Updated Datasets/Object Detection/"
DATA = D + "fashion mnist"
WEIGHTS = D + "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"  # local pre-trained weights
classes = ["T-shirt-top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle-boot"]  # skips the stray "train" folder

# b. Load and preprocess data (one folder per class; 28x28 grayscale -> 64x64 RGB)
args = dict(class_names=classes, color_mode="rgb", validation_split=0.2, seed=42,
            image_size=(64, 64), batch_size=64)
train = tf.keras.utils.image_dataset_from_directory(DATA, subset="training", **args)
val = tf.keras.utils.image_dataset_from_directory(DATA, subset="validation", **args)
prep = lambda x, y: (preprocess_input(x), y)
train, val = train.map(prep).prefetch(2), val.map(prep).prefetch(2)

# c. Pre-trained model
base = VGG16(weights=WEIGHTS, include_top=False, input_shape=(64, 64, 3))

# d. Freeze base layers, add custom layers
base.trainable = False
model = models.Sequential([base, layers.GlobalAveragePooling2D(),
                           layers.Dense(256, activation="relu"), layers.Dropout(0.5),
                           layers.Dense(len(classes), activation="softmax")])

# e/f. Train; loss = sparse categorical cross-entropy
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train, validation_data=val, epochs=5)

# g. Evaluate
print("Evaluation  [loss, acc]:", model.evaluate(val))
