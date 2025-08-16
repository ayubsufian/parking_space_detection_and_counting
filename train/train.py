import os
from tensorflow.keras import applications  
from tensorflow.keras.preprocessing.image import ImageDataGenerator  
from tensorflow.keras import optimizers  
from tensorflow.keras.models import Model  
from tensorflow.keras.layers import Flatten, Dense

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

files_train = 0
files_validation = 0
cwd = os.getcwd()

# train data
folder = "data/train"

for sub_folder in os.listdir(folder):
    path, dirs, files = next(os.walk(os.path.join(folder, sub_folder)))
    files_train += len(files)

# test data
folder = "data/test"

for sub_folder in os.listdir(folder):
    path, dirs, files = next(os.walk(os.path.join(folder, sub_folder)))
    files_validation += len(files)


## Set Key Parameters

img_width, img_height = 48, 48
train_data_dir = "data/train"
validation_data_dir = "data/test"
nb_train_sample = files_train
nb_validation_sample = files_validation
batch_size = 32
epochs = 15
num_classes = 2

## build the CNN-VGG Model
model = applications.VGG16(
    weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3)
)

# Freeze the layers except the last 4 layers
for layer in model.layers[:10]:
    layer.trainable = False

x = model.output
x = Flatten()(x)
predictions = Dense(num_classes, activation="softmax")(x)
model_final = Model(inputs=model.input, outputs=predictions)

model_final.compile(
    loss="categorical_crossentropy",
    optimizer=optimizers.SGD(learning_rate=0.001, momentum=0.9),
    metrics=["accuracy"],
)

## Data Augmentation

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    horizontal_flip=True,
    fill_mode="nearest",
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
)

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    horizontal_flip=True,
    fill_mode="nearest",
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",
)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",
)

## fit the model
history = model_final.fit(
    train_generator,
    steps_per_epoch=nb_train_sample // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_sample // batch_size,
)

# Save model
folder_name = "model"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

model_file_path = os.path.join(folder_name, "model_final.h5")
model_final.save(model_file_path)
