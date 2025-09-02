import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


train_data = ImageDataGenerator(rescale = 1./255 )
val_data = ImageDataGenerator(rescale = 1./255)
test_data = ImageDataGenerator(rescale = 1./255)

# Loading images from directories
train_gen = train_data.flow_from_directory('data\training',
                                            target_size=(128,128), batch_size = 32, # Target size must match CNN input size
                                              class_mode = "categorical", shuffle = True)

val_gen = val_data.flow_from_directory('data\validation', 
                                       target_size=(128,128), batch_size = 32,
                                         class_mode = "categorical")

test_gen = test_data.flow_from_directory('data\testing', 
                                         target_size=(128,128), batch_size = 32,
                                           class_mode = "categorical")