from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import model_from_json


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
import tensorflow as tf

class FaceShapeTrainer():
    def __init__(self, model_name, batch_size, num_of_epoch) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_of_epoch = num_of_epoch

    def train(self):
        # All images will be rescaled by 1./255
        train_datagen = ImageDataGenerator(rescale=1/255)

        # Flow training images in batches of 128 using train_datagen generator
        train_generator = train_datagen.flow_from_directory(
                'dataset',  # This is the source directory for training images
                target_size=(200, 200),  # All images will be resized to 200 x 200
                batch_size=self.batch_size,
                # Specify the classes explicitly
                classes = ['heart','oblong','oval','round', 'square'],
                # Since we use categorical_crossentropy loss, we need categorical labels
                class_mode='categorical')


        model = tf.keras.models.Sequential([
            # Note the input shape is the desired size of the image 200x 200 with 3 bytes color
            # The first convolution
            tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(200, 200, 3)),
            tf.keras.layers.MaxPooling2D(2, 2),
            # The second convolution
            tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            # The third convolution
            tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            # The fourth convolution
            tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            # The fifth convolution
            tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            # The sixth convolution
            tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            # Flatten the results to feed into a dense layer
            tf.keras.layers.Flatten(),
            # 128 neuron in the fully-connected layer
            tf.keras.layers.Dense(128, activation='relu'),
            # 5 output neurons for 5 classes with the softmax activation
            tf.keras.layers.Dense(5, activation='softmax')
        ])

        model.summary()


        model.compile(loss='categorical_crossentropy',
                    optimizer=RMSprop(lr=0.001),
                    metrics=['acc'])

        total_sample=train_generator.n

        history = model.fit_generator(
                train_generator, 
                steps_per_epoch=int(total_sample/self.batch_size),  
                epochs=self.num_of_epoch,
                verbose=1)

        model.save(self.model_name)
