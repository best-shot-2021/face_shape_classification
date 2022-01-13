import tensorflow as tf 
import numpy as np
from keras.preprocessing import image

class FaceShapeEvaluator():
    def __init__(self, model_path, test_data) -> None:
        self.model_path = model_path
        self.test_data = test_data
        self.model = tf.keras.models.load_model(self.model_path)

    def evluate(self):
        test_image = image.load_img(self.test_data, target_size = (200,200))
        #test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = self.model.predict(test_image)

        final_result=""

        if result[0][1] == 1:
            final_result = "heart"
        elif result[0][0] == 1:
            final_result = "oblong"
        elif result[0][2] == 1:
            final_result = "oval"
        elif result[0][3] == 1:
            final_result = "round"
        elif result[0][4] == 1:
            final_result = "square"

        print(final_result)

        return final_result
