import tensorflow as tf


class XceptionFeatureExtractor:
    """
    Feature extractor using the Xception model.
    """
    def __init__(self, input_shape=(299, 299, 3)):
        self.input_shape = input_shape
        self.base_model = tf.keras.applications.Xception(include_top=False, input_shape=self.input_shape, weights='imagenet', pooling='avg')

        self.output_shape = self.base_model.output_shape[1:]

        self.model = self.build()
    
    def build(self):
        input_layer = self.base_model.input
        output_layer = self.base_model.output

        return tf.keras.Model(inputs=input_layer, outputs=output_layer)

    def extract_features(self, images):
        """
        Extract features from images using the Xception model.
        images: numpy array of images. Have to be preprocessed using tf.keras.applications.xception.preprocess_input.
        """
        return self.model.predict(images)

