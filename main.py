from src.objects.image_handler import ImageHandler
from src.objects.caption_handler import CaptionHandler
from src.objects.data_handler import DataHandler
from src.models.captioner_model import CaptionerModel
import tensorflow as tf

if __name__ == "__main__":

    tf.config.run_functions_eagerly(True)

    image_handler = ImageHandler(
        folder_path="./test_data/test", dimensions=(299, 299), preprocess_function=tf.keras.applications.xception.preprocess_input)
    
    caption_handler = CaptionHandler(
        filenames=image_handler.filenames, filepath="./test_data/test.txt")
    
    data_handler = DataHandler(image_handler, caption_handler)

    train_generator, val_generator = data_handler.get_generators(batch_size=32)

    captioner_model = CaptionerModel(tokenizer_wrapper=caption_handler.get_tokenizer_wrapper())

    captioner_model.compile(optimizer='adam', run_eagerly=True)

    captioner_model.fit(train_generator, epochs=1, validation_data=val_generator)