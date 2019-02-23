from yolo import Yolo
import json
if __name__ == "__main__":

    CONFIG_FILE = 'config.json'
    with open(CONFIG_FILE) as f:
        config = json.load(f)
    yolo = Yolo(config)
    yolo.start_training()
























    # Initiate the train and test generators with data Augumentation
    # train_datagen = ImageDataGenerator(
    #     rescale=1. / 255,
    #     horizontal_flip=True,
    #     fill_mode="nearest",
    #     zoom_range=0.3,
    #     width_shift_range=0.3,
    #     height_shift_range=0.3,
    #     rotation_range=30)
    #
    # test_datagen = ImageDataGenerator(
    #     rescale=1. / 255,
    #     horizontal_flip=True,
    #     fill_mode="nearest",
    #     zoom_range=0.3,
    #     width_shift_range=0.3,
    #     height_shift_range=0.3,
    #     rotation_range=30)
    #
    # train_generator = train_datagen.flow_from_directory(
    #     train_data_dir,
    #     target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    #     batch_size=batch_size,
    #     class_mode="categorical")
    #
    # validation_generator = test_datagen.flow_from_directory(
    #     validation_data_dir,
    #     target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    #     class_mode="categorical")