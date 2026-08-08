import pandas as pd
import tensorflow as tf

IMG_SIZE = (224, 224)
BATCH_SIZE = 16


def load_dataset(csv_file, training=False):

    df = pd.read_csv(csv_file)

    image_paths = df["image_path"].values
    labels = df["label"].values

    # Convert class names to numbers
    class_names = sorted(df["label"].unique())
    label_to_number = {
        name: i for i, name in enumerate(class_names)
    }

    numeric_labels = [
        label_to_number[label] for label in labels
    ]

    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, numeric_labels)
    )

    def process_image(image_path, label):

        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, IMG_SIZE)

        # Convert pixels from 0-255 to 0-1
        image = image / 255.0

        return image, label

    dataset = dataset.map(
        process_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if training:
        dataset = dataset.shuffle(1000)

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset, class_names