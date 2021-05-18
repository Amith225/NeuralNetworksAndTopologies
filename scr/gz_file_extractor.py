import gzip
import numpy as np

tr_file = r'C:\Users\Amith M\Downloads\emnist-balanced-test-images-idx3-ubyte.gz'
tl_file = r'C:\Users\Amith M\Downloads\emnist-balanced-test-labels-idx1-ubyte.gz'
def training_images():
    with gzip.open(tr_file, 'r') as f:
        # first 4 bytes is a magic number
        magic_number = int.from_bytes(f.read(4), 'big')
        # second 4 bytes is the number of images
        image_count = int.from_bytes(f.read(4), 'big')
        # third 4 bytes is the row count
        row_count = int.from_bytes(f.read(4), 'big')
        # fourth 4 bytes is the column count
        column_count = int.from_bytes(f.read(4), 'big')
        # rest is the image pixel data, each pixel is stored as an unsigned byte
        # pixel values are 0 to 255
        image_data = f.read()
        images = np.frombuffer(image_data, dtype=np.uint8)\
            .reshape((image_count, row_count, column_count))
        return images


def training_labels():
    with gzip.open(tl_file, 'r') as f:
        # first 4 bytes is a magic number
        magic_number = int.from_bytes(f.read(4), 'big')
        # second 4 bytes is the number of labels
        label_count = int.from_bytes(f.read(4), 'big')
        # rest is the label data, each label is stored as unsigned byte
        # label values are 0 to 9
        label_data = f.read()
        labels = np.frombuffer(label_data, dtype=np.uint8)
        return labels


training_img, labels_img = training_images(), training_labels()
labels = np.zeros((labels_img.size, labels_img.max()+1))
labels[np.arange(labels_img.size), labels_img] = 1
training_img.resize((len(training_img), 784, 1))

# save at your path
np.savez_compressed('image_classification_47_balanced_test.npz', training_img, labels)
