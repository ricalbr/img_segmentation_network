from PIL import Image
from datetime import datetime
import os
import numpy as np
import cv2
import random as rn
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# DIRECTORIES
# I set up all the directories for working with the data in Kaggle
ROOT = '../input/ann-and-dl-image-segmentation/Segmentation_Dataset'
IMG_DIR = os.path.join(ROOT, 'training', 'images')
MSK_DIR = os.path.join(ROOT, 'training', 'masks')
TEST_DIR = os.path.join(ROOT, 'test')

# dimension of both axes of the images
DIM = 256

# set the seeds for random, numpy and tensorflow packages


def set_seed(SEED):
    rn.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    return

# check the availability of the GPU


def check_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(
                logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    return

# Import data function


def train_val_dataset(val_split, apply_data_augmentation, BS, SEED, preprocess_input=None):
    """
    The function is meant to import the data, eventually preprocess the inputs and create the dataset
    that will be used in the training.

    Parameters:
    val_split: float value indicating the percetage of the validation dataset over the whole dataset.
    apply_data_augmentation: boolean value, discriminate whether or not to use data augmentation.
    BS: integer for the batch size.
    SEED: integer, seed that is used to split (and shuffle) the training and validation sets.
    preprocess_input: function to perform a preprocess of the input data. Required for transfer learning.
    """

    # GENERATORS
    num_threads = 8

    def prepare_target(x_, y_):
        y_ = tf.cast(y_, tf.float32)
        return x_, y_
    # if preprocess_input is not present a rescale of the data is needed.
    if preprocess_input:
        if apply_data_augmentation:
            train_img_data_gen = ImageDataGenerator(horizontal_flip=True,
                                                    vertical_flip=True,
                                                    preprocessing_function=preprocess_input,
                                                    validation_split=val_split)
            train_mask_data_gen = ImageDataGenerator(horizontal_flip=True,
                                                     vertical_flip=True,
                                                     rescale=1./255,
                                                     validation_split=val_split)
        else:
            train_img_data_gen = ImageDataGenerator(
                preprocessing_function=preprocess_input,  validation_split=val_split)
            train_mask_data_gen = ImageDataGenerator(
                rescale=1/255.,  validation_split=val_split)

        valid_img_data_gen = ImageDataGenerator(
            preprocessing_function=preprocess_input, validation_split=val_split)
        valid_mask_data_gen = ImageDataGenerator(
            rescale=1/255., validation_split=val_split)
    else:
        if apply_data_augmentation:
            train_img_data_gen = ImageDataGenerator(horizontal_flip=True,
                                                    vertical_flip=True,
                                                    rescale=1./255,
                                                    validation_split=val_split)
            train_mask_data_gen = ImageDataGenerator(horizontal_flip=True,
                                                     vertical_flip=True,
                                                     rescale=1./255,
                                                     validation_split=val_split)
        else:
            train_img_data_gen = ImageDataGenerator(
                rescale=1./255,  validation_split=val_split)
            train_mask_data_gen = ImageDataGenerator(
                rescale=1/255.,  validation_split=val_split)

        valid_img_data_gen = ImageDataGenerator(
            rescale=1./255, validation_split=val_split)
        valid_mask_data_gen = ImageDataGenerator(
            rescale=1/255., validation_split=val_split)

    # DATASET
        # TRAINING
    train_img_gen = train_img_data_gen.flow_from_directory(
        IMG_DIR,
        target_size=(DIM, DIM),
        batch_size=BS,
        class_mode=None,
        shuffle=True,
        color_mode='rgb',
        interpolation='bilinear',
        seed=SEED,
        subset='training')
    train_mask_gen = train_mask_data_gen.flow_from_directory(
        MSK_DIR,
        target_size=(DIM, DIM),
        batch_size=BS,
        class_mode=None,
        shuffle=True,
        interpolation='bilinear',
        color_mode='grayscale',
        seed=SEED,
        subset='training')

    # creation of the dataset for the training set
    train_gen = zip(train_img_gen, train_mask_gen)
    train_dataset = tf.data.Dataset.from_generator(lambda: train_gen,
                                                   output_types=(
                                                       tf.float32, tf.float32),
                                                   output_shapes=([None, DIM, DIM, 3], [None, DIM, DIM, 1]))
    train_dataset = train_dataset.map(
        prepare_target, num_parallel_calls=num_threads)
    train_dataset = train_dataset.repeat()
    train_dataset = train_dataset.prefetch(1)

    # VALIDATION
    valid_img_gen = valid_img_data_gen.flow_from_directory(
        IMG_DIR,
        target_size=(DIM, DIM),
        batch_size=BS,
        class_mode=None,
        shuffle=True,
        color_mode='rgb',
        interpolation='bilinear',
        seed=SEED,
        subset='validation')
    valid_mask_gen = valid_mask_data_gen.flow_from_directory(
        MSK_DIR,
        target_size=(DIM, DIM),
        batch_size=BS,
        class_mode=None,
        shuffle=True,
        color_mode='grayscale',
        interpolation='bilinear',
        seed=SEED,
        subset='validation')

    # creation of the dataset for the validation set
    valid_gen = zip(valid_img_gen, valid_mask_gen)
    valid_dataset = tf.data.Dataset.from_generator(lambda: valid_gen,
                                                   output_types=(
                                                       tf.float32, tf.float32),
                                                   output_shapes=([None, DIM, DIM, 3], [None, DIM, DIM, 1]))
    valid_dataset = valid_dataset.map(
        prepare_target, num_parallel_calls=num_threads)
    valid_dataset = valid_dataset.repeat()
    valid_dataset = valid_dataset.prefetch(1)
    return train_dataset, train_img_gen, valid_dataset, valid_img_gen


def create_csv(results, results_dir='./'):
    """
    This function will output the csv file for the submission.

    Parameters:
    results: dictionary with images Id as key and rle encoding of the image as values.
    results_dir: optional directory to save the output.
    """

    csv_fname = 'results_'
    csv_fname += datetime.now().strftime('%b%d_%H-%M-%S') + '.csv'
    with open(csv_fname, 'w') as f:
        f.write('ImageId,EncodedPixels,Width,Height\n')
        for key, value in results.items():
            f.write(key + ',' + str(value) + ',' + '256' + ',' + '256' + '\n')


def rle_encode(img):
    """
    This function takes as input an image and returns its RLE encoding.
    """
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def sub_csv(model, preprocess_input=None, th=0.5):
    """
    This function is meant to be used at the end of the training of the model to make prediction and produce the csv file.
    It loads all the images from the test directory (one at the time), process them, make the prediction and create the dictionary
    for the create_csv function.

    Parameters:
    model: tf.keras model for the predictions.
    preprocess_input: function to perform a preprocess of the input data. Required for transfer learning.
    th: optional, threshold for discriminate between the two classes (0,1) after making the prediction.
    """
    results = {}
    imgs = []
    valid_images = [".tif"]
    i = 0

    TEST_IMG = os.path.join(TEST_DIR, 'images', 'img')
    for f in os.listdir(TEST_IMG):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        img = Image.open(os.path.join(TEST_IMG, f)).convert('RGB')
        img = img.resize((256, 256))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, 0)
        img_array.astype('float32')
        if preprocess_input:
            img_array = preprocess_input(img_array)
        else:
            img_array = img_array/255.

        predict = model.predict(img_array)
        predict = tf.cast(predict > th, tf.float32)
        predict = np.reshape(predict, (DIM, DIM, 1)).squeeze()

        # plot some of the prediction
        if i % 513 == 0:
            plt.figure()
            plt.subplot(121)
            plt.imshow(img_array.squeeze())
            plt.title('img')
            plt.subplot(122)
            plt.imshow(predict)
            plt.title('prediction')
        i += 1

        results[os.path.splitext(f)[0]] = rle_encode(predict)
    create_csv(results)


def sub_csv_postprocess(model, preprocess_input=None, th=0.5):
    """
    This function is meant to be used at the end of the training of the model to make prediction and produce the csv file + POST PROCESSING.
    It loads all the images from the test directory (one at the time), process them, make the prediction and create the dictionary
    for the create_csv function.

    Parameters:
    model: tf.keras model for the predictions.
    preprocess_input: function to perform a preprocess of the input data. Required for transfer learning.
    th: optional, threshold for discriminate between the two classes (0,1) after making the prediction.
    """
    results = {}
    imgs = []
    valid_images = [".tif"]
    i = 0

    TEST_IMG = os.path.join(TEST_DIR, 'images', 'img')
    for f in os.listdir(TEST_IMG):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        img = Image.open(os.path.join(TEST_IMG, f)).convert('RGB')
        img = img.resize((256, 256))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, 0)
        img_array.astype('float32')
        if preprocess_input:
            img_array = preprocess_input(img_array)
        else:
            img_array = img_array/255.

        pred = model.predict(img_array)
        predict = tf.cast(pred > th, tf.float32)
        predict = np.reshape(predict, (DIM, DIM, 1)).squeeze()

        kernel = np.ones((3, 3), np.uint8)
        closing = cv2.morphologyEx(predict, cv2.MORPH_CLOSE, kernel)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

        # plot some of the predictions
        if i % 513 == 0:
            plt.figure(figsize=(15, 15))
            plt.subplot(131), plt.imshow(np.array(img).squeeze())
            plt.title('img')
            plt.subplot(132), plt.imshow(predict)
            plt.title('prediction')
            plt.subplot(133), plt.imshow(opening)
            plt.title('post_process')

        i += 1

        results[os.path.splitext(f)[0]] = rle_encode(opening)
    create_csv(results)
