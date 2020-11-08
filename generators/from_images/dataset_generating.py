import tensorflow as tf

def prepare_generators(image_dir,
                       batch_size,
                       image_size,
                       val_ratio=0.3,
                       buffer_size=32,
                       holdout_dir=None):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        image_dir,
        validation_split=val_ratio,
        subset="training",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        image_dir,
        validation_split=val_ratio,
        subset="validation",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
    )

    train_ds = train_ds.prefetch(buffer_size=buffer_size)
    val_ds = val_ds.prefetch(buffer_size=buffer_size)

    if holdout_dir:
        holdout_ds = tf.keras.preprocessing.image_dataset_from_directory(
            holdout_dir,
            validation_split=0.0,
            subset=None,
            seed=1337,
            image_size=image_size,
            batch_size=batch_size,
        )
        holdout_ds = holdout_ds.prefetch(buffer_size=buffer_size)
        return train_ds, val_ds, holdout_ds

    return train_ds, val_ds

import shutil
import os
import numpy as np

def get_files_from_folder(path):
    files = os.listdir(path)
    return np.asarray(files)

def split_train_holdout(path_to_data, path_to_holdout, holdout_ratio=0.1):
    # get dirs
    _, dirs, _ = next(os.walk(path_to_data))

    # calculates how many train data per class
    data_counter_per_class = np.zeros((len(dirs)))

    for i in range(len(dirs)):
        path = os.path.join(path_to_data, dirs[i])
        files = get_files_from_folder(path)
        data_counter_per_class[i] = len(files)
    holdout_counter = np.round(data_counter_per_class * holdout_ratio).astype(np.int)
    print('holdout_counter', dict(zip(dirs, holdout_counter)))

    # transfers files
    for i in range(len(dirs)):
        path_to_original = os.path.join(path_to_data, dirs[i])
        path_to_save = os.path.join(path_to_holdout, dirs[i])

        #creates dir
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
        files = get_files_from_folder(path_to_original)
        # moves data
        print(f'move holdout_counter {int(holdout_counter[i])} to {path_to_save}')
        for j in range(int(holdout_counter[i])):
            dst = os.path.join(path_to_save, files[j])
            src = os.path.join(path_to_original, files[j])
            shutil.move(src, dst)
