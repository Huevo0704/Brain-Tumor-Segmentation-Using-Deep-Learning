# data_loader.py
"""
Chịu trách nhiệm cho việc tải, tăng cường (augment) và tạo
tf.data.Dataset pipeline cho quá trình huấn luyện.
"""

import cv2
import numpy as np
import tensorflow as tf
import albumentations as A
import config

# ==============================================================================
# 1. ĐỊNH NGHĨA AUGMENTATION PIPELINE
# ==============================================================================
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Affine(scale={"x": (0.85, 1.15), "y": (0.85, 1.15)}, translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)}, rotate=(-25, 25), p=0.7),
    A.ElasticTransform(alpha=1.5, sigma=55, p=0.7),
    A.GridDistortion(p=0.3),
    A.GaussNoise(p=0.5),
])

# ==============================================================================
# 2. CÁC HÀM PARSE VÀ TẠO DATASET
# ==============================================================================
def parse_image_mask(img_path, mask_path, augment=True):
    """Đọc và xử lý một cặp ảnh-mask từ đường dẫn."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, config.IMG_SIZE) / 255.0
    img = np.expand_dims(img, axis=-1).astype(np.float32)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, config.IMG_SIZE) / 255.0
    mask = np.expand_dims(mask, axis=-1).astype(np.float32)

    if augment:
        augmented = transform(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']

    return img, mask

def tf_parse_image_mask(img_path, mask_path, augment=True):
    """Wrapper để sử dụng hàm `parse_image_mask` trong TensorFlow graph."""
    def _parse_wrapper(img_path_tensor, mask_path_tensor):
        img_path_str = img_path_tensor.numpy().decode('utf-8')
        mask_path_str = mask_path_tensor.numpy().decode('utf-8')
        return parse_image_mask(img_path_str, mask_path_str, augment)

    img, mask = tf.py_function(func=_parse_wrapper,
                              inp=[img_path, mask_path],
                              Tout=[tf.float32, tf.float32])
    img.set_shape((*config.IMG_SIZE, 1))
    mask.set_shape((*config.IMG_SIZE, 1))
    return img, mask

def create_dataset(image_paths, mask_paths, augment=False):
    """Tạo một đối tượng tf.data.Dataset từ danh sách đường dẫn."""
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.map(lambda img, mask: tf_parse_image_mask(img, mask, augment),
                            num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        dataset = dataset.cache().shuffle(1000) # Cache sau khi map và trước khi shuffle
    
    dataset = dataset.batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset
