# model.py
"""
Định nghĩa kiến trúc của mô hình U-Net với Attention Gates.
"""

import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Conv2DTranspose,
                                     concatenate, BatchNormalization, Dropout,
                                     LeakyReLU, Add, Activation)
from tensorflow.keras.models import Model
import config

def attention_gate(g, x, filters):
    """Khối Attention Gate."""
    g1 = Conv2D(filters, 1, padding='same')(g)
    x1 = Conv2D(filters, 1, padding='same')(x)
    psi = Activation('relu')(Add()([g1, x1]))
    psi = Conv2D(1, 1, padding='same', activation='sigmoid')(psi)
    return tf.keras.layers.multiply([x, psi])

def conv_block(x, filters, dropout_rate=0.1):
  """Khối tích chập kép (Conv-BN-LeakyReLU) x2."""
  x = Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
  x = BatchNormalization()(x); x = LeakyReLU(alpha=0.1)(x); x = Dropout(dropout_rate)(x)
  x = Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
  x = BatchNormalization()(x); x = LeakyReLU(alpha=0.1)(x)
  return x

def build_unet_with_attention(input_shape=(config.IMG_SIZE[0], config.IMG_SIZE[1], 1)):
    """Xây dựng toàn bộ kiến trúc mô hình Attention U-Net."""
    inputs = tf.keras.Input(input_shape)

    # Encoder
    c1 = conv_block(inputs, 64, 0.1); p1 = MaxPooling2D(2)(c1)
    c2 = conv_block(p1, 128, 0.1); p2 = MaxPooling2D(2)(c2)
    c3 = conv_block(p2, 256, 0.2); p3 = MaxPooling2D(2)(c3)
    c4 = conv_block(p3, 512, 0.3); p4 = MaxPooling2D(2)(c4)

    # Bottleneck
    c5 = conv_block(p4, 1024, 0.4)

    # Decoder
    u6 = Conv2DTranspose(512, 2, strides=2, padding='same')(c5)
    attn6 = attention_gate(u6, c4, 512); u6 = concatenate([u6, attn6]); c6 = conv_block(u6, 512, 0.3)

    u7 = Conv2DTranspose(256, 2, strides=2, padding='same')(c6)
    attn7 = attention_gate(u7, c3, 256); u7 = concatenate([u7, attn7]); c7 = conv_block(u7, 256, 0.2)

    u8 = Conv2DTranspose(128, 2, strides=2, padding='same')(c7)
    attn8 = attention_gate(u8, c2, 128); u8 = concatenate([u8, attn8]); c8 = conv_block(u8, 128, 0.1)

    u9 = Conv2DTranspose(64, 2, strides=2, padding='same')(c8)
    attn9 = attention_gate(u9, c1, 64); u9 = concatenate([u9, attn9]); c9 = conv_block(u9, 64, 0.1)

    outputs = Conv2D(1, 1, activation='sigmoid', dtype='float32')(c9)

    model = Model(inputs, outputs, name="Attention_U-Net")
    return model
