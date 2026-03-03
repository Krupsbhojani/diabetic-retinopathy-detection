"""
model.py — EfficientNetB3 architecture for Diabetic Retinopathy Detection
Supports two-phase fine-tuning: frozen backbone → full fine-tuning.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB3


def build_efficientnet_model(
    img_size: int = 224,
    num_classes: int = 5,
    dropout1: float = 0.4,
    dropout2: float = 0.3,
    dense1: int = 512,
    dense2: int = 256,
    trainable_backbone: bool = False,
) -> keras.Model:
    """
    Build EfficientNetB3-based model for DR severity grading.
    
    Architecture:
        Input → EfficientNetB3 (pretrained) → GAP → Dense(512) → BN → Dropout
              → Dense(256) → BN → Dropout → Dense(num_classes, softmax)
    
    Args:
        img_size:            Input image size (square)
        num_classes:         Number of output classes (5 for DR grades 0–4)
        dropout1:            Dropout rate after first dense layer
        dropout2:            Dropout rate after second dense layer
        dense1:              Units in first dense layer
        dense2:              Units in second dense layer
        trainable_backbone:  If False, backbone is frozen (Phase 1)
                             Set True for Phase 2 fine-tuning
    
    Returns:
        Compiled-ready Keras Model
    """
    # Load pretrained backbone
    backbone = EfficientNetB3(
        weights='imagenet',
        include_top=False,
        input_shape=(img_size, img_size, 3)
    )
    backbone.trainable = trainable_backbone

    inputs = keras.Input(shape=(img_size, img_size, 3), name='image_input')

    # Feature extraction
    x = backbone(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name='gap')(x)

    # Classifier head
    x = layers.Dense(dense1, name='dense_1')(x)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.Activation('relu', name='relu_1')(x)
    x = layers.Dropout(dropout1, name='dropout_1')(x)

    x = layers.Dense(dense2, name='dense_2')(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    x = layers.Activation('relu', name='relu_2')(x)
    x = layers.Dropout(dropout2, name='dropout_2')(x)

    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    model = keras.Model(inputs, outputs, name='EfficientNetB3_DR')
    return model


def unfreeze_backbone(model: keras.Model, learning_rate: float = 1e-5) -> keras.Model:
    """
    Unfreeze all backbone layers for Phase 2 fine-tuning.
    Recompiles the model with a lower learning rate.
    
    Args:
        model:         Model from build_efficientnet_model()
        learning_rate: Low learning rate for careful fine-tuning (default 1e-5)
    
    Returns:
        Recompiled model with trainable backbone
    """
    # The backbone is layers[1] (after the Input layer)
    model.layers[1].trainable = True

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print(f'Backbone unfrozen. Total trainable params: {model.count_params():,}')
    return model


def get_callbacks(
    model_save_path: str,
    monitor: str = 'val_accuracy',
    patience_early_stop: int = 7,
    patience_lr: int = 3,
) -> list:
    """
    Return standard training callbacks.
    
    Returns:
        List of [EarlyStopping, ReduceLROnPlateau, ModelCheckpoint]
    """
    return [
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience_early_stop,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=patience_lr,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor=monitor,
            save_best_only=True,
            verbose=1
        ),
    ]
