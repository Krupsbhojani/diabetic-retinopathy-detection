"""
train.py — CLI training script for Diabetic Retinopathy Detection

Usage:
    python src/train.py --epochs 50 --batch_size 32 --img_size 224 --lr 1e-4

Two-phase training:
    Phase 1 — Frozen EfficientNetB3 backbone, train head only (10 epochs)
    Phase 2 — Full fine-tuning with low LR (remaining epochs)
"""

import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path

from model import build_efficientnet_model, unfreeze_backbone, get_callbacks
from utils import set_seed, load_image


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='Train DR Detection Model')
    parser.add_argument('--img_size',    type=int,   default=224)
    parser.add_argument('--batch_size',  type=int,   default=32)
    parser.add_argument('--epochs',      type=int,   default=50,  help='Phase 2 epochs')
    parser.add_argument('--phase1_ep',   type=int,   default=10,  help='Phase 1 epochs')
    parser.add_argument('--lr',          type=float, default=1e-5, help='Phase 2 LR')
    parser.add_argument('--seed',        type=int,   default=42)
    parser.add_argument('--data_dir',    type=str,   default='data/processed')
    parser.add_argument('--model_dir',   type=str,   default='models')
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Data pipeline
# ─────────────────────────────────────────────────────────────────────────────

def make_tf_dataset(df, img_size, batch_size, augment=False, shuffle=False, seed=42):
    def load(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, [img_size, img_size])
        img = tf.cast(img, tf.float32) / 255.0
        img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        if augment:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_flip_up_down(img)
            img = tf.image.random_brightness(img, 0.2)
            img = tf.image.random_contrast(img, 0.8, 1.2)
        return img, tf.one_hot(label, 5)

    ds = tf.data.Dataset.from_tensor_slices((
        tf.constant(df['image_path'].values),
        tf.constant(df['diagnosis'].values)
    ))
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(1000, seed=seed)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)

    DATA_DIR  = Path(args.data_dir)
    MODEL_DIR = Path(args.model_dir)
    MODEL_DIR.mkdir(exist_ok=True)

    print(f'\nGPU: {[g.name for g in tf.config.list_physical_devices("GPU")]}')
    print(f'Config: img_size={args.img_size}, batch={args.batch_size}, epochs={args.epochs}')

    # Load data
    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    val_df   = pd.read_csv(DATA_DIR / 'val.csv')

    train_ds = make_tf_dataset(train_df, args.img_size, args.batch_size, augment=True,  shuffle=True)
    val_ds   = make_tf_dataset(val_df,   args.img_size, args.batch_size, augment=False, shuffle=False)

    cw_arr  = compute_class_weight('balanced', classes=np.arange(5), y=train_df['diagnosis'].values)
    cw_dict = dict(enumerate(cw_arr))
    print(f'Class weights: {cw_dict}')

    # ── Phase 1 ──────────────────────────────────────────────────────────────
    print('\n' + '='*50)
    print('PHASE 1: Frozen backbone — training classifier head')
    print('='*50)
    model = build_efficientnet_model(img_size=args.img_size, trainable_backbone=False)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.fit(
        train_ds, validation_data=val_ds,
        epochs=args.phase1_ep,
        class_weight=cw_dict,
        callbacks=get_callbacks(str(MODEL_DIR / 'phase1_best.keras'), patience_early_stop=5),
        verbose=1
    )

    # ── Phase 2 ──────────────────────────────────────────────────────────────
    print('\n' + '='*50)
    print('PHASE 2: Full fine-tuning — all layers trainable')
    print('='*50)
    model = unfreeze_backbone(model, learning_rate=args.lr)
    model.fit(
        train_ds, validation_data=val_ds,
        epochs=args.epochs,
        class_weight=cw_dict,
        callbacks=get_callbacks(str(MODEL_DIR / 'best_model.keras')),
        verbose=1
    )

    print(f'\nTraining complete. Best model saved to: {MODEL_DIR}/best_model.keras')


if __name__ == '__main__':
    main()
