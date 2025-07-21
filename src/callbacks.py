# -*- coding: utf-8 -*-
"""
Módulo para a criação e gestão de callbacks do Keras.
"""
import tensorflow as tf
from typing import List

# Importa as configurações do projeto
try:
    from . import config
except ImportError:
    import config


def get_default_callbacks() -> List[tf.keras.callbacks.Callback]:
    """
    Retorna uma lista com os callbacks padrão para o treino.
    """
    callbacks = []

    # 1. Callback de Paragem Antecipada (EarlyStopping)
    early_stopper = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=30,
        verbose=1,
        mode="min",
        restore_best_weights=True,
    )
    callbacks.append(early_stopper)

    # 2. Callback para Salvar o Melhor Modelo (ModelCheckpoint)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(config.Paths.MODEL_FILE),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
        mode="max",
    )
    callbacks.append(model_checkpoint)

    # 3. Callback de Redução da Taxa de Aprendizagem (ReduceLROnPlateau)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1
    )
    callbacks.append(reduce_lr)

    # 4. Callback de TensorBoard
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=str(config.Paths.TMP_RUN_DIR / "logs"), histogram_freq=0
    )
    callbacks.append(tensorboard)

    print(
        "Callbacks configurados: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard"
    )
    return callbacks
