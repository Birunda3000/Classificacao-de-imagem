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
        monitor='val_loss',
        patience=10,
        verbose=1,
        mode='min',
        restore_best_weights=True
    )
    callbacks.append(early_stopper)

    # 2. Callback para Salvar o Melhor Modelo (ModelCheckpoint)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(config.Paths.MODEL_FILE),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1,
        mode='max'
    )
    callbacks.append(model_checkpoint)
    
    print(f"Callbacks configurados: EarlyStopping (monitora 'val_loss'), ModelCheckpoint (monitora 'val_accuracy')")
    return callbacks