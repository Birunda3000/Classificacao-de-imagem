# -*- coding: utf-8 -*-
"""
Módulo para a criação de camadas de aumento de dados (Data Augmentation).
"""
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# Importa as configurações do projeto
try:
    from . import config
except ImportError:
    import config

def build_augmentation_layer() -> Sequential:
    """
    Constrói um modelo Sequencial do Keras contendo as camadas de aumento de dados.

    As camadas são ativadas apenas durante o treino.

    Returns:
        Sequential: Um modelo Keras que pode ser aplicado a um lote de imagens.
    """
    if not config.Augmentation.ENABLED:
        # Se a aumentação estiver desativada, retorna um modelo vazio
        return Sequential(name="augmentation_disabled")
        
    # Cria um "mini-modelo" que contém apenas as camadas de aumentação
    data_augmentation = Sequential(
        [
            layers.RandomFlip("horizontal", 
                              input_shape=(config.Model.IMG_SIZE, config.Model.IMG_SIZE, config.Model.CHANNELS)),
            layers.RandomRotation(config.Augmentation.ROTATION_RANGE / 360), # Keras usa fração de 360 graus
            layers.RandomZoom(config.Augmentation.ZOOM_RANGE),
            layers.RandomTranslation(height_factor=config.Augmentation.HEIGHT_SHIFT_RANGE, 
                                     width_factor=config.Augmentation.WIDTH_SHIFT_RANGE),
            # layers.RandomContrast(0.1), # Pode adicionar outras como contraste
        ],
        name="data_augmentation"
    )
    print("Camada de aumento de dados construída com sucesso.")
    return data_augmentation

def apply_augmentation(dataset: tf.data.Dataset, augmentation_model: Sequential) -> tf.data.Dataset:
    """
    Aplica as transformações de aumento de dados a um dataset.

    Args:
        dataset (tf.data.Dataset): O dataset de treino.
        augmentation_model (Sequential): O modelo com as camadas de aumentação.

    Returns:
        tf.data.Dataset: O dataset com a aumentação aplicada a cada lote.
    """
    if not config.Augmentation.ENABLED:
        return dataset
        
    # Usa .map() para aplicar o "mini-modelo" de aumentação a cada lote do dataset.
    # A lógica interna do Keras garante que isto só acontece durante o treino.
    augmented_dataset = dataset.map(
        lambda image, label: (augmentation_model(image, training=True), label),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    print("Aumento de dados aplicado ao pipeline de treino.")
    return augmented_dataset