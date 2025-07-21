# -*- coding: utf-8 -*-
"""
Módulo para a definição de arquiteturas de modelos Keras (Versão 2.0)

As funções agora recebem os parâmetros de configuração como argumentos,
usando os valores do ficheiro config.py como padrão (injeção de dependência).
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
)
from typing import Tuple

# Importa as configurações do projeto para usar como valores padrão
try:
    from . import config
except ImportError:
    import config


def build_cnn(
    input_shape: Tuple[int, int, int] = (
        config.Model.IMG_SIZE,
        config.Model.IMG_SIZE,
        config.Model.CHANNELS,
    ),
    num_classes: int = config.NUM_CLASSES,
) -> Model:
    """
    Constrói um modelo CNN (Rede Neural Convolucional).

    Args:
        input_shape (Tuple[int, int, int], optional): A forma dos dados de entrada (altura, largura, canais).
                                                      O padrão é o valor de config.py.
        num_classes (int, optional): O número de classes para a camada de saída.
                                     O padrão é o valor de config.py.

    Returns:
        Model: O modelo Keras construído, mas ainda não compilado.
    """
    print(
        f"Construindo o modelo CNN para input_shape={input_shape} e {num_classes} classes..."
    )

    model = Sequential(
        [
            Input(shape=input_shape, name="input_layer"),
            Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1"),
            BatchNormalization(name="bn1"),
            MaxPooling2D(pool_size=(2, 2), name="pool1"),
            Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2"),
            BatchNormalization(name="bn2"),
            MaxPooling2D(pool_size=(2, 2), name="pool2"),
            Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3"),
            BatchNormalization(name="bn3"),
            MaxPooling2D(pool_size=(2, 2), name="pool3"),
            Flatten(name="flatten"),
            Dense(512, activation="relu", name="dense1"),
            Dropout(0.5, name="dropout"),
            Dense(num_classes, activation="softmax", name="output_layer"),
        ],
        name="Simple_CNN_v2",
    )

    print("\n" + "=" * 50)
    print("RESUMO DA ARQUITETURA DO MODELO")
    print("=" * 50)
    model.summary()
    print("=" * 50)

    return model


# --- Exemplo de como adicionar outros modelos ---
#
# def build_vgg_style_model() -> tf.keras.Model:
#     """ Constrói um modelo com uma arquitetura no estilo VGG. """
#     # ... (código para construir um modelo diferente)
#     return model
#

if __name__ == "__main__":
    # Este bloco só executa se você rodar `python src/model_arquiteture.py` diretamente
    # É útil para verificar rapidamente se a construção do modelo está a funcionar.
    print("A testar a construção do modelo...")
    meu_modelo = build_cnn()
    # O teste confirma que o modelo foi criado sem erros.
