# -*- coding: utf-8 -*-
"""
Módulo para criação de pipelines de dados (Versão 3.0)

Funcionalidades:
- Carregamento a partir de um subconjunto de dados para debug rápido.
- Redimensionamento condicional ou manutenção do tamanho original da imagem.
- Altamente otimizado com tf.data para performance.
"""
import tensorflow as tf
from pathlib import Path
from typing import Union, Tuple, List

# Importa as configurações do projeto
try:
    from . import config
except ImportError:
    import config

Dataset = tf.data.Dataset


def _get_file_paths_and_labels(
    data_path: Path, subset_size: Union[int, None]
) -> Tuple[List[str], List[int]]:
    """Função interna para listar os caminhos dos ficheiros e os seus rótulos."""

    image_paths = []
    image_labels = []
    class_name_to_id = {name: i for i, name in enumerate(config.CLASS_NAMES)}

    print(f"A procurar por imagens em: {data_path}")
    if subset_size:
        print(f"Modo Debug: A usar no máximo {subset_size} imagens por classe.")

    for class_dir in data_path.iterdir():
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        label_id = class_name_to_id[class_name]

        count = 0
        for image_file in sorted(
            class_dir.glob("*.*")
        ):  # Pega todos os tipos de imagem
            image_paths.append(str(image_file))
            image_labels.append(label_id)
            count += 1
            if subset_size and count >= subset_size:
                break  # Para de adicionar ficheiros desta classe ao atingir o limite

    return image_paths, image_labels


def _parse_image(filename: str, label: int) -> Tuple[tf.Tensor, tf.Tensor]:
    """Carrega, decodifica e processa uma única imagem."""

    image_raw = tf.io.read_file(filename)
    # Decodifica para 3 canais (RGB). O TensorFlow deteta o formato (JPEG, PNG, etc).
    image = tf.io.decode_image(
        image_raw, channels=config.Model.CHANNELS, expand_animations=False
    )

    # Redimensionamento condicional
    if isinstance(config.Model.IMG_SIZE, int):
        image = tf.image.resize(image, [config.Model.IMG_SIZE, config.Model.IMG_SIZE])
    # Se IMG_SIZE for 'original', a imagem mantém o seu tamanho.

    # Converte o rótulo para o formato one-hot encoding
    label_one_hot = tf.one_hot(label, depth=config.NUM_CLASSES)

    return image, label_one_hot


def create_full_pipeline(
    data_path: Path, batch_size: int, is_training: bool = True
) -> Tuple[Dataset, int]:
    """
    Cria o pipeline completo e retorna o dataset e o número de imagens.

    Args:
        data_path (Path): Caminho para a pasta de dados.
        batch_size (int): Tamanho do lote.
        is_training (bool): Se True, aplica embaralhamento.

    Returns:
        Tuple[Dataset, int]: Uma tupla contendo:
                             - O pipeline tf.data pronto para uso.
                             - O número total de imagens no pipeline.
    """
    # Passo 1: Obter a lista de ficheiros e rótulos
    paths, labels = _get_file_paths_and_labels(
        data_path, config.Debug.SUBSET_SIZE_PER_CLASS
    )
    if not paths:
        raise ValueError(
            f"Nenhuma imagem encontrada em {data_path}. Verifique o caminho."
        )

    num_images = len(paths)

    # Passo 2: Criar o dataset a partir das listas
    dataset = Dataset.from_tensor_slices((paths, labels))

    # Passo 3: Embaralhar os dados ANTES de carregar as imagens (muito mais rápido)
    if is_training:
        buffer_size = len(paths)
        dataset = dataset.shuffle(buffer_size, reshuffle_each_iteration=True)

    # Passo 4: Mapear a função de carregamento e processamento
    dataset = dataset.map(_parse_image, num_parallel_calls=tf.data.AUTOTUNE)

    # Passo 5: Normalizar as imagens (pixéis de 0-1)
    normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
    dataset = dataset.map(
        lambda image, label: (normalization_layer(image), label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Passo 6: Agrupar em lotes
    # AVISO: Se IMG_SIZE='original' e as imagens tiverem tamanhos diferentes, isto irá falhar.
    # O utilizador deve garantir que batch_size=1 nesse caso.
    dataset = dataset.batch(batch_size)

    # Passo 7: Otimizações finais de performance
    dataset = dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    print(f"Pipeline completo criado para o caminho: {data_path}")
    return dataset, num_images


if __name__ == "__main__":
    print("--- A testar o data_loader avançado ---")

    # Mude esta configuração no config.py para testar
    print(f"Tamanho do subconjunto por classe: {config.Debug.SUBSET_SIZE_PER_CLASS}")
    print(f"Tamanho da imagem definido para: {config.Model.IMG_SIZE}")

    if isinstance(config.Model.IMG_SIZE, str) and config.Training.BATCH_SIZE > 1:
        print(
            "\nAVISO: A usar tamanho 'original' com batch_size > 1. Isto só funcionará se todas as suas imagens tiverem exatamente as mesmas dimensões."
        )

    train_ds = create_full_pipeline(
        data_path=config.PATH_TRAIN,
        batch_size=config.Training.BATCH_SIZE,
        is_training=True,
    )

    print("\nA inspecionar um lote do dataset de treino...")
    for images, labels in train_ds.take(1):
        print("Shape do lote de imagens:", images.shape)
        print("Shape do lote de rótulos:", labels.shape)
