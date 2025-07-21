# -*- coding: utf-8 -*-
"""
Ficheiro de Configuração Centralizado (Versão Final e Corrigida)
"""
import os
from pathlib import Path
from datetime import datetime
from typing import Union

# ==============================================================================
# 1. CONFIGURAÇÕES PRINCIPAIS (OS PONTOS A MUDAR)
# ==============================================================================
DATASET_NAME = "cifar10"
HAS_TRAIN_TEST_SPLIT = True


class Debug:
    """Configurações para acelerar testes e depuração."""

    SUBSET_SIZE_PER_CLASS: Union[int, None] = 20  # Mantenha em 20 para o teste


class Model:
    """Configurações da imagem."""

    IMG_SIZE: Union[int, str] = 32
    CHANNELS = 3


# ==============================================================================
# 2. VARIÁVEIS AUTOMATIZADAS (NÃO PRECISA DE MUDAR)
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / DATASET_NAME

if HAS_TRAIN_TEST_SPLIT:
    PATH_TRAIN = DATA_DIR / "train"
    PATH_TEST = DATA_DIR / "test"
else:
    PATH_TRAIN = DATA_DIR
    PATH_TEST = DATA_DIR

try:
    CLASS_NAMES = sorted([d.name for d in PATH_TRAIN.iterdir() if d.is_dir()])
    NUM_CLASSES = len(CLASS_NAMES)
except FileNotFoundError:
    print(f"AVISO: Diretório de treino '{PATH_TRAIN}' não encontrado.")
    CLASS_NAMES = []
    NUM_CLASSES = 0

PROJECT_NAME = f"{DATASET_NAME}_CNN"
TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")


class Paths:
    """Organiza todos os caminhos de saída do projeto."""

    OUTPUT_DIR = BASE_DIR / "output"

    # Define um diretório TEMPORÁRIO para a execução atual
    TMP_RUN_DIR = OUTPUT_DIR / f"_tmp_{PROJECT_NAME}_{TIMESTAMP}"

    # O modelo será salvo dentro da pasta temporária
    MODEL_FILE = TMP_RUN_DIR / "model.h5"


class Training:
    """Hiperparâmetros para o processo de treino."""

    EPOCHS = 5  # Reduzi para 5 para um teste rápido
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    LOSS = "categorical_crossentropy"
    OPTIMIZER = "adam"


class Augmentation:
    """Parâmetros para o aumento de dados."""

    ENABLED = True
    ROTATION_RANGE = 20
    WIDTH_SHIFT_RANGE = 0.2
    HEIGHT_SHIFT_RANGE = 0.2
    SHEAR_RANGE = 0.15
    ZOOM_RANGE = 0.15
    HORIZONTAL_FLIP = True
    FILL_MODE = "nearest"


# ==============================================================================
# AÇÃO FINAL: Cria o diretório da execução
# ==============================================================================
# Esta linha agora cria a pasta temporária, evitando o erro.
Paths.TMP_RUN_DIR.mkdir(parents=True, exist_ok=True)
