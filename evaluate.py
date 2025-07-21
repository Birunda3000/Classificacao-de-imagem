# -*- coding: utf-8 -*-
"""
Script de Avaliação de Modelo Pós-Treino.

Este script é utilizado para carregar um modelo já treinado e avaliar a sua
performance no conjunto de dados de teste.

COMO USAR:
----------
Execute este script a partir do terminal, na pasta raiz do projeto.

1. Para avaliar a execução de treino mais recente (uso mais comum):
   -------------------------------------------------
   python evaluate.py
   -------------------------------------------------
   O script irá procurar automaticamente na pasta 'output/' pelo último treino
   realizado e avaliar o modelo salvo lá.

2. Para avaliar uma execução de treino específica:
   -----------------------------------------------------------------------------------------------
   python evaluate.py --run_dir output/NOME_DA_PASTA_DA_EXECUCAO
   -----------------------------------------------------------------------------------------------
   Substitua 'NOME_DA_PASTA_DA_EXECUCAO' pelo nome completo da pasta do treino
   que você deseja avaliar (ex: 'CIFAR10_CNN_acc_val=0.8765_20250722_103000').

O QUE O SCRIPT FAZ:
-------------------
- Carrega o modelo .h5 da pasta da execução.
- Carrega o conjunto de dados de teste usando o data_loader.
- Imprime a perda (loss) e a acurácia (accuracy) no conjunto de teste.
- Gera e exibe um relatório visual, como a matriz de confusão.

Script de Avaliação de Modelo Pós-Treino (Versão 2.0)

Gera um relatório de performance completo para os conjuntos de dados
de treino, teste e combinado, incluindo:
- Acurácia, Perda
- Relatório de Classificação (Precisão, Recall/Sensibilidade, F1-Score)
- Matriz de Confusão
"""
import tensorflow as tf
import numpy as np
import argparse
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score

# Importa os nossos módulos
from src import config
from src import data_loader
from src import visualization

def get_latest_run_dir(output_dir: Path) -> Path:
    """Encontra o diretório da execução de treino mais recente."""
    try:
        run_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
        latest_run_dir = max(run_dirs, key=lambda d: d.stat().st_mtime)
        return latest_run_dir
    except (ValueError, FileNotFoundError):
        print(f"ERRO: Nenhum diretório de execução encontrado em '{output_dir}'.")
        exit()

def generate_performance_report(
    model: tf.keras.Model, 
    dataset: tf.data.Dataset, 
    dataset_name: str
):
    """
    Gera e exibe um relatório de performance completo para um determinado dataset.
    """
    print("\n" + "-"*60)
    print(f"RELATÓRIO DE PERFORMANCE - CONJUNTO DE DADOS: {dataset_name.upper()}")
    print("-"*60)

    # Passo 1: Avaliação geral com model.evaluate()
    loss, accuracy = model.evaluate(dataset, verbose=0)
    print(f"\n  - Acurácia Geral (evaluate): {accuracy:.4f}")
    print(f"  - Perda Geral (evaluate): {loss:.4f}")

    # Passo 2: Previsões para métricas detalhadas
    print("\nGerando previsões para o relatório detalhado...")
    y_pred_probs = model.predict(dataset, verbose=0)
    y_pred_labels = np.argmax(y_pred_probs, axis=1)
    
    y_true_labels_one_hot = np.concatenate([y for _, y in dataset], axis=0)
    y_true_labels = np.argmax(y_true_labels_one_hot, axis=1)

    # Passo 3: Relatório de Classificação (Precisão, Recall, F1-Score)
    print("\n  - Relatório de Classificação Detalhado:")
    report = classification_report(
        y_true_labels, 
        y_pred_labels, 
        target_names=config.CLASS_NAMES
    )
    print(report)

    # Passo 4: Matriz de Confusão
    print("Gerando Matriz de Confusão...")
    visualization.plot_confusion_matrix(
        y_true_labels, 
        y_pred_labels,
        title=f"Matriz de Confusão - {dataset_name}"
    )

def evaluate(run_dir: Path):
    """
    Orquestra o processo de avaliação completo.
    """
    print("="*60)
    print(f"INICIANDO AVALIAÇÃO PARA A EXECUÇÃO: {run_dir.name}")
    print("="*60)

    # --- Carregar o Modelo ---
    print("\n[ETAPA 1/3] Carregando o modelo treinado...")
    model_path = run_dir / "model.h5"
    if not model_path.exists():
        print(f"ERRO: Ficheiro do modelo não encontrado em '{model_path}'"); return
    model = tf.keras.models.load_model(model_path)

    # --- Carregar os Datasets ---
    print("\n[ETAPA 2/3] Carregando datasets de treino e teste...")
    train_ds = data_loader.create_full_pipeline(config.PATH_TRAIN, config.Training.BATCH_SIZE, is_training=False)
    test_ds = data_loader.create_full_pipeline(config.PATH_TEST, config.Training.BATCH_SIZE, is_training=False)
    
    # Cria um dataset combinado
    combined_ds = train_ds.concatenate(test_ds)

    # --- Gerar Relatórios ---
    print("\n[ETAPA 3/3] Gerando relatórios de performance...")
    
    # Relatório para Dados de Teste (o mais importante para avaliar a generalização)
    generate_performance_report(model, test_ds, "Dados de Teste")
    
    # Relatório para Dados de Treino (útil para diagnosticar overfitting)
    generate_performance_report(model, train_ds, "Dados de Treino")
    
    # Relatório para Dados Combinados
    generate_performance_report(model, combined_ds, "Dados Combinados (Treino + Teste)")

    print("\n" + "="*60)
    print("AVALIAÇÃO CONCLUÍDA")
    print("="*60)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script de Avaliação de Modelo.")
    parser.add_argument(
        '--run_dir', 
        type=str, 
        help="(Opcional) Caminho para a pasta da execução a ser avaliada."
    )
    args = parser.parse_args()

    if args.run_dir:
        run_to_evaluate = Path(args.run_dir)
    else:
        print("Nenhum diretório especificado. A procurar pela execução mais recente...")
        run_to_evaluate = get_latest_run_dir(config.Paths.OUTPUT_DIR)
    
    evaluate(run_to_evaluate)