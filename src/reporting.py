# -*- coding: utf-8 -*-
"""
Módulo para a geração de relatórios de texto pós-treino.
"""
from pathlib import Path
from typing import Dict, Any
import tensorflow as tf

# Importa as configurações do projeto
try:
    from . import config
except ImportError:
    import config

def save_training_report(
    report_path: Path, 
    history: tf.keras.callbacks.History, 
    timings: Dict[str, Any],
    num_train_images: int,
    num_val_images: int
):
    """
    Gera e escreve o ficheiro de relatório final (report.txt) para uma execução de treino.

    Args:
        report_path (Path): O caminho completo onde o relatório será salvo.
        history (tf.keras.callbacks.History): O objeto history retornado por model.fit().
        timings (Dict[str, Any]): Um dicionário contendo os tempos de execução.
        num_train_images (int): Número de imagens de treino usadas.
        num_val_images (int): Número de imagens de validação usadas.
    """
    try:
        final_val_accuracy = max(history.history.get('val_accuracy', [0]))

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("      RELATÓRIO FINAL DA EXECUÇÃO DE TREINO\n")
            f.write("="*60 + "\n\n")

            f.write(f"Projeto: {config.PROJECT_NAME}\n")
            f.write(f"Dataset: {config.DATASET_NAME}\n")
            f.write(f"Timestamp da Execução: {config.TIMESTAMP}\n\n")

            f.write("-" * 60 + "\n")
            f.write("Resumo dos Tempos\n")
            f.write("-" * 60 + "\n")
            f.write(f"Início do Processo: {timings['start_dt']}\n")
            f.write(f"Fim do Processo:    {timings['end_dt']}\n")
            f.write(f"Duração Total: {timings['total_duration']:.2f} segundos\n\n")
            f.write(f"Tempo de Carregamento de Dados: {timings['data_loading_duration']:.2f} segundos\n")
            f.write(f"Tempo de Treino do Modelo (model.fit): {timings['model_training_duration']:.2f} segundos\n\n")

            f.write("-" * 60 + "\n")
            f.write("Resumo do Dataset\n")
            f.write("-" * 60 + "\n")
            f.write(f"Número de Imagens de Treino: {num_train_images}\n")
            f.write(f"Número de Imagens de Validação/Teste: {num_val_images}\n\n")

            f.write("-" * 60 + "\n")
            f.write("Resumo do Treino\n")
            f.write("-" * 60 + "\n")
            f.write(f"Número de Épocas Treinadas: {len(history.history['loss'])}\n")
            f.write(f"Configuração de Épocas (config.py): {config.Training.EPOCHS}\n")
            f.write(f"Tamanho do Lote (Batch Size): {config.Training.BATCH_SIZE}\n\n")

            f.write("-" * 60 + "\n")
            f.write("Métricas Finais (Última Época)\n")
            f.write("-" * 60 + "\n")
            f.write(f"Acurácia de Treino: {history.history['accuracy'][-1]:.4f}\n")
            f.write(f"Perda de Treino:    {history.history['loss'][-1]:.4f}\n")
            f.write(f"Acurácia de Validação: {history.history['val_accuracy'][-1]:.4f}\n")
            f.write(f"Perda de Validação:    {history.history['val_loss'][-1]:.4f}\n\n")
            
            f.write("-" * 60 + "\n")
            f.write("Melhor Performance (Baseado em 'val_accuracy')\n")
            f.write("-" * 60 + "\n")
            f.write(f"Melhor Acurácia de Validação: {final_val_accuracy:.4f}\n")

        print(f"✅ Relatório da execução salvo em: {report_path}")
        
    except Exception as e:
        print(f"Erro ao escrever o relatório: {e}")