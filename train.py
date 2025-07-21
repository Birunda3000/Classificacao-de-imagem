# -*- coding: utf-8 -*-
"""
Script Principal para Treino do Modelo (Versão Final Comentada)

Este script é o orquestrador central do projeto. Ele executa o fluxo
completo de treino, desde o carregamento dos dados até o salvamento dos
resultados finais e do relatório da execução.
"""
import time
from datetime import datetime
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# --------------------------------------------------------------------------
# 1. IMPORTAÇÃO DOS MÓDULOS DO PROJETO
# --------------------------------------------------------------------------
# Importa todos os nossos módulos customizados. Cada um tem uma
# responsabilidade específica, mantendo este script limpo e organizado.
from src import (
    config, 
    data_loader, 
    model_architectures, 
    augmentation, 
    callbacks, 
    visualization,
    reporting
)

# Configura o TensorFlow para usar Mixed Precision e XLA (JIT Compilation)
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'

# --------------------------------------------------------------------------
# 2. FUNÇÃO PRINCIPAL DE TREINO
# --------------------------------------------------------------------------
def main():
    """Função principal que executa todo o fluxo de treino."""
    
    # --- INICIALIZAÇÃO E MEDIÇÃO DE TEMPO ---
    # Guarda o tempo de início para calcular a duração total no final.
    process_start_time = time.time()
    process_start_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print("="*50); print(f"INICIANDO TREINO: {config.PROJECT_NAME}"); print("="*50)

    # --- ETAPA 1: CARREGAMENTO DOS DADOS ---
    # Delega a tarefa de carregar e pré-processar os dados para o data_loader.
    # Ele retorna os pipelines de dados e o número de imagens encontradas.
    print("\n[ETAPA 1/5] Carregando datasets...")
    data_loading_start_time = time.time()
    train_ds, num_train_images = data_loader.create_full_pipeline(config.PATH_TRAIN, config.Training.BATCH_SIZE, is_training=True)
    validation_ds, num_val_images = data_loader.create_full_pipeline(config.PATH_TEST, config.Training.BATCH_SIZE, is_training=False)
    data_loading_duration = time.time() - data_loading_start_time

    # --- ETAPA 2: CONFIGURAÇÃO DO AUMENTO DE DADOS (AUGMENTATION) ---
    # Cria a camada de aumentação e a aplica ao pipeline de treino.
    # Se a aumentação estiver desativada no config, estas funções não farão nada.
    print("\n[ETAPA 2/5] Configurando aumento de dados...")
    augmentation_model = augmentation.build_augmentation_layer()
    train_ds_augmented = augmentation.apply_augmentation(train_ds, augmentation_model)

    # --- ETAPA 3: CONSTRUÇÃO DA ARQUITETURA DO MODELO ---
    # Chama a função do nosso módulo de arquiteturas para construir o modelo.
    # CORREÇÃO: O nome do módulo é 'model_architectures' e a função 'build_cnn'.
    print("\n[ETAPA 3/5] Construindo o modelo...")
    model = model_architectures.build_cnn()

    # --- ETAPA 4: COMPILAÇÃO DO MODELO ---
    # Define o otimizador, a função de perda e as métricas para o treino.
    print("\n[ETAPA 4/5] Compilando o modelo...")
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.Training.LEARNING_RATE)
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    model.compile(optimizer=optimizer, loss=config.Training.LOSS, metrics=['accuracy'])

    # --- ETAPA 5: EXECUÇÃO DO TREINO ---
    # Pega os callbacks (EarlyStopping, ModelCheckpoint) e inicia o 'model.fit'.
    print("\n[ETAPA 5/5] Iniciando o treino do modelo...")
    training_callbacks = callbacks.get_default_callbacks()
    training_start_time = time.time()
    history = model.fit(train_ds_augmented, epochs=config.Training.EPOCHS, validation_data=validation_ds, callbacks=training_callbacks, verbose=1)
    model_training_duration = time.time() - training_start_time

    # --------------------------------------------------------------------------
    # 3. FINALIZAÇÃO E GERAÇÃO DE RELATÓRIOS
    # --------------------------------------------------------------------------
    # Esta secção só é executada se o treino for concluído com sucesso.
    print("\nTreino concluído com sucesso!")
    process_end_time = time.time()
    process_end_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # --- RENOMEAÇÃO DA PASTA DA EXECUÇÃO ---
    # Pega a pasta temporária (_tmp_...) e a renomeia para o nome final,
    # incluindo a melhor acurácia de validação alcançada.
    final_val_accuracy = max(history.history.get('val_accuracy', [0]))
    new_run_dir_name = f"{config.PROJECT_NAME}___val_acc={final_val_accuracy:.4f}___{config.TIMESTAMP}"
    final_run_dir = config.Paths.OUTPUT_DIR / new_run_dir_name
    try:
        config.Paths.TMP_RUN_DIR.rename(final_run_dir)
        print(f"✅ Diretório da execução finalizado e salvo como: {final_run_dir.name}")
    except Exception as e:
        print(f"Erro ao finalizar o diretório da execução: {e}")
        final_run_dir = config.Paths.TMP_RUN_DIR
    
    # --- SALVAMENTO DOS ARTEFACTOS ---
    # Salva o gráfico do histórico de treino na pasta final da execução.
    try:
        fig = visualization.plot_history(history)
        history_plot_path = final_run_dir / "training_history.png"
        fig.savefig(history_plot_path)
        plt.close(fig) # Fecha a figura para libertar memória
        print(f"Gráfico do histórico salvo em: {history_plot_path}")
    except Exception as e:
        print(f"Não foi possível salvar o gráfico do histórico: {e}")

    # --- GERAÇÃO DO RELATÓRIO DE TEXTO ---
    # Coleta todas as métricas e tempos e chama o módulo de relatório para
    # escrever o ficheiro 'report.txt' na pasta final.
    timings = {
        "start_dt": process_start_dt, "end_dt": process_end_dt,
        "total_duration": process_end_time - process_start_time,
        "data_loading_duration": data_loading_duration,
        "model_training_duration": model_training_duration,
    }
    report_path = final_run_dir / "report.txt"
    reporting.save_training_report(
        report_path, 
        history, 
        timings, 
        num_train_images, 
        num_val_images
    )

# --------------------------------------------------------------------------
# 4. PONTO DE ENTRADA DO SCRIPT
# --------------------------------------------------------------------------
if __name__ == '__main__':
    main()