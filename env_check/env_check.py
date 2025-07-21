# -*- coding: utf-8 -*-
"""
Script de Diagnóstico Avançado (Versão 3.1 - Verificações Explícitas)
"""
import os
import sys
import platform
import time
import numpy as np
import tensorflow as tf
import psutil

# (copie e cole todas as funções auxiliares da versão anterior aqui)
# ...
def formatar_bytes(b):
    if b < 1024: return f"{b} Bytes"
    elif b < 1024**2: return f"{b/1024:.2f} KB"
    elif b < 1024**3: return f"{b/1024**2:.2f} MB"
    elif b < 1024**4: return f"{b/1024**3:.2f} GB"
    else: return f"{b/1024**4:.2f} TB"
def imprimir_cabecalho(titulo):
    print("\n" + "=" * 80); print(f"### {titulo.upper()} ###"); print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}"); print("-" * 80)
def verificar_execucao_docker():
    print("\n[INFO] Verificando ambiente de execução...")
    is_docker = os.path.exists('/.dockerenv')
    print(f"  - [{'SUCESSO' if is_docker else 'INFO'}] Script a ser executado {'dentro de um container Docker.' if is_docker else 'num ambiente não-Docker.'}")
    return is_docker
def verificar_recursos_hardware():
    print("\n[INFO] Verificando recursos de hardware...")
    print("  - CPU:"); print(f"    - Modelo: {platform.processor()}"); print(f"    - Núcleos Físicos: {psutil.cpu_count(logical=False)}"); print(f"    - Núcleos Lógicos (Threads): {psutil.cpu_count(logical=True)}")
    try:
        freq = psutil.cpu_freq(); print(f"    - Frequência Atual: {freq.current:.2f} Mhz"); print(f"    - Frequência Máxima: {freq.max:.2f} Mhz")
    except (NotImplementedError, PermissionError): print("    - Frequência da CPU não pôde ser determinada neste SO.")
    print("  - Memória RAM:"); mem = psutil.virtual_memory(); print(f"    - Total: {formatar_bytes(mem.total)}"); print(f"    - Disponível: {formatar_bytes(mem.available)}"); print(f"    - Usada: {formatar_bytes(mem.used)} ({mem.percent}%)")
    print("  - Disco (Partição Raiz '/'):"); disk = psutil.disk_usage('/'); print(f"    - Total: {formatar_bytes(disk.total)}"); print(f"    - Usado: {formatar_bytes(disk.used)}"); print(f"    - Livre: {formatar_bytes(disk.free)} ({disk.percent}%)")
def verificar_versoes_sistema():
    print("\n[INFO] Verificando versões do sistema e bibliotecas..."); print(f"  - Versão do TensorFlow: {tf.__version__}"); print(f"  - Versão do Python: {sys.version.split()[0]}"); print(f"  - Sistema Operacional: {platform.system()} {platform.release()}")
def obter_info_build_tf():
    try:
        build_info = tf.sysconfig.get_build_info(); print(f"    - Versão CUDA (compilada com TF): {build_info.get('cuda_version', 'N/A')}"); print(f"    - Versão cuDNN (compilada com TF): {build_info.get('cudnn_version', 'N/A')}")
    except Exception as e: print(f"    - [ERRO] Não foi possível obter informações de build do TF: {e}")
def verificar_dispositivos_gpu():
    print("\n[INFO] Procurando por dispositivos GPU..."); gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"  - [SUCESSO] {len(gpus)} GPU(s) encontrada(s):")
        for i, gpu in enumerate(gpus): print(f"    - GPU[{i}]: Nome: {gpu.name}, Tipo: {gpu.device_type}")
        obter_info_build_tf()
    else: print("  - [AVISO] Nenhuma GPU foi encontrada pelo TensorFlow.")
    return gpus
def executar_teste_performance(nome_dispositivo, tamanho_matriz):
    print(f"\n[INFO] Iniciando teste de performance em: {nome_dispositivo} (Matriz {tamanho_matriz}x{tamanho_matriz})")
    try:
        with tf.device(nome_dispositivo):
            matriz_a = tf.random.uniform([tamanho_matriz, tamanho_matriz], dtype=tf.float32); matriz_b = tf.random.uniform([tamanho_matriz, tamanho_matriz], dtype=tf.float32)
            _ = tf.matmul(matriz_a, matriz_b)
            inicio = time.perf_counter(); resultado = tf.matmul(matriz_a, matriz_b); np.asarray(resultado); fim = time.perf_counter()
            duracao = fim - inicio; print(f"  - [RESULTADO] Tempo de execução em {nome_dispositivo}: {duracao:.4f} segundos."); return duracao
    except Exception as e: print(f"  - [ERRO] Falha ao executar o teste em {nome_dispositivo}: {e}"); return float('inf')
def imprimir_sumario_final(is_docker, gpus, tempo_gpu, tempo_cpu):
    imprimir_cabecalho("Diagnóstico Final")
    if is_docker: print("✔️ Ambiente de Execução: Container Docker.")
    else: print("✔️ Ambiente de Execução: Sistema Host (Não-Docker).")
    if gpus and tempo_gpu < float('inf'):
        print("✔️ Aceleração por GPU: Ativa e Testada.")
        if tempo_cpu < float('inf'):
            print(f"   - Tempo de execução na GPU: {tempo_gpu:.4f}s"); print(f"   - Tempo de execução na CPU: {tempo_cpu:.4f}s")
            if tempo_gpu < tempo_cpu:
                speedup = tempo_cpu / tempo_gpu; print(f"   - Otimização: A GPU é aproximadamente {speedup:.2f}x mais rápida que a CPU."); print("\n   >>> [AVALIAÇÃO GERAL]: SUCESSO! Ambiente otimizado para Deep Learning. <<<")
            else: print("\n   >>> [AVALIAÇÃO GERAL]: ATENÇÃO! A GPU está ativa, mas foi mais lenta que a CPU.")
    else: print("❌ Aceleração por GPU: INATIVA ou com falhas."); print("\n   >>> [AVALIAÇÃO GERAL]: ATENÇÃO! O ambiente não está a usar a aceleração por GPU. <<<")
    print("=" * 80)
# --- Fim das funções auxiliares ---

def verificar_otimizacoes_avancadas():
    """Verifica o status de otimizações como XLA (JIT) e Mixed Precision."""
    print("\n[INFO] Verificando otimizações avançadas do TensorFlow...")
    
    # 1. Verificação do XLA (JIT Compilation)
    # Forçamos a ativação para verificar se causa algum erro
    try:
        os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'
        # Criamos uma função simples para testar a compilação JIT
        @tf.function(jit_compile=True)
        def test_xla(x, y):
            return tf.add(x, y)
        
        test_xla(tf.constant(1.0), tf.constant(2.0))
        print(f"  - [SUCESSO] XLA (JIT Compilation) foi ativado e testado com sucesso.")
    except Exception as e:
        print(f"  - [AVISO] Não foi possível ativar e testar o XLA (JIT Compilation): {e}")

    # 2. Verificação de Mixed Precision
    try:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        # Verificação mais robusta
        if tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
            print("  - [SUCESSO] Mixed Precision (mixed_float16) foi ativada com sucesso.")
        else:
            print("  - [AVISO] Tentativa de ativar Mixed Precision não resultou na política global esperada.")
    except Exception as e:
        print(f"  - [AVISO] Não foi possível ativar Mixed Precision: {e}")

def main():
    """Função principal para orquestrar a execução do script de diagnóstico."""
    imprimir_cabecalho("Início do Diagnóstico Avançado do Ambiente")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    TAMANHO_DA_MATRIZ_DE_TESTE = 8192
    
    verificar_execucao_docker()
    verificar_recursos_hardware()
    verificar_versoes_sistema()
    verificar_otimizacoes_avancadas() # <-- Verificações melhoradas
    gpus_encontradas = verificar_dispositivos_gpu()
    
    tempo_cpu = executar_teste_performance('/CPU:0', tamanho_matriz=TAMANHO_DA_MATRIZ_DE_TESTE)
    tempo_gpu = float('inf')
    if gpus_encontradas:
        tempo_gpu = executar_teste_performance('/GPU:0', tamanho_matriz=TAMANHO_DA_MATRIZ_DE_TESTE)

    imprimir_sumario_final(verificar_execucao_docker(), gpus_encontradas, tempo_gpu, tempo_cpu)

if __name__ == "__main__":
    main()