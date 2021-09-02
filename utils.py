import threading
import math
import matplotlib.pyplot as plt
import pickle

from tqdm import tqdm

def print_list_img(lista_de_imagens, limite=100, imagens_por_linha:int=6, imagens_por_coluna:int=6):#limite de imaens exibidas
    linha = imagens_por_linha
    coluna = imagens_por_coluna
    numero = math.ceil( len(lista_de_imagens)/ (linha * coluna) )
    k=0
    print('Numero imagens - {}'.format(len(lista_de_imagens)))
    if len(lista_de_imagens[0]) > 1:
#-------------------------------------------------------------------------------        
        if lista_de_imagens[1]!=None:#     classes == None:
            pass
            #classes = range(len(lista_de_imagens))
#-------------------------------------------------------------------------------         
        for j in range(numero):
            plt.figure(figsize=(10,10))
            m = (linha * coluna)    
            if( len(lista_de_imagens) < (linha * coluna) ):
                m = len(lista_de_imagens)
            for i in range(m):
                
                k=k+1
                if k > limite:
                    break
                
                plt.subplot(linha, coluna, i+1)
                plt.xticks([])
                plt.yticks([])
                plt.grid(False)
                plt.imshow(lista_de_imagens[i][0], cmap='gray')
                plt.ylabel("Digito - {}".format( lista_de_imagens[i][1] ), color='white')
                plt.xlabel("{}".format( lista_de_imagens[i][2] ), color='white')
            plt.show()
            lista_de_imagens = lista_de_imagens[i:]
    else:
        for j in range(numero):
            plt.figure(figsize=(10,10))
            m = (linha * coluna)    
            if( len(lista_de_imagens) < (linha * coluna) ):
                m = len(lista_de_imagens)
            for i in range(m):
                
                k=k+1
                if k > limite:
                    break
                
                plt.subplot(linha, coluna, i+1)
                plt.xticks([])
                plt.yticks([])
                plt.grid(False)
                plt.imshow(lista_de_imagens[i], cmap='gray')  
            plt.show()
            lista_de_imagens = lista_de_imagens[i:]

class augmentor(threading.Thread):
    def __init__(self, augmentor_id: int, dados: list, times: int, pipe):
        self.meu_id = augmentor_id
        self.dados = dados
        self.times = times#numero de imagens geradas por imagens dada
        self.pipe = pipe
        self.result = []
        # Atenção essa instrução deve ser chamada para iniciar a thread
        threading.Thread.__init__(self)

    def run(self):
        
        for i in range(len(self.dados)):
            self.result.append(self.dados[i])
            self.result += self.pipe.operar(image=self.dados[i][0], class_img=self.dados[i][1], string_class=self.dados[i][2], vezes=self.times)
        print('Thread {}: Executada com sucesso'.format(self.meu_id))
        
        global result
        result += self.result

def dividir_base(base, n):  # n = numero de itens por thread
    for i in range(0, len(base), n):
        yield base[i:i + n]

result = []

def call_thread(data, pipe_instance, img_per_thread:int=0, image_per_image:int=1, salvar_imagens_gerada:bool=False, caminho:str=None):#Se img_per_thread <= -1 = img_per_thread=len(data)
    
    if img_per_thread <= 0:
        img_per_thread=len(data)
    
    global result
    
    threads = []
    data_array = list(dividir_base(data, img_per_thread))
    
    numbers_of_threads = int( math.ceil( len(data) / img_per_thread ) )

    for i in range(numbers_of_threads):
        new_augmentor = augmentor(augmentor_id = i, dados = data_array[i], times = image_per_image, pipe = pipe_instance)
        threads.append(new_augmentor)
        new_augmentor.start()
    for t in threads:
        t.join()
        
#    for t in threads:
#        result +=  t.result
    
#trecho a ser analisado

#Original
    #return result

    x = result #quando se retorna result direto e se chama 2 vezes no codigo principal o retorno volta com o resultado das execuções anteriores 
    result = []

    if salvar_imagens_gerada:
        #print('Dados ainda não salvos em: ', caminho)    
        pickle_out = open(caminho+".pickle","wb")
        print('Arquivo gravado como: '+caminho+".pickle")
        pickle.dump(data, pickle_out)
        pickle_out.close()

    return x