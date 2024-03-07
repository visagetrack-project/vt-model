# Simulando a obtenção de nomes e estados
import time

from analysis import createStatesAnalysis
from callUnityStates import getNamesFromUnity, getStatesListsCustomFromUnity
import requests


def adicionar_filepath(lista_de_listas):
    total_usuarios = sum(len(lista) for lista in lista_de_listas)  # Contagem total de usuários
    i = 10  # Começa com 10 e incrementa de 10 em 10 a cada 10 usuários processados
    contador_usuarios = 0  # Conta os usuários para determinar quando incrementar `i`

    for lista in lista_de_listas:
        contador_usuarios += 1
        for item in lista:
            item['filepath'] = f'/packages/graph/regressao_{i}.png'
            if contador_usuarios == 10:
                i+=10
                contador_usuarios = 0

    return lista_de_listas


# Processamento de estados
def processaEstados(nomes, estados):
    usuarios = []
    for nome, estado in zip(nomes, estados):
        usuario = {"name": nome, "state": estado}
        # Aqui você adicionaria lógica baseada no estado se necessário
        usuarios.append(usuario)
    return usuarios


def processaHumores(nomes, estados):
    contagem_faltas = {nome: 0 for nome in nomes}  # Inicializa contagem de faltas para todos os nomes
    humores = ["Normal", "Atento", "Surpreso", "Dormindo", "Feliz", "Triste", "Raiva", "Entediado", "Faltou"]
    # Lista para armazenar os resultados acumulados, incluindo evolução da contagem de faltas
    resultados = []
    # Itera sobre cada lista de estados na tupla de estados
    for indice, lista_de_estados in enumerate(estados):
        # Lista temporária para armazenar os usuários e seus humores para a iteração atual
        usuariosHumor = []
        for nome, estado in zip(nomes, lista_de_estados):
            humor_atual = humores[estado]
            # Atualiza a contagem de faltas para o usuário
            if estado == 8:  # Se o estado corresponde a "Faltou"
                contagem_faltas[nome] += 1
            # Adiciona o usuário, seu humor e a contagem de faltas atual à lista temporária
            usuariosHumor.append({"name": nome, "humor": humor_atual, "faltas": contagem_faltas[nome]})
        # Adiciona os resultados da iteração atual à lista de resultados acumulados
        resultados.append(usuariosHumor)
    resultados = adicionar_filepath(resultados)
    return resultados


def enviarDados(usuario):
    url = "http://localhost:8080/updateStates"
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=usuario, headers=headers)

    if response.status_code == 200:
        print("Dados enviados com sucesso!")
    else:
        print("Erro ao enviar dados:", response.text)
    print(f"Enviando dados para {url} com headers {headers}:\n{usuario}\n")


def runApi():
    nomes = getNamesFromUnity()
    estados = getStatesListsCustomFromUnity()
    usuariosHumor = processaHumores(nomes, estados)
    createStatesAnalysis(estados)

    print("Aguarde 30 Segundos Para Ligar a api de GO")
    # Envia os dados de cada usuá   rio individualmente
    for usuario in usuariosHumor:
        enviarDados(usuario)
        time.sleep(1)  # Espera 5 segundos antes de enviar o próximo

runApi()


