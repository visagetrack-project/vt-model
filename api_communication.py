# Simulando a obtenção de nomes e estados
import time

from callUnityStates import getNamesFromUnity, getStatesListsCustomFromUnity
import requests

# Processamento de estados
def processaEstados(nomes, estados):
    usuarios = []
    for nome, estado in zip(nomes, estados):
        usuario = {"name": nome, "state": estado}
        # Aqui você adicionaria lógica baseada no estado se necessário
        usuarios.append(usuario)
    return usuarios
def processaHumores(nomes, estados):
    # Assume que 'estados' é uma tupla de listas e pega apenas a primeira lista
    primeira_lista_de_estados = estados
    humores = ["Normal", "Atento", "Surpreso", "Dormindo", "Feliz", "Triste", "Raiva", "Entediado", "Faltou"]
    usuariosHumor = [{"name": nome, "humor": humores[estado]} for nome, estado in zip(nomes, primeira_lista_de_estados)]
    return usuariosHumor

def enviarDados(usuariosHumor):
    url = "http://localhost:8080/updateStates"
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=usuariosHumor, headers=headers)

    if response.status_code == 200:
        print("Dados enviados com sucesso!")
    else:
        print("Erro ao enviar dados:", response.text)
    print(f"Enviando dados para {url} com headers {headers}:\n{usuariosHumor}\n")

def runApi():
    nomes = getNamesFromUnity()
    estados = getStatesListsCustomFromUnity()
    for lista_de_estados in estados:
        usuariosHumor = processaHumores(nomes, lista_de_estados)
        url = "http://localhost:8080/updateStates"
        headers = {"Content-Type": "application/json"}
        enviarDados(usuariosHumor)

        print("Simulação de envio:")
        print(usuariosHumor)
        time.sleep(5)

runApi()