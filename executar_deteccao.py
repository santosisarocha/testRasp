import cv2
from ultralytics import YOLO
import time
from datetime import datetime, date
import schedule
import threading

id_fila = 3

# Globais para controlar o estado da gravação
filmagem = None
gravacao = None
gravando = False
fps = 30  

# Modelo v8 e id da classe de detecção para pessoas
model = YOLO("yolov8n.pt")
id_person = 0

# Tempo armazenado para verificação do horario
tempo_armazenado = time.time()

# Função para exibir situação atual do código
def exibir_mensagem(mensagem):
    print(f"{datetime.now()}: {mensagem}")

# Função para inicio da detecção com detecção e contagem de pessoas
def iniciar_gravacao():
    global filmagem, gravacao, gravando, fps, tempo_armazenado

    if not gravando:
        try:
            filmagem = cv2.VideoCapture(1)
            if not filmagem.isOpened():
                raise Exception("Não foi possível abrir a câmera!")

            w = int(filmagem.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(filmagem.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Nome do arquivo 
            dataHora = datetime.now()
            dia = dataHora.strftime("%d_%m")
            hora = dataHora.strftime("%H_%M")
            nome_mp4 = f"gravacao{dia}_{hora}.mp4"

            gravacao = cv2.VideoWriter(nome_mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

            # Gravação iniciada
            gravando = True
            exibir_mensagem("Início da Gravação")

            while gravando and filmagem.isOpened():
                success, frame = filmagem.read()
                if not success:
                    print('Não foi possível capturar o frame.')
                    
                
                results = model(frame, verbose=False)

                contagem_pessoas = 0

                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        cls = int(box.cls[0])
                        if cls == id_person:  
                            x1, y1, x2, y2 = map(int, box.xyxy[0]) # Coordenadas da caixa de detecção
                            conf = box.conf[0] # Confiança
                            label = f"{model.names[cls]} {conf:.2f}"
                            
                            # Desenhar a caixa de detecção no frame
                            color = (0, 255, 0)  # Verde 
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                            contagem_pessoas += 1
                
                exibir_contagem = f"Pessoas: {contagem_pessoas}"
                cv2.putText(frame, exibir_contagem, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

                cv2.imshow('Detection', frame)
                gravacao.write(frame)

                if cv2.waitKey(1) == 27:
                    parar_gravacao()
                    break
                
                # Verificar e salvar status da detecção com base no tempo que se passou
                tempo_atual = time.time()
                if (tempo_atual - tempo_armazenado >= 60):
                    horario = datetime.now().time()
                    horario_format = horario.strftime("%H:%M:%S")
                    data_registro = date.today().strftime('%Y-%m-%d')
                    
                    # print(f"fila_idfila: '{id_fila}' - dataRegistro: '{date.today()}' - horarioRegistro: '{horario_format}' - quantidade: '{contagem_pessoas}';")

                    with open(f'resgistro_{dia}_{hora}.txt', 'a') as arquivo:
                        arquivo.write(f'fila_idfila: {id_fila}; dataRegistro: {date.today()}; horarioRegistro: {horario_format}; quantidade: {contagem_pessoas};\n')
                        
                    tempo_armazenado = tempo_atual

        except Exception as e:
            exibir_mensagem(f"Erro ao iniciar a gravação: {e}")

def parar_gravacao():
    global filmagem, gravacao, gravando

    if gravando:
        exibir_mensagem("Finalizando a gravação...")

        if filmagem:
            filmagem.release()
        if gravacao:
            gravacao.release()
            cv2.destroyAllWindows()

        # Gravação finalizada
        gravando = False
        filmagem = None
        gravacao = None
        exibir_mensagem("Gravação finalizada!")

# Função para rodar a gravação em uma thread separada
def iniciar_gravacao_thread():
    thread = threading.Thread(target=iniciar_gravacao)
    thread.start()

def agendar_gravacao():
    # Almoço
    schedule.every().day.at("07:42").do(iniciar_gravacao_thread)
    schedule.every().day.at("07:46").do(parar_gravacao)
    # Jantar
    schedule.every().day.at("17:25").do(iniciar_gravacao_thread)
    schedule.every().day.at("20:05").do(parar_gravacao)
    # Ceia
    schedule.every().day.at("01:15").do(iniciar_gravacao_thread)
    schedule.every().day.at("02:35").do(parar_gravacao)

if __name__ == "__main__":
    agendar_gravacao()

    while True:
        schedule.run_pending()
        time.sleep(1)
