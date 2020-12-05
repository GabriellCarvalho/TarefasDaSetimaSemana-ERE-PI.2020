# -*- coding: utf-8 -*-

import cv2
import matplotlib.pyplot as plt
import os
import dlib
import youtube_dl
import os.path
import face_recognition
import pickle
from tqdm import tqdm

# Função que baixa video do youtube
def download_youtube_video(youtube_url, video_filename):
    youtube_url = youtube_url.strip() 
    ydl_opts = {'outtmpl': video_filename}

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])

# Função que gera uma pasta contendo as faces        
def generateFolderWithFaces(folderName, videoFilename, thresholdFaces = 5, stepFrame = 15):
    detectedFacesCounter = 0
    faces = []
    frameCounter = 0

    #carrega vídeo
    videoCapture = cv2.VideoCapture(videoFilename)
    while 1:
        #contabiliza frames
        frameCounter += 1
        
        #pega o próximo frame
        success, frame = videoCapture.read()
        
        #acabou o vídeo ou detectou o limite de faces?
        if success == False or detectedFacesCounter >= thresholdFaces:
            return faces

        if frameCounter % stepFrame != 0:
            continue

        # detecta a face, o 1 indica superamostragem, pra achar faces mais facilmente
        detections = detector(frame, 1)
        for i, detection in enumerate(detections):
            
            #salvar região das faces em uma pasta
            x, y = detection.left(), detection.top()
            w, h = detection.right() - detection.left(), detection.bottom() - detection.top()
            faceCrop = frame[y:y + h, x:x + w]
            img_path = folderName + "/" + str(frameCounter) + ".jpg"
            cv2.imwrite(img_path, faceCrop)

            #essa parte é só pra adicionar retângulo ao redor da face e marcações, pra exibir bonitinho depois
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 4)
            landmarks = predictor(frame, detection)
            for i in range(0, 68): 
                cv2.circle(frame, (landmarks.part(i).x , landmarks.part(i).y), 4, (0, 0, 255), -1)      
            faces.append(frame)
        
        #total de faces detectadas até o momento
        detectedFacesCounter += len(detections)
        
    videoCapture.release()

# Função que gera codificações de faces
def generateEncodings(folderName, labelName, knownEncodings, knownNames):
    for filename in os.listdir(folderName):
        img = face_recognition.load_image_file(folderName + filename)
        boxes = face_recognition.face_locations(img, model = 'cnn')
        encodings = face_recognition.face_encodings(img, boxes)

        for encoding in encodings:
            knownEncodings.append(encoding)
            knownNames.append(labelName)

#### 
youtube_url = "https://www.youtube.com/watch?v=ZZXEMrzzsTU"
video_filename = "leon.mp4"

if os.path.isfile(video_filename) == False:
    download_youtube_video(youtube_url, video_filename)

folder_name = "Leon"
if os.path.isdir(folder_name) == False:
    os.mkdir(folder_name)

shape_predictor_filename = "shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_filename)

exampleFaces = generateFolderWithFaces("Leon/", "leon.mp4")

#mostra algumas faces detectadas de exemplo
fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize=(16, 16))
i = 0
for y in range(0, 2):
    for x in range(0, 2):
        axes[x, y].set_title('Face ' + str(i), fontsize = 14)
        exampleFaces[i] = cv2.cvtColor(exampleFaces[i], cv2.COLOR_BGR2RGB)
        axes[x, y].imshow(exampleFaces[i])
        i += 1
####
folder_name = "Nilce"
if os.path.isdir(folder_name) == False:
    os.mkdir(folder_name)

youtube_url = "https://www.youtube.com/watch?v=OvBfj6gG1yI"
video_filename = "nilce.mp4"

if os.path.isfile(video_filename) == False:
    download_youtube_video(youtube_url, video_filename)

exampleFaces = generateFolderWithFaces("Nilce/", "nilce.mp4")

#mostra algumas faces detectadas de exemplo
fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize=(16, 16))
i = 0
for y in range(0, 2):
    for x in range(0, 2):
        axes[x, y].set_title('Face ' + str(i), fontsize = 14)
        exampleFaces[i] = cv2.cvtColor(exampleFaces[i], cv2.COLOR_BGR2RGB)
        axes[x, y].imshow(exampleFaces[i])
        i += 1
###

knownEncodings = []
knownNames = []

folderName = "Leon/"
labelName = "Leon"
generateEncodings(folderName, labelName, knownEncodings, knownNames)
       
folderName = "Nilce/"
labelName = "Nilce"
generateEncodings(folderName, labelName, knownEncodings, knownNames)

data_encoding = {"encodings": knownEncodings, "names": knownNames}

f = open("face_encodings", "wb")
f.write(pickle.dumps(data_encoding))
f.close()

youtube_url = "https://www.youtube.com/watch?v=luqwhG2CiHU"
video_filename = "leon_e_nilce.mp4"

if os.path.isfile(video_filename) == False:
    download_youtube_video(youtube_url, video_filename)

#carrega arquivo binário contendo faces codificadas
data_encoding = pickle.loads(open("face_encodings", "rb").read())

#carrega vídeo do disco
inp = cv2.VideoCapture(video_filename)

#set contendo as possíveis pessoas reconhecidas
unique_names = set(data_encoding["names"])

#gerador de vídeo contendo saída com faces reconhecidas
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

#gera reconhecimento em vídeo para os 60 primeiros frames
for i in tqdm(range(0, 60)):
    
    #para cada frame
    success, frame = inp.read()

    #acabou o vídeo?
    if success == False:
        break
        
    #converte frame de formato BGR (OpenCV) para RGB (face_recognition)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    boxes = face_recognition.face_locations(frame, model = 'cnn')
    encodings = face_recognition.face_encodings(frame, boxes)
    
    names = []

    #para cada codificação de faces encontrada
    for encoding in encodings:
        matches = face_recognition.compare_faces(data_encoding["encodings"], encoding)
        
        #retorna o identificador da lista das faces da base que "batem" com a codificação verificada
        matchesId = [i for i, value in enumerate(matches) if value == True]
        
        #faz uma espécie de "votação": quem tiver mais codificações "próximas" das faces treinadas na base "ganha"
        counts = {}
        for name in unique_names:
            counts[name] = 0  
        for i in matchesId:
            name = data_encoding["names"][i]
            counts[name] += 1
        name = max(counts, key = counts.get)
        names.append(name)

    #desenha o retângulo e escreve o nome da pessoa no frame
    for ((top, right, bottom, left), name) in zip(boxes, names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 4)
        cv2.putText(frame, name, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 4)
    
    #converte o frame de volta pro formato do OpenCV (BGR)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    #escreve o frame no arquivo de vídeo
    out.write(frame)
    
inp.release()
out.release()