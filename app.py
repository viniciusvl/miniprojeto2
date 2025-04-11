import streamlit as st
import numpy as np
import librosa
import joblib
import tensorflow as tf
import os
import soundfile as sf
import tempfile
import pandas as pd

# --------------------------------- PARTE 1: EXTRAIR FEATURES --------------------------------- #

# Carregar o modelo e o scaler
MODEL_PATH = "models/audio_emotions_modelo.keras" 
SCALER_PATH = "models/scaler.pkl"

model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Lista de emoções
EMOTIONS = ["angry", "calm", "disgust", "fear",
            "happy", "neutral", "sad", "surprise"]

# Função para extrair features
def extract_features(audio_path):
    data, sr = librosa.load(audio_path, sr=16000, mono=True)
    features = []

    # Zero Crossing Rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data, pad=False), axis=1)
    features.extend(zcr)

    # Chroma STFT
    chroma_stft = np.mean(librosa.feature.chroma_stft(y=data, sr=sr), axis=1)
    features.extend(chroma_stft)

    # MFCCs
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sr), axis=1) 
    features.extend(mfcc)

    # RMS
    rms = np.mean(librosa.feature.rms(y=data, frame_length=2048, hop_length=512), axis=1)
    features.extend(rms)

    # Mel Spectrogram
    mel = np.mean(librosa.feature.melspectrogram(y=data), axis=1)
    features.extend(mel)

    # Garantir que tenha exatamente 162 features (ou truncar/zerar)
    target_length = 162
    if len(features) < target_length:
        features.extend([0] * (target_length - len(features)))
    elif len(features) > target_length:
        features = features[:target_length]

    return np.array(features).reshape(1, -1) # transforma o array em uma matriz 2D (1 linha e 162 colunas)


# --------------------------------- PARTE 2: STREAMLIT --------------------------------- #

# Configuração do app Streamlit (Título e descrição)
st.title("🔊 Detector de emoções em áudios")

# Upload de arquivo de áudio (wav, mp3, ogg)
uploaded_file = st.file_uploader(
    "Selecione ou arraste um arquivo de áudio... 👇", type=["wav", "mp3", "ogg"])

if uploaded_file is not None: # se o usuário enviar algo
    # Salvar temporariamente o áudio

    # este método abre um arquivo binário em modo leitura e escrita
    audio_temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False) # cria um arquivo de extensão .wav temporário
    audio_temp.write(uploaded_file.getvalue()) # escreve o conteúdo do audio enviado no audio temporario
    path = audio_temp.name # retorna o caminho de audio
    audio_temp.close()

    # Reproduzir o áudio enviado
    st.audio(uploaded_file)

    # Extrair features
    features = extract_features(path)

    # Normalizar os dados com o scaler treinado
    features = scaler.transform(features) # scaler já foi treinado com .fit()

    # Ajustar formato para o modelo

    # O modelo foi treinado com dados do tipo 3D, então precisamos alterar a dimensão 
    features = np.expand_dims(features, axis=2) # os dados são reorganizados, NÃO ALTERADOS

    # Fazer a predição
    prediction = model.predict(features)

    # prediction é um vetor que possui valores de probabilidade para cada emoção: [0.11, 0.02, 0.88, 0.32, 0.03, 0.04, 0.13, 0.34]
    # então, devemos tomar nesse vetor o ELEMETNO COM O MAIOR numero entre eles  
    # np.argmax retorna o maior elemento de um array 
    # prediction[0] retorna a primeira linha de prediction, que é um vetor de probabilidades
    # como prediction e emotion tem indices respectivos, jogamos o indice do maior de prediction em emotion

    emotion = EMOTIONS[np.argmax(prediction[0])]

    # Exibir o resultado
    if (emotion == 'angry'):
        st.success(f"#### 💡Emoção detectada: {emotion} 😠")
    
    elif (emotion == 'calm'):
        st.success(f"#### 💡Emoção detectada: {emotion} 😌")
    
    elif (emotion == 'disgust'):
        st.success(f"#### 💡Emoção detectada: {emotion} 🤮")
    
    elif (emotion == 'fear'):
        st.success(f"#### 💡Emoção detectada: {emotion} 😨")
    
    elif (emotion == 'happy'):
        st.success(f"#### 💡Emoção detectada: {emotion} 😃")
    
    elif (emotion == 'neutral'):
        st.success(f"#### 💡Emoção detectada: {emotion} 😐")
    
    elif (emotion == 'sad'):
        st.success(f"#### 💡Emoção detectada: {emotion} 😭")

    elif (emotion == 'surprise'):
        st.success(f"#### 💡Emoção detectada: {emotion} 😯")

    # Exibir probabilidades (gráfico de barras)
    d = pd.DataFrame(prediction[0], columns=['Predction'])
    b = pd.DataFrame(EMOTIONS, columns=['Emotion'])
    data_bar = pd.concat([d, b], axis = 1)

    st.bar_chart(data_bar, x='Emotion', y='Predction', y_label='Probability', height=600, width=600)

    st.write('**Probabilidades:** ')
    for emotion, prob in zip(EMOTIONS, prediction[0]):
        st.write(f"{emotion}: {prob*100:.2f}%")

    # Remover o arquivo temporário
    os.remove(path)