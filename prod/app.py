# app.py
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision import transforms
from facenet_pytorch import MTCNN
import pandas as pd
import numpy as np
import os
import urllib.request
import av
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from model import resnet18_model

# ========= MODELO Y CONFIG =========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet18_model()
model_path = "dev/modelo_entrenado.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval().to(device)

# ========= CLASES Y COLORES =========
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
emotion_colors = {
    'Angry': '#E63946', 'Disgust': '#6A994E', 'Fear': '#9A8C98',
    'Happy': '#F4D35E', 'Sad': '#457B9D', 'Surprise': '#F9844A', 'Neutral': '#A8A7A7'
}

# ========= MTCNN Y TRANSFORM =========
mtcnn = MTCNN(keep_all=True, device=device, post_process=True)
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(), 
])
# ========= TÍTULO =========
st.markdown("""
<h1 style='text-align:center; color:#3B4252; font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;'>
🎭 Detector de emociones faciales 😄😠😢
</h1>""", unsafe_allow_html=True)

# ========= SUBIR O SACAR FOTO =========
# Widgets para cargar imagen o sacar foto
uploaded_file = st.file_uploader("Subí una foto grupal (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])
photo = st.camera_input("O tomá una foto con la cámara 📷")

# Imagen prioritaria
image = None
if photo is not None:
    image = Image.open(photo).convert("RGB")
elif uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

# ========= ANÁLISIS DE FOTO =========
if image is not None:
    if uploaded_file is not None:  # Solo mostrar si la imagen fue subida, no tomada
        st.image(image, caption="Imagen original", use_container_width=True)
    boxes, _ = mtcnn.detect(image)

    if boxes is None:
        st.warning("No se detectaron caras. 🤔 Intentá con otra foto.")
    else:
        boxes = boxes[boxes[:, 0].argsort()]
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("arial.ttf", size=150)
        except:
            font = ImageFont.load_default()

        resultados = []

        for i, box in enumerate(boxes):
            face = image.crop(box).resize((48, 48))
            input_tensor = transform(face).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)[0]
                pred = torch.argmax(probs).item()
                emotion = class_names[pred]
                topk = torch.topk(probs, 3)
                top_emotions = [(class_names[idx], float(p) * 100) for idx, p in zip(topk.indices, topk.values)]

            color = emotion_colors.get(emotion, "red")
            draw.rectangle(box.tolist(), outline=color, width=5)
            etiqueta = f"Persona #{i+1}: {emotion} ({top_emotions[0][1]:.1f}%)"
            draw.text((box[0], box[1] - 30), etiqueta, fill=color, font=font)
            resultados.append((i+1, emotion, color, top_emotions))

        st.image(image, caption="Emociones detectadas 🎉", use_container_width=True)

        st.markdown("## Resultados detallados por persona")

        for persona_num, emocion_pred, color, emociones_top in resultados:
            st.markdown(f"### 🧠 Persona #{persona_num}: <span style='color:{color};'>{emocion_pred}</span>", unsafe_allow_html=True)

            # Mostrar recorte de la cara
            box = boxes[persona_num - 1]
            face_crop = image.crop(box)
            st.image(face_crop, width=150, caption=f"Cara #{persona_num}")

            # Mostrar tabla de emociones
            df_emociones = pd.DataFrame(emociones_top, columns=["Emoción", "Confianza (%)"])
            df_emociones["Confianza (%)"] = df_emociones["Confianza (%)"].map(lambda x: f"{x:.1f}%")
            st.table(df_emociones)

# ========= DETECCIÓN EN TIEMPO REAL =========
st.markdown("---")
st.markdown("## 🎥 Detección de emociones en tiempo real")

class EmotionDetector(VideoTransformerBase):
    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        boxes, _ = mtcnn.detect(image_pil)

        if boxes is not None:
            draw = ImageDraw.Draw(image_pil)
            try:
                font = ImageFont.truetype("arial.ttf", size=150)
            except:
                font = ImageFont.load_default()

            for box in boxes:
                face = image_pil.crop(box).resize((48, 48))
                input_tensor = transform(face).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(input_tensor)
                    probs = torch.nn.functional.softmax(output, dim=1)[0]
                    pred = torch.argmax(probs).item()
                    emotion = class_names[pred]
                    top_emotion_prob = float(probs[pred]) * 100

                color = emotion_colors.get(emotion, "red")
                draw.rectangle(box.tolist(), outline=color, width=3)
                draw.text((box[0], box[1] - 20), f"{emotion} ({top_emotion_prob:.1f}%)", fill=color, font=font)

            image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

        return image

webrtc_streamer(
    key="realtime",
    video_processor_factory=EmotionDetector,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# ========= FOOTER =========
st.markdown("---")
st.markdown("<p style='text-align:center; color:gray;'>💡 Pro tip: ¡Sonreí para que te detecte 'Happy'! 😄</p>", unsafe_allow_html=True)
