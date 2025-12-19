# app.py

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

# ----------------------------
# Config page
# ----------------------------
st.set_page_config(
    page_title="Segmentation & Recoloration intelligente",
    layout="wide"
)

# ----------------------------
# Mode clair / sombre
# ----------------------------
theme = st.sidebar.radio("🌗 Choisir le thème", ["Clair", "Sombre"])

if theme == "Sombre":
    bg_color = "#0D1117"
    text_color = "#F5F5F5"
    sidebar_color = "#011627"
    button_color = "#1E90FF"
else:
    bg_color = "#FFFFFF"
    text_color = "#000000"
    sidebar_color = "#0B3D91"
    button_color = "#007ACC"

st.markdown(f"""
<style>
/* Background et textes */
.stApp {{background-color: {bg_color}; color: {text_color};}}
.stMarkdown p, .stText, .stHeader, .stSubheader {{color: {text_color};}}
.stButton>button {{background-color: {button_color}; color: white; border-radius: 10px; padding: 8px; font-weight:bold;}}
.stSidebar {{background-color: {sidebar_color}; color: white;}}
.stSidebar .stSelectbox>div>div>div {{color: white; background-color: {sidebar_color};}}
hr {{border:1px solid #ccc;}}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Header
# ----------------------------
st.title("🎨 PiColor")
st.write("💡 Sélectionnez une couleur et recolorez facilement vos images !")

# ----------------------------
# Table des couleurs
# ----------------------------
COLORS = {
    "Rouge": {"ranges":[([0,100,50],[10,255,255]),([170,100,50],[179,255,255])],"hue":0},
    "Orange": {"ranges":[([11,100,100],[19,255,255])],"hue":15},
    "Jaune": {"ranges":[([20,80,150],[35,255,255])],"hue":25},
    "Vert clair": {"ranges":[([36,50,50],[60,255,255])],"hue":50},
    "Vert foncé": {"ranges":[([61,100,50],[85,255,255])],"hue":70},
    "Bleu clair": {"ranges":[([86,50,50],[105,255,255])],"hue":95},
    "Bleu foncé": {"ranges":[([106,100,50],[130,255,255])],"hue":120},
    "Violet": {"ranges":[([131,50,50],[155,255,255])],"hue":140},
    "Rose": {"ranges":[([156,50,50],[169,255,255])],"hue":165},
    "Blanc": {"ranges":[([0,0,200],[179,40,255])],"hue":0},
    "Noir": {"ranges":[([0,0,0],[179,255,40])],"hue":0},
    "Gris": {"ranges":[([0,0,50],[179,30,200])],"hue":0},
}

# ----------------------------
# Upload image
# ----------------------------
uploaded_file = st.file_uploader("📤 Téléchargez votre image", type=["jpg","jpeg","png"])

if uploaded_file:
    image_pil = Image.open(uploaded_file)
    image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    st.subheader("📷 Image originale")
    st.image(image_pil, use_column_width=True)

    st.sidebar.header("🔧 Choisir les couleurs")
    target_color = st.sidebar.selectbox("🎯 Couleur à modifier", list(COLORS.keys()))
    replacement_color = st.sidebar.selectbox("✨ Nouvelle couleur", list(COLORS.keys()))

    # ----------------------------
    # Masquage
    # ----------------------------
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in COLORS[target_color]["ranges"]:
        mask += cv2.inRange(hsv, np.array(lower), np.array(upper))
    mask = cv2.GaussianBlur(mask, (15, 15), 0)

    st.subheader("🧠 Masque de la couleur sélectionnée")
    st.image(mask, clamp=True, use_column_width=True)

    # ----------------------------
    # Segmentation
    # ----------------------------
    segmented = cv2.bitwise_and(image, image, mask=mask)
    st.subheader("🎯 Objet segmenté")
    st.image(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB), use_column_width=True)

    # ----------------------------
    # Recoloration
    # ----------------------------
    seg_hsv = cv2.cvtColor(segmented, cv2.COLOR_BGR2HSV).astype(np.float32)
    seg_hsv[..., 0] = COLORS[replacement_color]["hue"]
    recolored = cv2.cvtColor(seg_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    alpha = (mask/255.0)[..., np.newaxis]
    final = (recolored*alpha + image*(1-alpha)).astype(np.uint8)

    # ----------------------------
    # Comparaison Avant / Après
    # ----------------------------
    st.subheader("✨ Comparaison Avant / Après")
    col1, col2 = st.columns(2)
    col1.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Avant")
    col2.image(cv2.cvtColor(final, cv2.COLOR_BGR2RGB), caption="Après")

    # ----------------------------
    # Sauvegarde
    # ----------------------------
    if st.sidebar.button("💾 Sauvegarder l'image"):
        os.makedirs("images_modified", exist_ok=True)
        name = uploaded_file.name.split(".")[0]
        path = f"images_modified/{name}_modifiedbyjoyce.png"
        Image.fromarray(cv2.cvtColor(final, cv2.COLOR_BGR2RGB)).save(path)
        st.sidebar.success(f"✅ Image sauvegardée : {path}")

# ----------------------------
# Footer / copyright
# ----------------------------
st.markdown("""
<hr style='border:1px solid #ccc'>
<p style='text-align:center; color: gray;'>© 2025 Joyce. Tous droits réservés.</p>
""", unsafe_allow_html=True)
