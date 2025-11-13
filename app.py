import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras import layers, Model
from PIL import Image

# === Bangun ulang arsitektur UNet ===
def conv_block(x, filters, kernel_size=3, batch_norm=True, dropout=0.0):
    x = layers.Conv2D(filters, kernel_size, padding='same', kernel_initializer='he_normal')(x)
    if batch_norm: x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, kernel_size, padding='same', kernel_initializer='he_normal')(x)
    if batch_norm: x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    if dropout > 0.0: x = layers.Dropout(dropout)(x)
    return x

def encoder_block(x, filters, pool=True, dropout=0.0):
    c = conv_block(x, filters, dropout=dropout)
    if pool:
        p = layers.MaxPooling2D((2,2))(c)
        return c, p
    return c

def decoder_block(x, skip, filters):
    x = layers.Conv2DTranspose(filters, (2,2), strides=(2,2), padding='same')(x)
    x = layers.Concatenate()([x, skip])
    x = conv_block(x, filters)
    return x

def build_unet(input_shape=(256, 256, 3), out_channels=1, base_filters=32, dropout_rate=0.1):
    inputs = layers.Input(shape=input_shape)
    c1, p1 = encoder_block(inputs, base_filters, dropout=dropout_rate)
    c2, p2 = encoder_block(p1, base_filters*2, dropout=dropout_rate)
    c3, p3 = encoder_block(p2, base_filters*4, dropout=dropout_rate)
    c4, p4 = encoder_block(p3, base_filters*8, dropout=dropout_rate)
    b = conv_block(p4, base_filters*16, dropout=dropout_rate)
    d1 = decoder_block(b, c4, base_filters*8)
    d2 = decoder_block(d1, c3, base_filters*4)
    d3 = decoder_block(d2, c2, base_filters*2)
    d4 = decoder_block(d3, c1, base_filters)
    outputs = layers.Conv2D(out_channels, (1,1), activation='sigmoid')(d4)
    return Model(inputs, outputs)

# === Load model ===
@st.cache_resource
def load_unet_model():
    model = build_unet(input_shape=(256, 256, 3))
    model.load_weights("unet_best.h5")  # pastikan path sesuai
    return model

# === Streamlit UI ===
st.set_page_config(page_title="Brain Tumor Segmentation", page_icon="🧠", layout="wide")

# ===== Sidebar Navigasi =====
menu = st.sidebar.radio("Navigasi", ["🏠 Home", "🧠 Deteksi Tumor"])

# ===== Halaman HOME =====
if menu == "🏠 Home":
    st.title("🧠 Brain Tumor Segmentation App")
    st.image("assets/brain_cover.jpeg", use_column_width=True)
    st.markdown("""
    ## Tentang Aplikasi
    Aplikasi ini menggunakan **UNet berbasis Deep Learning** untuk mendeteksi area tumor pada citra MRI/CT otak.

    ### Fitur Utama:
    - Deteksi otomatis area tumor
    - Visualisasi mask dan overlay
    - Antarmuka sederhana berbasis Streamlit
    """)

# ===== Halaman DETEKSI TUMOR =====
elif menu == "🧠 Deteksi Tumor":
    st.title("🧠 Deteksi Tumor Otak dari Citra MRI/CT")
    
    uploaded_file = st.file_uploader("📤 Upload gambar otak (.jpg, .png, .tif)", type=["jpg","png","tif"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Citra Input", use_column_width=True)

        img_resized = image.resize((256, 256))
        img_array = img_to_array(img_resized) / 255.0
        input_img = np.expand_dims(img_array, axis=0)

        with st.spinner("🔍 Memproses segmentasi tumor..."):
            model = load_unet_model()
            pred_mask = model.predict(input_img)[0]
            binary_mask = (pred_mask > 0.5).astype(np.uint8)

        overlay = img_array.copy()
        overlay[...,0] = np.maximum(overlay[...,0], binary_mask.squeeze())  # merah untuk tumor

        st.subheader("🎯 Hasil Segmentasi")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(img_array, caption="Citra Asli", use_column_width=True)
        with col2:
            st.image(binary_mask.squeeze()*255, caption="Mask Prediksi", use_column_width=True, clamp=True)
        with col3:
            st.image(overlay, caption="Overlay Tumor", use_column_width=True)
    else:
        st.info("Silakan upload gambar untuk mulai prediksi.")
