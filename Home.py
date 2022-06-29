import av
from streamlit_webrtc import (RTCConfiguration, VideoProcessorBase, WebRtcMode, webrtc_streamer)
import streamlit as st
from PIL import Image
import numpy as np
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import os
import cv2

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

main_page_markdown = f"""
    ### Situs ini digunakan untuk proyek demonstrasi klasifikasi penyakit tanaman pada daun jagung

  """


def prediction(image_name, transform, inference_model, prediction_type=None):
    if prediction_type == "realtime":
        image_prediction = Image.fromarray(image_name)
    else:
        image_prediction = Image.open(image_name)

    image_prediction = transform(image_prediction).float()
    image_prediction = image_prediction.unsqueeze(0)
    loader = torch.utils.data.DataLoader(image_prediction, batch_size=1, shuffle=True)
    data_iteration = iter(loader).next()

    with torch.no_grad():
        inference_model.eval()
        output = inference_model(data_iteration)
        sm = torch.nn.Sigmoid()
        sm_output = (sm(output).squeeze(0)).tolist()
        index = sm_output.index(max(sm_output))

    return index, sm_output


def app_corn_disease_predictor():
    model_name = 'mobilenetv2'
    num_classes = 4
    feature_extract = False
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transAug = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean, std),
         transforms.Resize((256, 256))])
    idx_to_class = {
        0: 'Karat Daun',
        1: 'Bercak Daun',
        2: 'bulai',
        3: 'Hawar Daun'
    }

    model_ft, _ = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

    class VideoPredictor(VideoProcessorBase):
        def __init__(self):
            pass
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            idx, output = prediction(img, transAug, model_ft, prediction_type="realtime")

            print(f"Prediction: {idx} - {idx_to_class[idx]}")
            print(f"Probability : {output[idx]}")
            print(f"Overall prediction : {output}")

            cv2.putText(img, f'{idx} - {idx_to_class[idx]}, {output}', (50, 50),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        0.9, (255, 255, 255), 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_ctx = webrtc_streamer(
        key="corn-disease-classification",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=VideoPredictor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )


@st.cache(allow_output_mutation=True)
def retrieve_model(pretrained):
    model = models.mobilenet_v2(pretrained=pretrained)
    return model


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    model_ft = None
    input_size = 0

    if model_name == 'mobilenetv2':
        model_ft = retrieve_model(use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_features = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(num_features, num_classes)
        model_state_dict = os.path.join(os.path.dirname(__file__), 'model (1).pth')
        model_ft.load_state_dict(torch.load(model_state_dict, map_location=torch.device('cpu')))

    else:
        print("invalid model name")
        exit()

    return model_ft, input_size


def main():
    activities = ["Home", "Live Classification", "Upload Image", "Camera Input", "About"]
    choice = st.sidebar.selectbox('Select Activity', activities)

    model_name = 'mobilenetv2'
    num_classes = 4
    feature_extract = False
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    idx_to_class = {
        0: 'Karat Daun',
        1: 'Bercak Daun',
        2: 'bulai',
        3: 'Hawar Daun'
    }

    transAug = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean, std),
         transforms.Resize((256, 256))])
    model_ft, _ = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)


    if choice == "Live Classification":
        st.markdown(main_page_markdown)
        app_corn_disease_predictor()
        st.markdown(f'''Press Start 👈 to start the show!''')

    elif choice == 'Upload Image':
        st.subheader('Inference model using image')
        image_file = st.file_uploader('Upload Image', type=["jpg", "png", "jpeg"])
        if image_file is not None:
            our_static_image = Image.open(image_file)
            st.image(our_static_image, width=400)
            idx, output = prediction(image_file, transAug, model_ft)

            st.write(f""" ### Prediction: {idx} - {idx_to_class[idx]}""")
            st.write(f"Probability : {output[idx]}")
            st.write(f"Overall prediction : **{output}**")

    elif choice == "Camera Input":
        img_file_buffer = st.camera_input("Take a picture")

        if img_file_buffer is not None:
            # To read image file buffer as a PIL Image:
            img = Image.open(img_file_buffer)
            st.image(img, width=400)
            idx, output = prediction(img_file_buffer, transAug, model_ft)

            st.write(f""" ### Prediction: {idx} - {idx_to_class[idx]}""")
            st.write(f"Probability : {output[idx]}")
            st.write(f"Overall prediction : **{output}**")

    elif choice == "About":
        image = Image.open('2018-07-31 06.32.30 2.jpg')
        st.markdown("""
# About
---
### Profil tentang Author:""")
            
        st.image(image, caption='Wiwin Nur Kholifah')
        st.markdown(
                """
| Nama | Wiwin Nur Kholifah|
|------|-------------------|
| NIM | 18090030 |
| Prodi | Teknik Informatika |
| No. Telp | 085848718618 |
| Email | wiwinnurkholifah850@gmail.com|""")
    
    elif choice == "Home":
        st.markdown("""
        
        # Aplikasi Demo Klasifikasi Penyakit Jagung dengan Metode CNN
### Tugas Akhir D4 Teknik Informatika Politeknik Harapan Bersama Tegal

Website ini merupakan proyek demonstrasi klasifikasi penyakit tanaman pada daun jagung, yang terdiri dari penyakit bulai, bercak daun, karat daun dan hawar daun.

## Latar Belakang
Kendala utama yang terjadi pada masa penanaman jagung yaitu gangguan biotis yaitu gangguan oleh makroorganisme (gangguan hama) dan gangguan oleh mikroorganisme (gangguan penyakit). Penyakit tanaman jagung merupakan hasil interaksi dari tiga komponen utama yaitu patogen, inang, dan lingkungan. Kurangnya informasi dan pengetahuan tentang penyakit dari tanaman jagung bagi petani dapat menyebabkan kesalahan diagnosa penyakit yang menyerang tanaman jagung yang berdampak pula pada kesalahan pengendaliannya. Terdapat beberapa penyakit yang dapat menyerang tanaman jagung seperti, penyakit bulai (Downy mildew), hawar daun (Northern Leaf Blight), karat daun (Southern Rust), dan  bercak daun (Southern Leaf Blight).

Berdasarkan permasalahan diatas maka perlu dilakukan pendeteksian dini serta pengidentifikasian penyakit tanaman mejadi faktor utama untuk mencegah dan mengurangi penyebaran penyakit pada tanaman jagung. Penyakit yang tidak terdeteksi dan dibiarkan berkembang dapat mengakibatkan kerusakan

Oleh karena itu, penelitian ini dilakukan bertujuan untuk mengklasifikasi penyakit jagung. Adapun metode yang digunakan dalam proses klasifikasi adalah Convolutional Neural Network, yang pada prosesnya data training dilatih untuk menghasilkan tingkat akurasi yang maksimal. Convolutional Neural Network (CNN) merupakan salah satu metode Deep learning yang dapat digunakan untuk mendeteksi dan mengenali sebuah objek pada sebuah citra digital. 

## Fitur Aplikasi

- Realtime Klasifikasi
- Klasifikasi menggunakan input dari upload file
- Klasifikasi menggunakan input kamera

## Library yang Digunakan

Pada aplikasi ini menggunakan beberapa library:

- [Pytorch] - Digunakan sebagai basis deeplearning (pretrained, training dan testing model)
- [Albumentation] - Augmentasi gambar
- [Streamlit] - Digunakan sebagai frontend dan juga backend untuk melakukan inferensi model
- [Streamlit-WebRtc] - Digunakan untuk melakukan inferensi realtime menggunakan WebRtc
- [Matplotlib] - Library untuk visualisasi data

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   [pytorch]: <https://pytorch.org/>
   [albumentation]: <https://albumentations.ai/>
   [Streamlit-WebRtc]: <https://github.com/whitphx/streamlit-webrtc>
   [streamlit]: <https://streamlit.io/>
   [Matplotlib]: <https://matplotlib.org/>
        
        """)

if __name__ == '__main__':
    main()
