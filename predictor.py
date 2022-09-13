import numpy as np
from streamlit_webrtc import (RTCConfiguration, VideoProcessorBase, WebRtcMode, webrtc_streamer)
from PIL import Image
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import Dataset
import torch.nn as nn
import torch
import cv2
import os
import av
import streamlit as st

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


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
    num_classes = 6
    feature_extract = False
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transAug = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean, std),
         transforms.Resize((256, 256))])

    idx_to_class = {0: 'Hawar Daun', 1: 'Karat Daun', 2: 'Bukan Daun', 3: 'Daun Sehat', 4: 'bulai', 5: 'Bercak Daun'}

    model_ft, _ = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

    class VideoPredictor(VideoProcessorBase):
        def transform(self, frame: av.VideoFrame) -> np.ndarray:
            pass

        def __init__(self):
            pass

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            idx, output = prediction(img, transAug, model_ft, prediction_type="realtime")

            output = [float("{:.2f}".format(i)) for i in output]

            st.write(f""" ### Prediction: {idx} - {idx_to_class[idx]}""")
            st.write(f"Probability : {output[idx]}")
            st.write(f"Overall prediction : **{output}**")

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
        model_state_dict = os.path.join(os.path.dirname(__file__), 'model.pth')
        model_ft.load_state_dict(torch.load(model_state_dict, map_location=torch.device('cpu')))

    else:
        print("invalid model name")
        exit()

    return model_ft, input_size
