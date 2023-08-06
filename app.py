import pathlib
import streamlit as st
import logging
from demo.study import Study
from time import sleep
from demo.dataset import TCIADataset
from demo.pipeline import val_transformer
from demo.model import UNet
from demo.patient import Patient
import pytorch_lightning  as pl
import torch

import pandas

_DATABASE_PATH = pathlib.Path(__file__).parent / "data"

@st.cache_data
def call_create_patient_database():
    return Study(_DATABASE_PATH)

def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    try:
        model = UNet.load_from_checkpoint("best_model.ckpt", n_classes=1).to(device())
        model.eval()
        return model
    except FileNotFoundError:
        return None

def eval_model(model):
    with torch.no_grad():
        return model(image_to_predict).cpu().detach().numpy().squeeze().round()

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    st.title("MRI Cancer diagnostic")
    st.write(":warning: This is a demo of the MRI Cancer diagnostic app.")

    st.header("Study data")

    
    study = call_create_patient_database()
    model = load_model()

    positive_negative_count = pandas.DataFrame(
        [study.count_positive_mri, study.count_negative_mri],
        index=["Positive", "Negative"],
        columns=["Diagnosis"])
    st.bar_chart(positive_negative_count)

    st.header("Patient data")
    patient = st.selectbox("Select a patient", study.patient_list) # type: Patient
    tensor_dataset = TCIADataset(patient._mri_images_data, patient._mri_masks_data, transform=val_transformer())

    slider = st.empty()
    animate = st.button('Animate')
    
    cut_number = slider.slider("Cut to see :", 0, len(patient.mri_images)-1, step=1)

    show = st.columns(2)
    with show[0]:
        im1 = st.image(patient._mri_images_data[cut_number], caption=f"Patient {patient.patient_id} / Slice {cut_number}")
    with show[1]:
        im2 = st.image(patient._mri_masks_data[cut_number], caption=f"Patient {patient.patient_id} / Slice {cut_number}")
        if model is not None:
            image_to_predict = torch.unsqueeze(tensor_dataset[cut_number][0],0).to(device())
            im3 = st.image(eval_model(model), caption=f"Patient {patient.patient_id} / Prediction of Slice {cut_number}")
        
    if animate:
        for i in range(len(patient.mri_images)):
            im1.empty()
            im2.empty()
            slider.slider("Cut to see :", 0, len(patient.mri_images)-1, step=1, value=i, key=f"slider{i}")
            with show[0]:
                im1 = st.image(patient._mri_images_data[i], caption=f"Patient {patient.patient_id} / Slice {i}")
            with show[1]:
                im2 = st.image(patient._mri_masks_data[i], caption=f"Patient {patient.patient_id} / Slice {i}")
            sleep(0.5)



    