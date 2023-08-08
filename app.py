import pathlib
import streamlit as st
import logging
from demo.study import Study
from time import sleep
from demo.dataset import TCIADataset
from demo.pipeline import val_transformer
from demo.model import UNet
import numpy
import cv2
import torch

import pandas

_DATABASE_PATH = pathlib.Path(__file__).parent / "data"

@st.cache_resource
def call_create_patient_database():
    return Study(_DATABASE_PATH)

def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    return UNet.load_from_checkpoint("model.ckpt", n_classes=1).to(device())


def eval_model(model):
    with torch.inference_mode():
        return model(image_to_predict).cpu().numpy().squeeze().round()

def show_results(show, mri, mask, mask_predicted, patient, cut_number):    
    with show[0]:
        # Getting the contours of the mask
        imcont_real, contours = cv2.findContours(mask.astype(numpy.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        imcont_pred, contours = cv2.findContours(mask_predicted, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # Draw the contours on the MRI image
        out_real = cv2.drawContours(mri.copy(), imcont_real, -1, (0,204,102), 2)
        out_pred = cv2.drawContours(mri.copy(), imcont_pred, -1, (102,178,255), 2)

        im_real = st.image(out_real, caption=f"Patient {patient.patient_id} / Slice {cut_number}")
        im_withctr = st.image(out_pred, caption=f"Patient {patient.patient_id} / Slice {cut_number}")
        
    with show[1]:
        mask_real = st.image(mask, caption=f"Patient {patient.patient_id} / Slice {cut_number}")
        mask_pred = st.image(mask_predicted, caption=f"Patient {patient.patient_id} / Prediction of Slice {cut_number}")

    return im_real, im_withctr, mask_real, mask_pred

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
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
    patient_list: list[str] = [str(patient) for patient in study.patient_list]
    patient = st.selectbox("Select a patient", patient_list)

    patient = study.patient_list[patient_list.index(patient)]

    tensor_dataset = TCIADataset(patient._mri_images_data, patient._mri_masks_data, transform=val_transformer())

    slider = st.empty()
    animate = st.button('Animate')
    
    cut_number = slider.slider("Cut to see :", 0, len(patient.mri_images)-1, step=1)

    mri = patient._mri_images_data[cut_number]
    mask = patient._mri_masks_data[cut_number]

    mask_predicted = numpy.zeros_like(mask)
    if model is not None:
            st.toast("Prediction in progress...")
            image_to_predict = torch.unsqueeze(tensor_dataset[cut_number][0],0).to(device())
            mask_predicted = eval_model(model)
            st.toast("Prediction done ! ✅")
            # Convert to 8-bit single-channel image
            mask_predicted = mask_predicted.astype(numpy.uint8)*255

    show = st.columns(2)
    st.caption("**Color used**: :green[real mask], :blue[predicted mask]")
    images = show_results(show, mri, mask, mask_predicted, patient, cut_number)
        
    if animate:
        for i in range(len(patient.mri_images)):
            for img in images:
                img.empty()
            mask_predicted = numpy.zeros_like(mask)
            if model is not None:
                    image_to_predict = torch.unsqueeze(tensor_dataset[i][0],0).to(device())
                    mask_predicted = eval_model(model)
                    # Convert to 8-bit single-channel image
                    mask_predicted = mask_predicted.astype(numpy.uint8)*255
            images = show_results(show, patient._mri_images_data[i], patient._mri_masks_data[i], mask_predicted, patient, i)
            slider.slider("Cut to see :", 0, len(patient.mri_images)-1, step=1, value=i, key=f"slider{i}")
            sleep(3)
   