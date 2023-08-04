# encoding: utf-8
import concurrent.futures
import logging
import pathlib
import queue
import re

import cv2
import natsort
import numpy
import plotly.graph_objects as go
from plotly.subplots import make_subplots

""" This file handle a class named Patient. 
This class has several attributes :
    - id : int, the id of the patient
    - mri_path : pathlib.Path, the path to the mri file
    - mri_images: list, the list of the mri images
    - mri_masks: list, the list of the mri masks
    
There is a method to populate the class called read
There is a method to classify the mri images called classify_mri
"""


class Patient:
    def __init__(
        self,
        institution: str,
        patient_id: str,
        slice_number: str,
        data_base_path: pathlib.Path,
    ) -> None:
        self.slice_number: str = slice_number
        self.institution: str = institution
        self.patient_id: str = patient_id
        self._mri_images: list = []
        self._mri_masks: list = []
        self._mri_masks_data = []
        self._mri_images_data = []
        self._data_base_path = data_base_path

    @property
    def mri_path(self) -> pathlib.Path:
        return (
            self._data_base_path
            / f"TCGA_{self.institution}_{self.patient_id}_{self.slice_number}"
        )

    @property
    def mri_images(self) -> list[pathlib.Path]:
        return natsort.natsorted(self._mri_images)

    @property
    def mri_masks(self) -> list[pathlib.Path]:
        return natsort.natsorted(self._mri_masks)

    @property
    def positive_mri(self):
        return [image.any() for image in self._mri_masks_data]

    @property
    def count_positive_mri(self):
        return numpy.sum(self.positive_mri)

    @property
    def count_negative_mri(self):
        return numpy.sum([not value for value in self.positive_mri])

    @property
    def logger(self):
        return logging.getLogger(self.__class__.__name__)

    def read(self):
        """Populate the class with the mri images and masks.
        masks have the same name as mri images but with a _mask suffix.
        """
        self.logger.debug(f"Importing images from {self.mri_path}")
        self._mri_masks = [
            file
            for file in self.mri_path.iterdir()
            if file.is_file() and file.name.endswith("mask.tif")
        ]
        self._mri_images = [
            file
            for file in self.mri_path.iterdir()
            if file.is_file() and not file.name.endswith("mask.tif")
        ]

        self._mri_masks_data = [cv2.imread(str(mask)) for mask in self.mri_masks]
        self._mri_images_data = [cv2.imread(str(images)) for images in self.mri_images]

        return self

    def show_images(self, cut_number: int) -> go.Figure:
        """Show the images and masks of the patient."""

        fig = make_subplots(
            rows=1,
            cols=2,
            shared_xaxes=True,
            shared_yaxes=True,
            subplot_titles=["MRI", "Mask"],
            column_widths=[
                0.5,
            ]
            * 2,
        )

        mri, mask = self._mri_images_data[cut_number], self._mri_masks_data[cut_number]

        fig.add_trace(go.Image(z=mri), row=1, col=1)
        fig.add_trace(
            go.Heatmap(z=mask, colorscale="gray", showscale=False),
            row=1,
            col=2,
        )

        fig.update_layout(yaxis=dict(scaleanchor="x"))

        return fig

    def __repr__(self) -> str:
        return f"Patient {self.patient_id} from {self.institution} with slice number {self.slice_number}"
