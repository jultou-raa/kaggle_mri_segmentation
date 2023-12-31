# coding: utf-8

"""This file handles a Study class that contains a list of patients and methods to manipulate them."""

import concurrent.futures
import logging
import pathlib
import re

from demo.patient import Patient


class Study:
    def __init__(self, database_path: pathlib.Path) -> None:
        self._study_path = database_path
        self._patient_list = self.create_patient_database()

    @property
    def study_path(self):
        return self._study_path

    @property
    def logger(self):
        return logging.getLogger(self.__class__.__name__)

    @property
    def patient_list(self):
        return self._patient_list

    @property
    def count_positive_mri(self):
        return sum((patient.count_positive_mri for patient in self.patient_list))

    @property
    def count_negative_mri(self):
        return sum((patient.count_negative_mri for patient in self.patient_list))

    def create_patient_database(self) -> list[Patient]:
        """For each subfolders in data, create a list of Patient."""

        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(
                    Patient(institution, patient_id, slice_number, self.study_path).read
                )
                for patient_folder in self.study_path.iterdir()
                if patient_folder.is_dir()
                for institution, patient_id, slice_number in [
                    self._extract_info_from_folder_name(patient_folder.name)
                ]
            ]

        patient_list = [
            future.result() for future in concurrent.futures.as_completed(futures)
        ]

        return patient_list

    @staticmethod
    def _extract_info_from_folder_name(folder_name: str) -> tuple[str, str, str]:
        """
        Extract institution, patient id and slice number from the folder named
        `TCGA_<institution-code>_<patient-id>_<slice-number>`
        """

        match = re.match(r"TCGA_(\w+)_(\w+)_(\d+)", folder_name)

        institution = match.group(1)
        patient_id = match.group(2)
        slice_number = match.group(3)

        return institution, patient_id, slice_number

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} containing {len(self.patient_list)} patients>"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    data_path = pathlib.Path(__file__).parent.parent / "data"
    study = Study(data_path)
    print(study)
