# coding: utf-8

"""This file will have nodes to create a pipeline."""

from demo.study import Study
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from demo.dataset import TCIADataset
import torch


def create_image_database(study: Study):
    """Create a database of images from a study."""
    return (
        [image for patient in study.patient_list for image in patient._mri_images_data],
        [mask for patient in study.patient_list for mask in patient._mri_masks_data],
        [
            diagnosis
            for patient in study.patient_list
            for diagnosis in patient.positive_mri
        ],
    )


def train_transformer():
    """Apply augmentation to the training set."""
    return A.Compose(
        [
            A.Resize(256, 256),  # Given data is 256x256 so it is more for robustness
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.RandomGamma(p=0.5),
            A.Blur(p=0.5),
            A.GaussNoise(p=0.5),
            A.MotionBlur(p=0.5),
            A.OpticalDistortion(p=0.5),
            A.GridDistortion(p=0.5),
            A.ElasticTransform(p=0.5),
            # The following values are often used to normalize MRI images:
            # Mean: 0.019, 0.027, 0.045
            # Standard deviation: 0.052, 0.058, 0.077
            A.Normalize(mean=(0.019, 0.027, 0.045), std=(0.052, 0.058, 0.077)),
            ToTensorV2(transpose_mask=True),
        ]
    )


def val_transformer():
    return A.Compose(
        [
            A.Resize(256, 256),  # Given data is 256x256 so it is more for robustness
            # The following values are often used to normalize MRI images:
            # Mean: 0.019, 0.027, 0.045
            # Standard deviation: 0.052, 0.058, 0.077
            A.Normalize(mean=(0.019, 0.027, 0.045), std=(0.052, 0.058, 0.077)),
            ToTensorV2(transpose_mask=True),
        ]
    )


def pre_treatement_pipeline(
    study: Study,
) -> tuple[TCIADataset, TCIADataset, TCIADataset]:
    # Load data from the study (not using DataLoader as we already load everything for plotting purpose in the demo app)
    X, y, diagnosis = create_image_database(study)

    # First split into Train and Test samples
    (
        X_train,
        X_test,
        y_train,
        y_test,
        diagnosis_train,
        diagnosis_test,
    ) = train_test_split(
        X,
        y,
        diagnosis,
        train_size=0.9,
        random_state=42,
        stratify=diagnosis,
        shuffle=True,
    )

    # Second split into Train and Validation sample (from first train set : cross-validation purpose)
    (
        X_train,
        X_val,
        y_train,
        y_val,
        diagnosis_train,
        diagnosis_val,
    ) = train_test_split(
        X_train,
        y_train,
        diagnosis_train,
        train_size=0.8,
        random_state=42,
        stratify=diagnosis_train,
        shuffle=True,
    )

    train_dataset = TCIADataset(X_train, y_train, transform=train_transformer())

    validation_dataset = TCIADataset(X_val, y_val, transform=val_transformer())

    test_dataset = TCIADataset(X_test, y_test, transform=val_transformer())

    return train_dataset, validation_dataset, test_dataset


if __name__ == "__main__":
    import pathlib
    import logging
    import copy
    import matplotlib.pyplot as plt

    logging.basicConfig(level=logging.DEBUG)

    def visualize_augmentations(dataset, idx=0, samples=5):
        dataset = copy.deepcopy(dataset)
        dataset.transform = A.Compose(
            [
                t
                for t in dataset.transform
                if not isinstance(t, ToTensorV2)
            ]
        )
        figure, ax = plt.subplots(nrows=samples, ncols=2, figsize=(10, 24))
        for i in range(samples):
            image, mask = dataset[idx]
            ax[i, 0].imshow(image)
            ax[i, 1].imshow(mask, interpolation="nearest")
            ax[i, 0].set_title("Augmented image")
            ax[i, 1].set_title("Augmented mask")
            ax[i, 0].set_axis_off()
            ax[i, 1].set_axis_off()
        figure.tight_layout()
        plt.show()

    study = Study(pathlib.Path(__file__).parent.parent / "data")

    train_dataset, validation_dataset, test_dataset = pre_treatement_pipeline(study)

    print(f"Train Dataset: {len(train_dataset)}")
    print(f"Validation Dataset: {len(validation_dataset)}")
    print(f"Test Dataset: {len(validation_dataset)}")

    visualize_augmentations(train_dataset, idx=20)
