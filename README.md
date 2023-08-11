[![codecov](https://codecov.io/gh/jultou-raa/kaggle_mri_segmentation/branch/main/graph/badge.svg?token=JSLZ4167JI)](https://codecov.io/gh/jultou-raa/kaggle_mri_segmentation)
![CI](https://github.com/jultou-raa/kaggle_mri_segmentation/actions/workflows/main.yml/badge.svg?event=push)
[![Try](https://img.shields.io/badge/Azure%20-%20Try%20it%20out%20!%20-%20green?logo=microsoftazure&logoColor=%230078D4&label=Azure&labelColor=lightgrey&color=green)](http://demomri.azurewebsites.net)

# MRI Segmentation with Pytorch (Demo code)

This repository contains code and data for a Kaggle competition on brain tumor segmentation using MRI images. The goal is to develop a machine learning model that can automatically segment the tumor region from the surrounding healthy tissue in MRI scans of patients with low-grade gliomas (LGG).

## Dataset

[Dataset][dataset_kaggle] used in:

> Mateusz Buda, AshirbaniSaha, Maciej A. Mazurowski "Association of genomic subtypes of lower-grade gliomas with shape features automatically extracted by a deep learning algorithm." Computers in Biology and Medicine, 2019.

and

> Maciej A. Mazurowski, Kal Clark, Nicholas M. Czarnek, Parisa Shamsesfandabadi, Katherine B. Peters, Ashirbani Saha "Radiogenomics of lower-grade glioma: algorithmically-assessed tumor shape is associated with tumor genomic subtypes and patient outcomes in a multi-institutional study with The Cancer Genome Atlas data." Journal of Neuro-Oncology, 2017.

This dataset contains brain MR images together with manual FLAIR abnormality segmentation masks.
The images were obtained from The Cancer Imaging Archive (TCIA).
They correspond to 110 patients included in The Cancer Genome Atlas (TCGA) lower-grade glioma collection with at least fluid-attenuated inversion recovery (FLAIR) sequence and genomic cluster data available.

## Application

The application is a web-based interface that allows users to parse the MRI scans and get the segmentation results from the trained model. The application is built using Streamlit, a Python web framework, and PyTorch, a deep learning library.

The application is deployed using Docker, a tool that allows users to run applications in isolated containers. Docker makes it easy to install and run the application without worrying about dependencies or compatibility issues. To run the application using Docker, follow these steps:

1. Install Docker on your machine by following the instructions [here][install_docker].
2. Pull the Docker image from the repository [jultou/kaggle_mri_demo] by running the command `docker pull jultou/kaggle_mri_demo`.
3. Run the Docker container by running the command `docker run -p 80:80 jultou/kaggle_mri_demo`.
4. Open your web browser and go to http://localhost to access the application.
5. Wait for the model to load the database and process MRI scan then it will display the results.

## Model

The model used is the Modified UNet described in the paper :

>Zeineldin, R.A., Karar, M.E., Coburger, J. et al. DeepSeg: deep neural network framework for automatic brain tumor segmentation using magnetic resonance FLAIR images. Int J CARS 15, 909–920 (2020). https://doi.org/10.1007/s11548-020-02186-z

The training is done using the [Pytorch Lightning][Lightning] framework and 2 [Nvdidia T4 GPU][Nvdidia_T4_GPU].

You can train again by your own after installing this package and using this snippet :

1. `pip install -U git+https://github.com/jultou-raa/kaggle_mri_segmentation.git#egg=demo`

2. Execute the following code :
    ```python
    import pathlib
    from demo.pipeline import training_pipeline

    base_dir = pathlib.Path("/path/to/dataset")

    training_pipeline(
        study_path=base_dir,
        num_nodes=1,
        devices=2,
        max_time="00:10:00:00",
        num_workers=2,
        strategy='ddp_notebook', # Only when executing on notebook
        batch_size=24,
        auto_lr=False,
        max_epochs=225)
    ```

    This call will train the model and save it to the current directory as `model.cpkt`. You will be able to use it with the folowing snippet :
    
    ```python
    from demo.model import UNet

    model = UNet.load_from_checkpoint("model.ckpt", n_classes=1)
    ```

To avoid overfitting, an EarlyStopping callback is used with a patience of 10 epochs.

Also, a learning rate scheduler ([ReduceLROnPlateau][ReduceLROnPlateau]) is used to reduce the learning rate by a factor of 0.1 when the validation loss is not improving for 5 epochs.

The validation loss used is a mix between the Dice loss and the BCE loss.

## License

This project is licensed under the GNU General Public License v3 (GPLv3) - see the [LICENSE][LICENSE] file for details.

[dataset_kaggle]: https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation
[install_docker]: https://docs.docker.com/get-docker/
[LICENSE]: https://github.com/jultou-raa/kaggle_mri_segmentation/blob/main/LICENSE
[ReduceLROnPlateau]: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html
[Nvdidia_T4_GPU]: https://www.nvidia.com/en-us/data-center/tesla-t4/
[Lightning]: https://www.lightning.ai/