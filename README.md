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

## License

This project is licensed under the GNU General Public License v3 (GPLv3) - see the [LICENSE][LICENSE] file for details.

[dataset_kaggle]: https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation
[install_docker]: https://docs.docker.com/get-docker/
[LICENSE]: https://github.com/jultou-raa/kaggle_mri_segmentation/blob/main/LICENSE