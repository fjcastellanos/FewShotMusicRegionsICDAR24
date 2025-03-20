# FewShotMusicRegionsICDAR24


<p align="center">
  <a href=""><img src="https://i.imgur.com/Iu7CvC1.png" alt="PRAIG-logo" width="100"></a>
</p>

<h1 align="center">A region-based approach for layout analysis of music score images in scarce data scenarios</h1>

<h4 align="center">Full text available <a href="" target="_blank">after publication in the ICDAR conference.</a>.</h4>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9.0-orange" alt="Gitter">
  <img src="https://img.shields.io/badge/Tensorflow-%FFFFFF.svg?style=flat&logo=Tensorflow&logoColor=orange&color=white" alt="Tensorflow">
  <!--<img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white" alt="PyTorch">-->
  <!--<img src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white" alt="Lightning">-->
  <img src="https://img.shields.io/static/v1?label=License&message=GNU GPL v3&color=blue" alt="License">
</p>


<p align="center">
  <a href="#about">About</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#citations">Citations</a> •
  <a href="#acknowledgments">Acknowledgments</a> •
  <a href="#license">License</a>
</p>


## About

This work presents a neural network approach for layout analysis of music score images to extract staff regions. The work focuses on cases with a limited amount of training data, a recurrent issue in the context of music.
The proposal includes a masking layer to ignore the non-annotated pixels and a random oversampling around the annotated data to train the model even with a few annotated staves.

## How To Use

To run the code, you'll need to meet certain requirements which are specified in the [`Dockerfile`](Dockerfile). Alternatively, you can set up a virtual environment if preferred.

Once you have prepared your environment (either a Docker container or a virtual environment), you are ready to begin. Execute the [`sh_run_experiments.sh`](sh_run_experiments.sh) script to replicate the experiments from our work:

```
./sh_run_experiments.sh
```
The script includes a series of loops that can be configured according to the experiments. These loops makes reference to arguments of the Python program that runs the experiments.
Parameters:
  * **-db_train_txt** `Path to file with the list of the paths to the JSON files for training (extracted from MURET)`
  * **-db_val_txt** `Path to file with the list of the paths to the JSON files for validation (extracted from MURET)`
  * **-db_test_txt** `Path to file with the list of the paths to the JSON files for testing (extracted from MURET)`
  * **-cls** `List of classes to be considered` (**Default:** *"staff empty_staff"*)
  * **-aug** `List of augmentation techniques to be considered. The **random** value represents the oversampling proposal.` (**Default:** *random rot scale*) (**Other:** *flipV flipH*)
  * **-window_w** `Width of the window considered to extract patch samples` (**Default:** *256*)
  * **-window_h** `Height of the window considered to extract patch samples` (**Default:** *256*)
  * **-l** `Number of layers for the encoder and the decoder.` (**Default:** *3*)
  * **-f** `Number of filters of the convolutional layers.` (**Default:** *32*)
  * **-k** `Kernel size.` (**Default:** *3*)
  * **-drop** `Dropout rate.` (**Default:** *0.4*)
  * **-npatches** `Number of random patches extracted per epoch.` (**Default:** *2048*)
  * **-n_annotated_patches** `Number of annotated staves considered in the experiment. **-1** indicates using all the complete images.` (**Default:** *-1*)
  * **-pages_train** `Number of training pages. *-1* represents using all the pages.` (**Default:** *-1*)
  * **-e** `Maximum number of epochs.` (**Default:** *200*)
  * **-b** `Batch size.` (**Default:** *32*)
  * **-gpu** `Index of the GPU employed for the experiment.` (**Default:** *0*)
  * **-res** `Path to an output file to save the results.`
  * **-iou** `Threshold for IoU to consider a prediction as True Positive. Values between 0 and 1.` (**Default:** *0.5*)
  * **-vr** `Vertical reduction proportion for each staff. Value between 0 and 1.` (**Default:** *0.4*)
  * **-adapt-size-patch** `Flag to activate the scale adaptation of the images. This is used to adapt the size of the images to the window size. This should be activated.`
  * **--test** `It activates the testing mode. In this case, the script does not train the model. First use the script without this parameter.` 

The datasets must be organized to obtain a folder structure with a JSON folder with the JSON files and a SRC folder with the images. The images and their respective JSON ground truth should have the same name except for the extension of the JSON files, which includes the **.json** extension to the name of the image.


## Citations

```bibtex
@inproceedings{Castellanos2024fewRegions,
  title     = {{A region-based approach for layout analysis of music score images in scarce data scenarios}},
  author    = {Francisco J. Castellanos, Juan P. Martínez-Esteso, Alejandro Galán-Cuenca, Antonio Javier Gallego},
  booktitle   = {{18th IAPR International Conference on Document Analysis and Recognition (ICDAR), August 30–September 4}},
  pages     = {58–75},
  year      = {2024},
  publisher = {Springer-Verlag},
  doi       = {10.1007/978-3-031-70546-5_4},
  location = {Athens, Greece}
}
```

## Acknowledgments
This research was supported by the Spanish Ministerio de Ciencia e Innovación through the I+D+i DOREMI project (TED2021-132103A-I00), funded by MCIN/AEI/10.13039/501100011033. 

<a href="https://www.ciencia.gob.es/" target="_blank"><img src="https://www.dlsi.ua.es/~jgallego/projects/DOReMI/images/logo_min.png" style="height:100px" alt="Ministerio de Ciencia e Innovación"></a> 
&nbsp;
<a href="https://commission.europa.eu/strategy-and-policy/recovery-plan-europe_es" target="_blank"><img src="https://www.dlsi.ua.es/~jgallego/projects/DOReMI/images/logo_ue.png" style="height:100px" alt="Financiado por la Unión Europea, fondos NextGenerationEU"></a>
<br>
<a href="https://planderecuperacion.gob.es/" target="_blank"><img src="https://www.dlsi.ua.es/~jgallego/projects/DOReMI/images/logo_plan_recuperacion_transformacion_resiliencia.png" style="height:100px" alt="Plan de Recuperación, Transformación y Resiliencia"></a>
&nbsp;
<a href="https://www.aei.gob.es/" target="_blank"><img src="https://www.dlsi.ua.es/~jgallego/projects/DOReMI/images/logo_aei.png" style="height:100px" alt="Agencia Estatal de Investigación"></a>

<br/>

## License
This work is under a [GNU GPL v3](LICENSE) license.
