# FewShotMusicRegionsICDAR24


<p align="center">
  <a href=""><img src="https://i.imgur.com/Iu7CvC1.png" alt="PRAIG-logo" width="100"></a>
</p>

<h1 align="center">A region-based approach for layout analysis of music score images in scarce data scenarios</h1>

<h4 align="center">Full text available <a href="" target="_blank">here</a>.</h4>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9.0-orange" alt="Gitter">
  <img src="https://img.shields.io/badge/Tensorflow-%FFFFFF.svg?style=flat&logo=Tensorflow&logoColor=orange&color=white" alt="Tensorflow">
  <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white" alt="Lightning">
  <img src="https://img.shields.io/static/v1?label=License&message=MIT&color=blue" alt="License">
</p>


<p align="center">
  <a href="#about">About</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#citations">Citations</a> •
  <a href="#acknowledgments">Acknowledgments</a> •
  <a href="#license">License</a>
</p>


## About

This work presents a neural network approach for layout analysis of music score images to extract staff regions. The work focusses on cases with limited amount of training data, a recurrent issue in the context of music.


## How To Use

To run the code, you'll need to meet certain requirements which are specified in the [`Dockerfile`](Dockerfile). Alternatively, you can set up a virtual environment if preferred.

Once you have prepared your environment (either a Docker container or a virtual environment), you are ready to begin. Execute the [`experiments/run.py`](experiments/run.py) script to replicate the experiments from our work:

```python
python experiments/run.py
```

## Citations

```bibtex
@inproceedings{Castellanos2024fewRegions,
  title     = {{A region-based approach for layout analysis of music score images in scarce data scenarios}},
  author    = {Francisco J. Castellanos, Juan P. Martínez-Esteso, Alejandro Galán-Cuenca, Antonio Javier Gallego},
  journal   = {{Proceedings of the International Conference on Document Analysis and Recognition, ICDAR}},
  volume    = {},
  pages     = {},
  year      = {2024},
  publisher = {},
  doi       = {},
}
```

## Acknowledgments
This research was supported by the Spanish Ministerio de Ciencia e Innovación through the I+D+i DOREMI project (TED2021-132103A-I00), funded by MCIN/AEI/10.13039/501100011033. 

This work is part of the I+D+i PID2020-118447RA-I00 ([MultiScore](https://sites.google.com/view/multiscore-project)) project, funded by MCIN/AEI/10.13039/501100011033. Computational resources were provided by the Valencian Government and FEDER funding through IDIFEDER/2020/003.

## License
This work is under a [MIT](LICENSE) license.
