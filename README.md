# ST-2019
Automated Glitch Detection in Video Games


## Description:

This work includes generating corrupted images by inserting several type of artifacts to images/videos and using the "ensemble" model to classify normal and corrupted images.


The program to generate corrupted images is under the "Glitchify" folder. 

The ensemble model to classify images as glitched or not is under the "Ensemble" folder. 

Methods to extract features that we have explored but not used in the ensemble model are included in the "Experiment" folder. 

Some old tools and outdated models are in the "Archive" folder.


## Accompanying paper

This repository contains the Glitchify generator and artifact-detection code
described in:

Parmida Davarmanesh, Kuanhao Jiang, Tingting Ou, Artem Vysogorets, Stanislav
Ivashkevich, Max Kiehn, Shantanu H. Joshi, and Nicholas Malaya.
**Automating Artifact Detection in Video Games.** arXiv:2011.15103, 2020.
[Paper](https://arxiv.org/abs/2011.15103) |
[PDF](https://arxiv.org/pdf/2011.15103) |
[DOI](https://doi.org/10.48550/arXiv.2011.15103).

If you use this code in research, please cite the paper. GitHub's "Cite this
repository" feature uses [CITATION.cff](CITATION.cff) to provide the preferred
paper citation. Citation is requested, not an additional license condition.

```bibtex
@misc{davarmanesh2020automating,
  title = {Automating Artifact Detection in Video Games},
  author = {Parmida Davarmanesh and Kuanhao Jiang and Tingting Ou and
            Artem Vysogorets and Stanislav Ivashkevich and Max Kiehn and
            Shantanu H. Joshi and Nicholas Malaya},
  year = {2020},
  eprint = {2011.15103},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV},
  doi = {10.48550/arXiv.2011.15103},
  url = {https://arxiv.org/abs/2011.15103}
}
```

## License

Copyright (c) 2019-2026 Advanced Micro Devices, Inc.

AMD-owned source code and accompanying software documentation, including
Glitchify, are available under the [MIT license](LICENSE). This permits research
and commercial use, modification, and redistribution, subject to the license's
notice requirements.

The grant excludes `Experiments/GAN/` pending a separate upstream license review,
as well as datasets, images, videos, model weights, papers, and reports.
Third-party components retain their own terms. See
[license scope and third-party material](THIRD_PARTY.md) for details.

## Setup:

Run the following two commands to install the required packages, or to install all the packages listed in "setup.txt".

chmod +x install_packages.sh

bash install_packages.sh


## Contributors:

Parmida Davarmanesh, University of Michigan

Kuanhao Jiang, University of Pennsylvania

Tingting Ou, Johns Hopkins University

Artem Vysogorets, University of Massachusetts Amherst

Shantanu Joshi, University of California, Los Angeles

Nicholas Malaya, AMD Research
