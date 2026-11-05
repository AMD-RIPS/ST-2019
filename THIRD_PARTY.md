# License scope and third-party material

The root [MIT license](LICENSE) applies to AMD-owned source code and
accompanying software documentation in this repository, including the Glitchify
generator. Copyright (c) 2019-2026 Advanced Micro Devices, Inc.

The following material is outside that grant:

- **GAN experiments:** `Experiments/GAN/` is excluded from the MIT grant in this
  update pending a separate review of its upstream licenses and bundled assets.
  Its [README](Experiments/GAN/readme.md) identifies
  [csinva/pytorch-gan-pretrained](https://github.com/csinva/pytorch-gan-pretrained)
  as the model source. The first line of
  [dcgan.py](Experiments/GAN/cifar10_dcgan/dcgan.py) also references
  [pytorch/examples](https://github.com/pytorch/examples/blob/main/dcgan/main.py).
  These references identify provenance; they do not establish the applicable
  license for each bundled file or revision.
- **Data and media:** Datasets, images, video clips, and model weights are not
  licensed by the root MIT file. This includes the sample inputs in
  `Glitchify/input/`, bundled training data and checkpoints, and report figures.
- **Papers and reports:** The accompanying paper and the material under
  `LaTeX Deliverable/` are not licensed by the root MIT file. Consult the
  [paper's arXiv record](https://arxiv.org/abs/2011.15103) for its terms.
- **Other third-party components:** Dependencies and any third-party material
  retain their applicable licenses and notices. The root MIT file does not
  replace those terms or grant rights in third-party content.

When running Glitchify, use input images or videos that you have permission to
use. The software license does not grant rights in the input content or remove
conditions that apply to that content when producing or distributing outputs.

This file records the scope of this licensing update and the upstream references
identified in the repository. It is not a complete third-party license inventory.
