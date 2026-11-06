# DARSM: Deep Algebraic Reynolds Stress Model

Trained models and inference code for

> D. Dehtyriov, J. F. MacArt, J. Sirignano. *Deep Learning-based Algebraic
> Reynolds Stress Closures for RANS Simulations of Turbulent Flows.*
> Advances in Neural Information Processing Systems (NeurIPS), 2026.
> arXiv: [2605.26358](https://arxiv.org/abs/2605.26358)

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{dehtyriov2026darsm,
  title     = {Deep Learning-based Algebraic {R}eynolds Stress Closures for {RANS} Simulations of Turbulent Flows},
  author    = {Dehtyriov, Daniel and MacArt, Jonathan F. and Sirignano, Justin},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## Code availability

This repository currently contains the trained checkpoints and the inference
code. The full code (forward RANS solver, hybrid discrete adjoint, and training
pipeline) is available on request and will be released in this repository at
the time of NeurIPS 2026. Please contact Dr Daniel Dehtyriov
(daniel.dehtyriov@maths.ox.ac.uk).

## Contents

```
.
├── README.md
├── LICENSE
├── checkpoints/
│   ├── LICENSE.md
│   ├── multi_re.pt          # square duct, Re sweep
│   ├── aspect_ratio.pt      # square duct, AR sweep
│   └── periodic_hills.pt    # periodic hills, alpha sweep
├── darsm/
│   └── pretrained.py        # network and input features for the checkpoints
├── examples/
│   └── example.py           # runs the network on a synthetic duct field
└── data/
    └── pointers.md          # URLs and citations for the DNS datasets
```

Requires Python 3.9+ and PyTorch 2.0+.

## Example

```bash
python examples/example.py
```

Run from the repository root. Prints the closure constants predicted by the
Reynolds-number checkpoint at one cell of a simple square-duct flow, next to
their default values.

## License

Copyright (c) 2026 Daniel Dehtyriov.

The code is licensed under the [GNU Affero General Public License v3.0](LICENSE).
The model checkpoints are licensed under [CC BY-NC 4.0](checkpoints/LICENSE.md).
For other licensing, including commercial use, please contact
daniel.dehtyriov@maths.ox.ac.uk.
