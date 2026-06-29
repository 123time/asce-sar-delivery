# ASCE-SAR YOLO11 Delivery

This repository is a clean delivery version of the ASCE-SAR model for SAR object detection experiments. It keeps the Ultralytics runtime needed to train and evaluate the delivered model, plus the custom modules used by the paper configuration.

## Paper Terminology

The paper-facing names used in the manuscript are:

| Paper name | Meaning |
| --- | --- |
| `ASCE-SAR` | The full adaptive shift-wise downsampling and frequency-spatial modulated feature fusion framework. |
| `ADSC` | Adaptive Downsampling with Shift-Wise Convolution backbone stage. |
| `DSFF` | Dynamic Shift-wise Feature Fusion neck stage. |
| `FSMB` | Frequency-Spatial Modulation Block. |

The codebase below keeps the implementation names aligned with these paper terms wherever possible.
## Custom Modules

The delivered model uses paper-facing module names in the YAML and parser:

| Module name | Location | Purpose |
| --- | --- | --- |
| `SWC` | `ultralytics/nn/extra_modules/delivery_modules.py` | Shift-wise convolution block used inside `ADSC`. |
| `EMA` | `ultralytics/nn/extra_modules/delivery_modules.py` | Efficient multi-scale attention block used inside `ASCE-SAR`. |
| `FMKernel` | `ultralytics/nn/extra_modules/delivery_modules.py` | Frequency-spatial modulation block used inside `FSMB`. |

Supporting modules used by the model are also kept:

| Module name | Location |
| --- | --- |
| `ADown` | `ultralytics/nn/modules/block.py` |
| `DySample` | `ultralytics/nn/extra_modules/delivery_modules.py` |
| `SPDConv` | `ultralytics/nn/extra_modules/delivery_modules.py` |

## Model Config

Main model YAML:

```text
ultralytics/cfg/models/11/yolo11-ADown-Backbone-SOEP1EMA-SWC-DySample-fixed.yaml
```

A copy is also kept at the repository root for quick inspection:

```text
yolo11-ADown-Backbone-SOEP1EMA-SWC-DySample-fixed.yaml
```

## Installation

Create or activate a Python environment with PyTorch, then install this repository in editable mode:

```bash
pip install -e .
```

The required runtime dependencies are declared in `pyproject.toml`, including PyTorch, NumPy, OpenCV, and Ultralytics runtime dependencies.

## Quick Check

Run a model construction smoke test:

```bash
python -c "from ultralytics import YOLO; model=YOLO('ultralytics/cfg/models/11/yolo11-ADown-Backbone-SOEP1EMA-SWC-DySample-fixed.yaml'); print(model.model.stride.tolist())"
```

Expected stride:

```text
[8.0, 16.0, 32.0]
```

Run a forward smoke test:

```bash
python -c "import torch; from ultralytics import YOLO; model=YOLO('ultralytics/cfg/models/11/yolo11-ADown-Backbone-SOEP1EMA-SWC-DySample-fixed.yaml'); model.model.eval(); y=model.model(torch.zeros(1,3,640,640)); print(type(y).__name__, len(y) if isinstance(y,(list,tuple)) else 'not_sequence')"
```

Expected output:

```text
tuple 2
```

## Training

Edit `DATA_CFG` in `train.py` to point to your dataset YAML:

```python
DATA_CFG = "path/to/data.yaml"
```

Then run:

```bash
python train.py
```

The default training entry uses:

```text
ultralytics/cfg/models/11/yolo11-ADown-Backbone-SOEP1EMA-SWC-DySample-fixed.yaml
```

## Project Scope

This delivery package intentionally removes unrelated experimental modules, extra YAML files, pretrained weights, datasets, examples, and temporary outputs. It is intended to expose only the model and modules required for reproducing the delivered ASCE-SAR architecture.

## License

This project keeps the original Ultralytics AGPL-3.0 license.
