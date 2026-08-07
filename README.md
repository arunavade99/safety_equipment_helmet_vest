# safety_equipment_helmet_vest

Small project for PPE (helmet & vest) detection using a YOLOv8-based model.

Overview
- Detects helmets and safety vests in images/videos to help automate PPE compliance checks.

Files
- `main.py`: inference script / entrypoint.
- `best38.pt`: trained YOLOv8 weights used for inference.
- `coco.txt`, `safety_helmet_vest_mask.txt`: class/label files.

Requirements
- Python 3.8+ and PyTorch compatible with your CUDA (or CPU-only).
- Typical dependencies (example): `ultralytics`, `torch`, `opencv-python`, `numpy`.
- You can install common dependencies with:

```bash
python -m pip install -U pip
pip install ultralytics torch torchvision opencv-python numpy
```

Usage
- From the project folder run inference (examples):

```bash
cd safety_equipment_helmet_vest
python main.py
```

- If `main.py` accepts a `--weights` argument, specify the provided model:

```bash
python main.py --weights best38.pt
```

Notes
- This repository does not include a `requirements.txt`; adjust the package list above as needed for your environment.
- If you run into CUDA/PyTorch compatibility issues, install a matching `torch` build from https://pytorch.org.

Contact
- For questions, open an issue or contact the maintainer.

Output
- The script will save or display detection results depending on how `main.py` is implemented.

