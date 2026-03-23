# [CVPR'26] VVS: Accelerating Speculative Decoding for Visual Autoregressive Model via Partial Verification Skipping

## 📦 Installation
```bash
conda create -n vvs python=3.10
conda activate vvs
pip install -r requirements.txt
```

## 🚀 Preparation

Please refer to [LANTERN](https://github.com/jadohu/LANTERN) to prepare draft model checkpoints. The verification skipping mechanism is primarily implemented in:

```
models/
├── ea_llamagen.py
├── skip_utils_llamagen.py
└── utils_llamagen.py
run_vvs.sh
```

Run `bash run_vvs.sh` to evaluate the VVS baseline.


## 📜 Citation
 
If you find our work useful, please cite:

```
@article{dong2025vvs,
  title={VVS: Accelerating Speculative Decoding for Visual Autoregressive Generation via Partial Verification Skipping},
  author={Dong, Haotian and Li, Ye and Lu, Rongwei and Tang, Chen and Xia, Shu-Tao and Wang, Zhi},
  journal={arXiv preprint arXiv:2511.13587},
  year={2025}
}
```

## 🤝 Acknowledgement

This project is built upon the following great open-source works:

- [LANTERN](https://github.com/jadohu/LANTERN/tree/main)
- [EAGLE](https://github.com/SafeAILab/EAGLE)
