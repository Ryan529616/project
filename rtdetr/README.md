# rtdetr_rggb

RT-DETR baseline for RGGB object detection.

## Key point

This project reuses the original data pipeline from `project/transformer/data`:
- `.npy` RGGB loading
- resize / keep-aspect / pad-to-multiple
- DETR target construction (`targets`)

So input preprocessing behavior stays aligned with your existing `transformer` setup.

## Train

```bash
cd /home/ryan529/project/rtdetr_rggb
./tools/train.sh
```

## Override options

```bash
./tools/train.sh --epochs 150 --batch-size 12 --out /home/ryan529/project/rtdetr_rggb/runs/exp1
```

## Main files

- `models/rtdetr_rggb.py`: RT-DETR model definition
- `train_rtdetr_rggb.py`: full training/eval/checkpoint loop
- `configs/train_rtdetr_rggb.json`: default training config
