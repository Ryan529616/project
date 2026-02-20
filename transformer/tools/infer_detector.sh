# 主模型 (hardware-aware)
python /home/ryan529/project/transformer/tools/infer_hardware_aware_detector.py \
  --input /home/ryan529/project/bdd100k/raw10_npy/val \
  --ckpt /home/ryan529/project/transformer/runs/hardware_aware_bdd100k_v1_init_best_20260218_192207/ckpt/best_mAP50.pt \
  --out /home/ryan529/project/transformer/runs/infer_hw \
  --device auto --timeit
