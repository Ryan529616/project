# baseline (vanilla DETR)
python /home/ryan529/project/transformer/tools/infer_vanilla_detr_rggb.py \
  --input /home/ryan529/project/bdd100k/raw10_npy/val \
  --ckpt /home/ryan529/project/transformer/runs/vanilla_detr_rggb_official_like_e180_b12/best_mAP50.pt \
  --out /home/ryan529/project/transformer/runs/infer_vanilla \
  --device auto --timeit
