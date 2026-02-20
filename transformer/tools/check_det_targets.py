# tools/check_det_targets.py
import json, sys
p = sys.argv[1]
data = json.load(open(p))
def count_boxes(item):
    # 依你的 schema 改：下面是常見兩種
    if "boxes" in item: return len(item["boxes"])
    if "labels" in item and "bbox" in item["labels"][0]: return len(item["labels"])
    return 0

n = [count_boxes(x) for x in data[:2000]]
print("items:", len(data), "non_empty:", sum(i>0 for i in n), "max:", max(n, default=0))
print("first 20:", n[:20])
