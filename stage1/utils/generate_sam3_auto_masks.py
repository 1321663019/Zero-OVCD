from transformers import pipeline
from PIL import Image
import os
from tqdm import tqdm
import numpy as np
import json
# ========= 新增：RLE 编解码 =========
def encode_rle(mask_np: np.ndarray):
    """
    把 (H, W) 的 0/1 掩码编码成简单 RLE：
    返回 {"size": [H, W], "counts": [run0, run1, ...]}，从背景0开始。
    """
    mask_np = mask_np.astype(np.uint8)
    h, w = mask_np.shape
    flat = mask_np.ravel()

    counts = []
    current = 0  # 当前值（0 or 1）
    run_len = 0

    # 保证第一段是背景0
    idx0 = 0
    if flat[0] == 1:
        counts.append(0)
        current = 1
    else:
        current = 0

    for v in flat:
        if v == current:
            run_len += 1
        else:
            counts.append(run_len)
            run_len = 1
            current = v
    counts.append(run_len)

    return {"size": [h, w], "counts": counts}
def decode_rle(rle):
    """
    以后你要从 json 里恢复掩码时可以用：
    mask_np = decode_rle(data["rles"][i])   # (H, W) 0/1
    """
    h, w = rle["size"]
    counts = rle["counts"]
    flat = []
    v = 0  # 从背景0开始
    for c in counts:
        flat.extend([v] * c)
        v = 1 - v
    flat = np.array(flat, dtype=np.uint8)
    return flat.reshape(h, w)
# ========= RLE 工具结束 =========
# 1. 本地 HF 模型目录
local_sam3_dir = r"D:\pyk\xm\DynamicEarth\checkpoint\SAM3"
device = 0
generator = pipeline(
    "mask-generation",
    model=local_sam3_dir,
    device=device,
)

input_dir = r"D:\pyk\xm\DynamicEarth\data_test\SECOND\val\B"
save_dir = r"D:\pyk\xm\DynamicEarth\result\SECOND\JSON\val\B"
os.makedirs(save_dir, exist_ok=True)
image_files_1 = sorted(os.listdir(input_dir))
for img1_file in tqdm(image_files_1, total=len(image_files_1)):
    img1_path = os.path.join(input_dir, img1_file)
    img1 = Image.open(img1_path).convert("RGB")
    # 4. 自动掩码生成
    outputs = generator(img1, points_per_batch=64,
                        points_per_side=32,
            pred_iou_thresh=0.5,
            stability_score_thresh=0.95,
            stability_score_offset=0.9,
            box_nms_thresh=0.7,
            min_mask_region_area=0,
            match_hist=False,
            area_thresh=0.8)

    print("num masks:", len(outputs["masks"]))

    masks = outputs["masks"]  # list，元素是 [H, W] 的掩码（bool 或 0/1）

    # ========= 修改点 1：不再存巨大 0/1 矩阵，而是存 RLE =========
    rles = []  # 之后 json 里的 data["rles"][i] 就是第 i 个实例的 RLE

    for m in masks:
        # 兼容 torch.Tensor / numpy
        if hasattr(m, "cpu"):
            m_np = m.cpu().numpy()
        else:
            m_np = np.array(m)

        # 保证是 (H, W) 的 0/1 掩码
        if m_np.ndim == 3:
            m_np = m_np.squeeze()
        m_np = (m_np > 0).astype(np.uint8)

        # 编码成 RLE（无损）
        rle = encode_rle(m_np)
        rles.append(rle)

    # ========= 修改点 2：每张图一定会写一个 json（即使没有掩码） =========
    w, h = img1.size
    data = {
        "image_filename": img1_file,
        "width": w,
        "height": h,
        "rles": rles,  # ★ 将来你用 decode_rle(data["rles"][i]) 得到第 i 个实例掩码
    }

    base, _ = os.path.splitext(img1_file)
    json_path = os.path.join(save_dir, f"{base}.json")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    print("RLE 掩码已保存到 JSON:", json_path)
