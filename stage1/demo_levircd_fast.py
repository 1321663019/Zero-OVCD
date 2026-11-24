import os
from pathlib import Path
import argparse
import torch
import numpy as np
from skimage.io import imread, imsave
from dynamic_earth.comparator.sam_ape_num_fast import bitemporal_match
from dynamic_earth.utils import get_model_and_processor
from smfm_identifier import get_identifier
from tqdm import tqdm
import cv2
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
import json
from mmseg.structures import SegDataSample
from scipy.ndimage import binary_fill_holes
def _cc_instances(mask_bool, area_min=500, fill_holes=True):
    """bool(H,W) -> np.ndarray(N,H,W)"""
    H, W = mask_bool.shape
    u8 = (mask_bool.astype(np.uint8) * 255)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(u8, connectivity=8)
    inst = []
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < area_min:
            continue
        m = (labels == i)
        if fill_holes:
            m = binary_fill_holes(m)
        inst.append(m.astype(bool))
    if len(inst) == 0:
        return np.zeros((0, H, W), dtype=bool)
    return np.stack(inst, axis=0).astype(bool)

def _union(instances, H, W):
    if instances.shape[0] == 0:
        return np.zeros((H, W), dtype=bool)
    return np.any(instances, axis=0)

def _mask_bboxes(masks: np.ndarray):
    """每个 bool mask 的 (x0,y0,x1,y1)；开区间终点，空掩码为零框。"""
    boxes = []
    for m in masks:
        x, y, w, h = cv2.boundingRect(np.ascontiguousarray(m, dtype=np.uint8))
        boxes.append((x, y, x + w, y + h))
    return boxes


def _exact_intersection(m1, m2, b1, b2):
    """只计算交叠矩形内的交集像素数，与整图 (m1 & m2).sum() 精确相等。"""
    x0, y0 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x1, y1 = min(b1[2], b2[2]), min(b1[3], b2[3])
    if x0 >= x1 or y0 >= y1:
        return 0
    return int(np.count_nonzero(m1[y0:y1, x0:x1] & m2[y0:y1, x0:x1]))


def _match_delete_both(inst1, inst2,
                       overlap_ratio_thresh=0.8,
                       area_large_thresh=6000,
                       area_ratio_max_big=1.5,
                       area_ratio_max_small=4.0):
    """匹配成功（inter/min(area)高且面积差不大） -> 两边都删"""
    inst1 = inst1.astype(bool, copy=False)
    inst2 = inst2.astype(bool, copy=False)
    N1, N2 = inst1.shape[0], inst2.shape[0]
    if N1 == 0 or N2 == 0:
        return inst1.copy(), inst2.copy()

    a1 = inst1.reshape(N1, -1).sum(axis=1).astype(np.int64)
    a2 = inst2.reshape(N2, -1).sum(axis=1).astype(np.int64)

    boxes1 = _mask_bboxes(inst1)
    boxes2 = _mask_bboxes(inst2)
    rm1 = np.zeros(N1, dtype=bool)
    rm2 = np.zeros(N2, dtype=bool)

    for i in range(N1):
        if rm1[i] or a1[i] == 0:
            rm1[i] = True
            continue
        m1 = inst1[i]

        for j in range(N2):
            if rm2[j] or a2[j] == 0:
                rm2[j] = True if a2[j] == 0 else rm2[j]
                continue

            m2 = inst2[j]
            inter = _exact_intersection(m1, m2, boxes1[i], boxes2[j])
            if inter == 0:
                continue

            small = min(a1[i], a2[j])
            large = max(a1[i], a2[j])
            if small == 0:
                continue

            overlap_ratio = inter / float(small)
            if overlap_ratio < overlap_ratio_thresh:
                continue

            area_ratio_max_local = area_ratio_max_big if large >= area_large_thresh else area_ratio_max_small
            if large <= area_ratio_max_local * small:
                rm1[i] = True
                rm2[j] = True
                break  # 一对一，避免连坐

    return inst1[~rm1], inst2[~rm2]

def add_change_with_building_cam(
    cam_b1: np.ndarray,
    cam_b2: np.ndarray,
    thr_high: float = 0.75,
    thr_low: float = 0.25,
    area_min: int = 500,
    overlap_ratio_thresh=0.8,
):
    if cam_b1.shape != cam_b2.shape:
        raise ValueError(f"cam_b1/cam_b2 尺寸不一致: {cam_b1.shape} vs {cam_b2.shape}")
    H, W = cam_b1.shape

    # 1) high/low
    mask_b1_high = cam_b1 > thr_high
    mask_b2_high = cam_b2 > thr_high
    mask_b1_low  = cam_b1 < thr_low
    mask_b2_low  = cam_b2 < thr_low

    # 2) high 先实例化
    inst_b1 = _cc_instances(mask_b1_high, area_min=area_min, fill_holes=True)
    inst_b2 = _cc_instances(mask_b2_high, area_min=area_min, fill_holes=True)

    # 3) 实例匹配：重叠高 -> 两边都删（认为不变，解决细边）
    inst_b1_left, inst_b2_left = _match_delete_both(
        inst_b1, inst_b2,
        overlap_ratio_thresh= overlap_ratio_thresh,
        area_large_thresh=6000,
        area_ratio_max_big=1.5,
        area_ratio_max_small=4.0,
    )

    # 4) 合并剩余 high -> high_clean
    b1_high_clean = _union(inst_b1_left, H, W)
    b2_high_clean = _union(inst_b2_left, H, W)

    # 5) high_clean + low 定义新建/拆除
    add_new    = b2_high_clean & mask_b1_low
    remove_old = b1_high_clean & mask_b2_low
    add_mask_bool = add_new | remove_old
    if not add_mask_bool.any():
        return []

    # 6) 对 add_mask 再 CC>=area_min + fill holes，输出实例列表
    inst_add = _cc_instances(add_mask_bool, area_min=area_min, fill_holes=True)
    return [inst_add[k].astype(bool) for k in range(inst_add.shape[0])]
def filter_by_overlap_and_area(
    pred_instances: np.ndarray,
    pseudo_instances: np.ndarray,
    overlap_ratio_thresh: float = 0.8,
    area_large_thresh: int = 6000,
    area_ratio_max_big: float = 1.5,
    area_ratio_max_small: float = 4.0,
):
    """
    在预测实例集和伪标签实例集之间做互相过滤：
      - 对每一对 (pred_i, pseudo_j)，若：
          1) 重叠比 >= overlap_ratio_thresh
          2) 两者中面积大的不超过面积小的 (1.5× 或 4×)，
             具体倍率由“最大面积是否超过 area_large_thresh”决定：
               - 若 a_large >= area_large_thresh:  使用 area_ratio_max_big (默认 1.5)
               - 否则:                             使用 area_ratio_max_small (默认 4.0)
        则：保留较小掩码，删除较大掩码（来源不变，只是在各自集合中删掉它）。

    重叠比定义：
      overlap_ratio = intersection_area / min(area_pred, area_pseudo)
    """

    pred_instances = pred_instances.astype(bool, copy=False)
    pseudo_instances = pseudo_instances.astype(bool, copy=False)

    Np = pred_instances.shape[0]
    Ns = pseudo_instances.shape[0]

    if Np == 0 or Ns == 0:
        # 有一边空，直接原样返回
        return pred_instances.copy(), pseudo_instances.copy()

    # 计算面积
    pred_areas = pred_instances.reshape(Np, -1).sum(axis=1).astype(np.int64)     # [Np]
    pseudo_areas = pseudo_instances.reshape(Ns, -1).sum(axis=1).astype(np.int64) # [Ns]

    pred_boxes = _mask_bboxes(pred_instances)
    pseudo_boxes = _mask_bboxes(pseudo_instances)
    remove_pred = np.zeros(Np, dtype=bool)
    remove_pseudo = np.zeros(Ns, dtype=bool)

    for i in range(Np):
        if pred_areas[i] == 0:
            remove_pred[i] = True
            continue
        pi = pred_instances[i]

        for j in range(Ns):
            if pseudo_areas[j] == 0:
                remove_pseudo[j] = True
                continue
            pj = pseudo_instances[j]

            inter = _exact_intersection(pi, pj, pred_boxes[i], pseudo_boxes[j])
            if inter == 0:
                continue

            a_pred = pred_areas[i]
            a_pseudo = pseudo_areas[j]

            a_small = min(a_pred, a_pseudo)
            a_large = max(a_pred, a_pseudo)
            if a_small == 0:
                continue

            # 重叠比：交集 / 小掩码面积
            overlap_ratio = inter / float(a_small)

            if overlap_ratio < overlap_ratio_thresh:
                continue

            # 按最大面积大小，选择不同的面积比例上限
            if a_large >= area_large_thresh:
                area_ratio_max_local = area_ratio_max_big   # 比如 1.5
            else:
                area_ratio_max_local = area_ratio_max_small # 比如 4.0

            if a_large <= area_ratio_max_local * a_small:
                # 满足条件：两个掩码高度重叠，且面积差不大 → 删掉较大的那个
                if a_pred > a_pseudo:
                    remove_pred[i] = True
                elif a_pseudo > a_pred:
                    remove_pseudo[j] = True
                else:
                    # 面积相等：默认删预测那边
                    remove_pred[i] = True

    pred_filtered = pred_instances[~remove_pred]
    pseudo_filtered = pseudo_instances[~remove_pseudo]

    return pred_filtered, pseudo_filtered
def decode_rle(rle):
    """与原实现相同的 C-order RLE 解析；避免逐像素构造 Python list。"""
    h, w = map(int, rle["size"])
    flat = np.empty(h * w, dtype=np.uint8)
    offset, value = 0, 0
    for count in rle["counts"]:
        count = int(count)
        if count < 0 or offset + count > h * w:
            raise ValueError("RLE counts 与图像大小不一致")
        flat[offset:offset + count] = value
        offset += count
        value = 1 - value
    if offset != h * w:
        raise ValueError("RLE counts 与图像大小不一致")
    return flat.reshape(h, w)


def load_sam3_json_masks(json_path):
    """
    从 SAM3 生成的 json 读取所有实例掩码，返回：
    masks: (N, H, W) bool 数组（N 可能为 0）
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rles = data.get("rles", [])
    if len(rles) == 0:
        # 没有掩码，就返回一个空的 (0, H, W) 或 (0,0,0)，视你需要
        h = data["height"]
        w = data["width"]
        return np.zeros((0, h, w), dtype=bool)

    masks = []
    for rle in rles:
        m = decode_rle(rle)          # (H, W) 0/1
        masks.append(m.astype(bool))

    masks = np.stack(masks, axis=0)  # (N, H, W)
    return masks
def load_sam3_pair_masks(json1_path, json2_path):
    """
    读取 img1 / img2 对应的 SAM3 json，
    返回:
      masks_all: (N1+N2, H, W) bool
      img1_mask_num: N1
    """
    masks1 = load_sam3_json_masks(json1_path)  # (N1, H, W)
    masks2 = load_sam3_json_masks(json2_path)  # (N2, H, W)

    img1_mask_num = masks1.shape[0]

    if img1_mask_num == 0 and masks2.shape[0] == 0:
        # 两边都没有掩码，保持空间尺寸一致，返回 (0, H, W)
        return masks1, 0

    if img1_mask_num == 0:
        masks_all = masks2
    elif masks2.shape[0] == 0:
        masks_all = masks1
    else:
        masks_all = np.concatenate([masks1, masks2], axis=0)  # (N1+N2, H, W)

    return masks_all, img1_mask_num
def refine_sam_with_ape(ape_masks: np.ndarray,
                        sam_masks: np.ndarray,
                        area_ratio_thresh: float = 0.6,
                        max_contained_instances: int = 4,
                        contain_frac: float = 0.95,
                        min_check_area: int = 5200   # 仅当 sm_area ≥ 此值才做“包含数量”判断
                        ) -> np.ndarray:
    """
        仅返回：被 APE 掩码抠除后，且剩余面积比例≥阈值 的 SAM 掩码集合。
        额外规则：若某 SAM 掩码“完整包含”(overlap/ape_area ≥ contain_frac) 的 APE 实例数
                 ≥ max_contained_instances，则直接丢弃该 SAM 掩码。
        改动点：
          1) 先做面积比判定；
          2) 仅对“面积≥min_check_area”的 SAM 再做 APE 包含数量判断。
    """
    # 兜底为 bool
    ape_masks = ape_masks.astype(bool, copy=False)
    sam_masks = sam_masks.astype(bool, copy=False)

    # 空输入的快捷路径
    if sam_masks.size == 0:
        return np.zeros_like(sam_masks)  # (0,H,W)
    H, W = sam_masks.shape[-2], sam_masks.shape[-1]
    if ape_masks.size == 0:
        # 没有 APE，SAM 原样返回
        return sam_masks

    # 形状一致性检查（避免静默错位）
    assert ape_masks.shape[-2:] == (H, W), "APE/SAM 的 H,W 不一致，请先对齐尺寸"

    ape_union = np.any(ape_masks, axis=0)  # APE 并集 [H,W]，作为“禁止覆盖区”
    ape_union_fan = ~ape_union
    # 预备：每个 APE 实例的面积（用于“完整包含”判定）
    ape_areas = ape_masks.reshape(ape_masks.shape[0], -1).sum(axis=1).astype(np.int64)  # [M]

    keep_list = []
    for sm in sam_masks:
        sm_area = int(sm.sum())
        if sm_area == 0:
            continue

        # —— 先做“面积比”判定（与原逻辑相同，只是提前）——
        sm_clean = sm & (ape_union_fan)
        sm_clean_area = int(sm_clean.sum())
        ratio = (sm_clean_area / sm_area) if sm_area > 0 else 0.0
        if not (ratio >= area_ratio_thresh and sm_clean_area > 0):
            continue

        # —— 小掩码：跳过“包含数量”判断，直接保留 ——
        if sm_area < int(min_check_area):
            keep_list.append(sm)
            continue

        # —— 大掩码：再做“APE 包含数量”判断（与原计算方式一致）——
        # (1,H,W)&(M,H,W)->(M,H,W) → overlap_px: [M]
        overlap_px = (sm[None, :, :] & ape_masks).reshape(ape_masks.shape[0], -1).sum(axis=1)
        overlap_frac_wrt_ape = overlap_px / ape_areas  # 相对 APE 的重叠比例
        num_contained = (overlap_frac_wrt_ape >= float(contain_frac)).sum()

        if num_contained >= int(max_contained_instances):
            # SAM 覆盖了太多“几乎完整”的 APE 实例 → 丢弃
            continue

        keep_list.append(sm)

    if len(keep_list) == 0:
        return np.zeros((0, H, W), dtype=bool)

    return np.stack(keep_list, axis=0).astype(bool)
def sam3_masks_to_np_bool(masks_torch: torch.Tensor) -> np.ndarray:
    """
    将 SAM3 输出的 masks 转成 numpy.bool_，形状 [N,H,W]
    自动兼容 NCHW / NHWC / 单通道 / 多通道。
    """
    if masks_torch is None or masks_torch.numel() == 0:
        return np.zeros((0, 0, 0), dtype=bool)

    # 在 GPU 上先阈值化，再拷贝 bool，避免传输浮点掩码的大量数据；>0.5 判定不变。
    m = masks_torch.detach()
    if m.dtype.is_floating_point:
        m = m > 0.5
    m = m.cpu()

    # 处理不同维度
    if m.dim() == 3:
        # [N,H,W] -> 直接返回
        pass
    elif m.dim() == 4:
        # 可能有两种情况：
        # 1) NCHW: [N,1,H,W] / [N,3,H,W]
        # 2) NHWC: [N,H,W,1] / [N,H,W,3]
        if m.shape[1] in (1, 3) and m.shape[-1] not in (1, 3):
            # 认为是 NCHW
            if m.shape[1] == 1:
                m = m[:, 0]              # [N,H,W]
            else:
                m = m.any(dim=1)         # [N,H,W]，3 通道合并
        elif m.shape[-1] in (1, 3):
            # 认为是 NHWC
            if m.shape[-1] == 1:
                m = m[..., 0]            # [N,H,W]
            else:
                m = m.any(dim=-1)        # [N,H,W]
        else:
            # 非预期形状，保守起见沿最后一维做 any
            m = m.any(dim=-1)
    else:
        raise ValueError(f"SAM3 masks 维度不支持: {m.shape}")

    return m.numpy().astype(bool)
def merge_masks(change_masks, shape):
    """Merges individual change masks into a single change mask.

    Args:
        change_masks (list of np.array): List of individual change masks.
        shape (tuple): Shape of the output mask (height, width).

    Returns:
        np.array: Merged binary change mask.
    """
    if len(change_masks) == 0:
        return np.zeros((shape[0], shape[1]), dtype=np.uint8)

    # Sum the masks and convert to binary (255 for changed areas)
    change_mask = np.sum(change_masks, axis=0).astype(np.uint8)
    change_mask[change_mask > 0] = 255

    return change_mask


def parse_arguments() -> argparse.Namespace:
    """Paths for LEVIR-CD images, pre-generated SAM3 JSON masks and checkpoints."""
    parser = argparse.ArgumentParser(description="Zero-OVCD Stage I Fast Demo (LEVIR-CD)")
    parser.add_argument("--input_image_1", default="data/LEVIR-CD/test/A", help="T1 images")
    parser.add_argument("--input_image_2", default="data/LEVIR-CD/test/B", help="T2 images")
    parser.add_argument("--sam3_json_a", default="data/LEVIR-CD/JSON/test/A", help="T1 SAM3 automatic-mask JSON")
    parser.add_argument("--sam3_json_b", default="data/LEVIR-CD/JSON/test/B", help="T2 SAM3 automatic-mask JSON")
    parser.add_argument("--output_dir", "-o", default="outputs/levircd_stage1_fast")
    parser.add_argument("--sam3_checkpoint", default="checkpoint/sam3.pt")
    parser.add_argument("--sam3_bpe_path", default="third_party/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz")
    parser.add_argument("--dinov3_repo", default="third_party/dinov3", help="Local DINOv3 repository")
    parser.add_argument("--dinov3_checkpoint", default="checkpoint/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth")
    parser.add_argument("--classname_path", default=str(Path(__file__).resolve().parent / "class_name.txt"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--name_list", nargs="+", default=["ground,road,vegetation", "building,roof,house"])
    return parser.parse_args()


def load_images(img1_path: str, img2_path: str) -> tuple:
    """
    Load images from the given paths.

    Args:
        img1_path (str): Path to the first image file.
        img2_path (str): Path to the second image file.

    Returns:
        tuple: Loaded images (img1, img2).
    """
    img1 = imread(img1_path)
    img2 = imread(img2_path)
    return img1, img2
@torch.no_grad()
def filter_classes_by_reliability_fast(seg_logit, mask_data, tau=0.4, bg_idx=0, eps=1e-6):
    """不改变原始 mask 平均、margin 或背景筛选，仅省去同尺寸的无意义插值。"""
    mask_classes = []
    same_shape = tuple(mask_data.shape[-2:]) == tuple(seg_logit.shape[-2:])
    for start in range(0, len(mask_data), 8):
        if same_shape:
            masks_gpu = torch.from_numpy(
                np.ascontiguousarray(mask_data[start:start + 8])
            ).to(device=seg_logit.device, dtype=torch.bool)
        else:
            masks_gpu = None
        for offset in range(min(8, len(mask_data) - start)):
            if same_shape:
                mask_gpu = masks_gpu[offset]
            else:
                proposal_mask = torch.tensor(
                    mask_data[start + offset], device=seg_logit.device
                ).to(torch.uint8)
                mask_gpu = torch.nn.functional.interpolate(
                    proposal_mask[None, None].float(),
                    size=seg_logit.shape[-2:], mode="nearest"
                ).squeeze(0).squeeze(0).to(torch.bool)

            mask_logit = seg_logit[:, mask_gpu].mean(dim=1)
            score_sum = mask_logit.sum().clamp_min(eps)
            mask_score = mask_logit / score_sum
            mask_pred = mask_score.argmax(0)
            if mask_pred.item() != bg_idx:
                pred_score = mask_score.max(0)[0]
                bg_score = mask_score[bg_idx]
                reliability_score = pred_score - bg_score
                if reliability_score.item() < tau:
                    mask_pred = torch.tensor(bg_idx, device=seg_logit.device)
            mask_classes.append(mask_pred.item())
    return mask_classes


@torch.no_grad()
def identify_fast(img1_path, img2_path, mask_data, img1_mask_num,
                  model, processor, device="cuda", is_instance_class=True):
    """SMFM: equivalent class decisions to the original SegEarth-OV3 identifier.

    The original first-pass per-proposal classes were unused. Query the same
    multiscale semantic maps once, then retain the original tau=0.1 margin rule.
    """
    results = []
    # 附带 data_samples 时，上传的 SegEarthOV3Segmentation.predict() 完全不读取 inputs；
    # 因此不再预先对图像重复进行 PIL 解码、ToTensor 和 CUDA 拷贝。
    empty_input = torch.empty((1, 3, 1, 1), device=device)
    for img_path in (img1_path, img2_path):
        with Image.open(img_path) as im:
            # ori_shape 在 mmseg 中为 (H,W)，纠正非正方形图像的宽高顺序。
            h, w = im.height, im.width
        data_sample = SegDataSample()
        data_sample.set_metainfo({'img_path': img_path, 'ori_shape': (h, w)})
        _unused_classes, seg_logits = model.predict(
            empty_input, data_samples=[data_sample], proposal_masks=()
        )
        # 无须修改实际生效的类别/裕度筛选，保证相同概率张量与 tau。
        classes = filter_classes_by_reliability_fast(
            seg_logits, mask_data, tau=0.1, bg_idx=0
        )
        results.append((classes, seg_logits))

    img1_classes, logit1 = results[0]
    img2_classes, logit2 = results[1]
    if is_instance_class:
        class_source = img1_classes[:img1_mask_num] + img2_classes[img1_mask_num:]
        keep = ((np.asarray(img1_classes) != np.asarray(img2_classes))
                & np.asarray(class_source).astype(bool))
    else:
        keep = (np.asarray(img1_classes) != np.asarray(img2_classes))
    indices = np.where(keep)
    cmasks = np.asarray(mask_data)[indices]
    n1 = int(np.sum(indices[0] < img1_mask_num))
    return cmasks, n1, logit1, logit2


@torch.no_grad()
def main(args):
    comparetor_config = {"model_type": "DINOv3", "feature_dim": 1024, "patch_size": 16}
    os.makedirs(args.output_dir, exist_ok=True)
    image_files_1 = sorted(os.listdir(args.input_image_1))
    image_files_2 = sorted(os.listdir(args.input_image_2))
    # Load frozen SAM3 with the same checkpoint and tokenizer used in the original experiment.
    model = build_sam3_image_model(
        checkpoint_path=args.sam3_checkpoint,
        bpe_path=args.sam3_bpe_path,
    ).to(args.device)
    model.eval()
    processorSAM3 = Sam3Processor(model,confidence_threshold=0.2)
    # Load models
    comparator_model, comparator_processor = get_model_and_processor(
        comparetor_config["model_type"], args.device,
        dinov3_checkpoint=args.dinov3_checkpoint,
        dinov3_repo=args.dinov3_repo,
    )
    comparator_model.eval()
    identifier_model, identifier_processor = get_identifier(
        "SegEarth-OV3", args.device, name_list=args.name_list,
        model_config={
            "classname_path": args.classname_path,
            "sam3_checkpoint": args.sam3_checkpoint,
            "sam3_bpe_path": args.sam3_bpe_path,
        },
    )
    identifier_model.eval()
    sam3autodirA = args.sam3_json_a
    sam3autodirB = args.sam3_json_b
    for img1_file, img2_file in tqdm(zip(image_files_1, image_files_2),total=min(len(image_files_1), len(image_files_2))):
        # Load images
        img1_path = os.path.join(args.input_image_1, img1_file)
        img2_path = os.path.join(args.input_image_2, img2_file)
        img1, img2 = load_images(img1_path, img2_path)
        base, _ = os.path.splitext(img1_file)
        json1_path = os.path.join(sam3autodirA, f"{base}.json")
        json2_path = os.path.join(sam3autodirB, f"{base}.json")
        masks, img1_mask_num = load_sam3_pair_masks(json1_path, json2_path)
        sam_masks_img1 = masks[:img1_mask_num]
        sam_masks_img2 = masks[img1_mask_num:]
        pil_img1 = Image.fromarray(img1) if not isinstance(img1, Image.Image) else img1
        pil_img2 = Image.fromarray(img2) if not isinstance(img2, Image.Image) else img2
        inference_state1 = processorSAM3.set_image(pil_img1)
        inference_state2 = processorSAM3.set_image(pil_img2)
        ape_img1_masks= processorSAM3.set_text_prompt(state=inference_state1, prompt="house")
        ape_img2_masks= processorSAM3.set_text_prompt(state=inference_state2, prompt="house")
        ape_img1_masks = ape_img1_masks["masks"]
        ape_img2_masks = ape_img2_masks["masks"]
        ape_img1_masks = sam3_masks_to_np_bool(ape_img1_masks)  # [N1,H,W] bool
        ape_img2_masks = sam3_masks_to_np_bool(ape_img2_masks)  # [N2,H,W] bool
        # Match empty text-prompt mask shapes to the input image dimensions.
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        if ape_img1_masks.size == 0:
            ape_img1_masks = np.zeros((0, h1, w1), dtype=bool)
        if ape_img2_masks.size == 0:
            ape_img2_masks = np.zeros((0, h2, w2), dtype=bool)
        # MRM: refine SAM3 automatic proposals against text-guided masks.
        sam_masks_img1 = refine_sam_with_ape(ape_img1_masks, sam_masks_img1)
        sam_masks_img2 = refine_sam_with_ape(ape_img2_masks, sam_masks_img2)
        # Record the four proposal partitions for cross-temporal filtering.
        sam1 = len(sam_masks_img1)
        ape1 = len(ape_img1_masks)
        sam2 = len(sam_masks_img2)
        ape2 = len(ape_img2_masks)
        img1_masks = np.concatenate([sam_masks_img1, ape_img1_masks], axis=0)
        img2_masks = np.concatenate([sam_masks_img2, ape_img2_masks], axis=0)
        img1_mask_num = len(img1_masks)
        masks = np.concatenate([img1_masks, img2_masks], axis=0)
        # Cross-temporal change proposal filtering using frozen DINOv3.
        cmasks,img1_mask_num,stats= bitemporal_match(img1, img2, masks, comparator_model, comparator_processor,
                                                 img1_mask_num, change_confidence_threshold=125, device=args.device,
                                                 model_config=comparetor_config,sam_ape_counts=(sam1, ape1, sam2, ape2))
        SAM_T1 = stats["SAM_T1"]
        SAM_T2 = stats["SAM_T2"]
        sam_masks_t1=cmasks[:SAM_T1]
        sam_masks_t2 = cmasks[img1_mask_num:img1_mask_num+SAM_T2]
        ape_masks_t1 = cmasks[SAM_T1:img1_mask_num]
        ape_masks_t2 = cmasks[img1_mask_num+SAM_T2:]
        sammasks_dinov3 = np.concatenate([sam_masks_t1, sam_masks_t2], axis=0)
        img1_mask_num=len(sam_masks_t1)
        # SMFM: SegEarth-OV3 multiscale semantic maps and margin-based verification.
        cmasks, img1_mask_num,seg_logit1,seg_logit2 = identify_fast(
            img1_path, img2_path, sammasks_dinov3, img1_mask_num,
            identifier_model, identifier_processor,
            device=args.device
        )
        cmasks1 = np.concatenate([cmasks[:img1_mask_num],  ape_masks_t1], axis=0)
        cmasks2 = np.concatenate([cmasks[img1_mask_num:], ape_masks_t2], axis=0)
        cmasks=np.concatenate([cmasks1,cmasks2], axis=0)
        seg_logits1 = seg_logit1.detach().cpu().numpy()
        seg_logits2 = seg_logit2.detach().cpu().numpy()
        cam_building1 = seg_logits1[1]
        cam_building2 = seg_logits2[1]
        # MCCM: CAM-guided false-positive rejection, correction and completion.
        kept, removed_idx = [], []
        # Precompute four CAM thresholds once per image pair.
        high1_04, high2_04 = cam_building1 >= 0.4, cam_building2 >= 0.4
        high1_09, high2_09 = cam_building1 >= 0.9, cam_building2 >= 0.9
        for i, m in enumerate(cmasks):
            m_bool = m > 0
            area = int(np.count_nonzero(m_bool))
            if area == 0:
                # 空实例不贡献任何像素，也不参与 MCCM。
                removed_idx.append(i)
                continue
            th1_large = int(np.count_nonzero(high1_04[m_bool])) / area
            th2_large = int(np.count_nonzero(high2_04[m_bool])) / area
            th1_small = int(np.count_nonzero(high1_09[m_bool])) / area
            th2_small = int(np.count_nonzero(high2_09[m_bool])) / area
            if ((th1_small <= 0.6 and th2_small <= 0.6 and area < 500)
                    or (th1_large <= 0.6 and th2_large <= 0.6 and area > 500)):
                removed_idx.append(i)
            else:
                kept.append(m)
        H, W = img1.shape[:2]
        addmask = add_change_with_building_cam(
            cam_building1,
            cam_building2,
        )
        addmask = np.stack(addmask, 0).astype(bool) if len(addmask) else np.zeros((0, H, W), dtype=bool)
        kept = np.stack(kept, 0).astype(bool) if len(kept) else np.zeros((0, H, W), dtype=bool)

        kept = refine_sam_with_ape(addmask,kept)
        addmask,kept=filter_by_overlap_and_area(addmask,kept)
        change_mask = np.concatenate([addmask,kept], axis=0)
        change_mask = merge_masks(change_mask, img1.shape[:2])  # .astype(bool)
        imsave(os.path.join(args.output_dir, img1_file), change_mask)
        print(f"save:{img1_file}")

        if torch.cuda.memory_reserved() > 18 * 1024 ** 3:
            torch.cuda.empty_cache()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)