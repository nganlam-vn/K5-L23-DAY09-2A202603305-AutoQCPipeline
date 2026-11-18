"""CHECK 6 — Ý kiến thứ hai: so nhãn với một model LỚN HƠN (yolo26l).   ★★★ (khó nhất, đáng điểm nhất)

Test:  pytest tests/test_second_opinion.py -q

Máy kiểm quy tắc, người kiểm ý nghĩa (slide). Các lỗi như thiếu nhãn / gán nhầm lớp hợp lệ / box lỏng
KHÔNG bắt được bằng quy tắc cú pháp. Ta nhờ một model mạnh hơn "nhìn lại" và CHỈ CẢNH BÁO (model cũng có thể sai):
những ảnh bị cờ sẽ vào hàng đợi cho NGƯỜI duyệt.

`preds[stem]` = list `[id_lớp, conf, cx, cy, w, h]` của model (lấy qua `get_preds(ds, cfg, "checker")` nếu preds=None).
`L = ds.valid_boxes(stem)` = list `(id_lớp, cx, cy, w, h)` của nhãn (đã bỏ dòng hỏng).
Ngưỡng ở `cfg["second_opinion"]`: conf, min_area_px, fp_conf, iou_missing, iou_match, iou_loose.
"""
from ..common import WARN, Issue, iou_xywh, parse_yolo_line
from ..model_cache import get_preds


def run(ds, cfg, preds=None):
    if preds is None:
        preds = get_preds(ds, cfg, "checker")

    so_cfg = cfg.get("second_opinion", {})
    conf_thresh = so_cfg.get("conf", 0.5)
    min_area_px = so_cfg.get("min_area_px", 600)
    fp_conf = so_cfg.get("fp_conf", 0.25)
    iou_missing = so_cfg.get("iou_missing", 0.3)
    iou_match = so_cfg.get("iou_match", 0.5)
    iou_loose = so_cfg.get("iou_loose", 0.6)

    issues = []

    for img_p in ds.image_files():
        stem = img_p.stem
        # Bỏ qua các ảnh không có trong preds hoặc không có file nhãn trên đĩa
        if stem not in preds:
            continue
        lab_p = ds.label_path(stem)
        if not lab_p.exists():
            continue

        raw_preds = preds[stem] or []
        img_size = ds.image_size(stem)
        W, H = img_size if img_size else (0, 0)

        valid_boxes = ds.valid_boxes(stem)

        # 1. P = dự đoán có conf >= conf VÀ diện tích px (w*W * h*H) >= min_area_px
        filtered_preds = []
        for p in raw_preds:
            r_cls = int(p[0])
            r_conf = float(p[1])
            r_box = (float(p[2]), float(p[3]), float(p[4]), float(p[5]))
            w, h = r_box[2], r_box[3]
            area_px = (w * W) * (h * H)
            if r_conf >= conf_thresh and area_px >= min_area_px:
                filtered_preds.append((r_cls, r_conf, r_box))

        for r_cls, r_conf, r_box in filtered_preds:
            cands_same = []
            cands_diff = []
            for l in valid_boxes:
                l_cls = l[0]
                l_box = l[1:5]
                iou = iou_xywh(r_box, l_box)
                if iou >= iou_missing:
                    if l_cls == r_cls:
                        cands_same.append((iou, l))
                    else:
                        cands_diff.append((iou, l))

            if not cands_same and not cands_diff:
                issues.append(
                    Issue(
                        check="second_opinion",
                        severity=WARN,
                        code="POSSIBLE_MISSING",
                        image=stem,
                        message=f"model phát hiện lớp {r_cls} (conf={r_conf:.2f}) nhưng nhãn thiếu",
                        line=None,
                        extra={"conf": r_conf, "class": r_cls},
                    )
                )
            elif cands_same:
                best_iou = max(iou for iou, _ in cands_same)
                if best_iou < iou_loose:
                    issues.append(
                        Issue(
                            check="second_opinion",
                            severity=WARN,
                            code="POSSIBLE_LOOSE_BOX",
                            image=stem,
                            message=f"nhãn lớp {r_cls} khớp lỏng với dự đoán (IoU={best_iou:.2f} < {iou_loose})",
                            line=None,
                            extra={
                                "conf": r_conf,
                                "iou": round(best_iou, 4),
                                "class": r_cls,
                            },
                        )
                    )
            else:
                best_diff_iou = max(iou for iou, _ in cands_diff)
                if best_diff_iou >= iou_match:
                    issues.append(
                        Issue(
                            check="second_opinion",
                            severity=WARN,
                            code="POSSIBLE_WRONG_CLASS",
                            image=stem,
                            message=f"nhãn có thể gán sai lớp (model: {r_cls}, IoU={best_diff_iou:.2f})",
                            line=None,
                            extra={
                                "conf": r_conf,
                                "iou": round(best_diff_iou, 4),
                                "class": r_cls,
                            },
                        )
                    )

        # 2. Check POSSIBLE_FALSE_POS for each valid line in original label file
        fp_preds = [
            (float(p[2]), float(p[3]), float(p[4]), float(p[5]))
            for p in raw_preds
            if float(p[1]) >= fp_conf
        ]

        try:
            content = lab_p.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue

        for line_no, raw_line in enumerate(content, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                cls_id, cx, cy, w, h = parse_yolo_line(line)
            except ValueError:
                continue
            if not (0 <= cls_id < len(ds.classes)):
                continue

            l_box = (cx, cy, w, h)
            matches_any = any(
                iou_xywh(l_box, p_box) >= iou_missing for p_box in fp_preds
            )
            if not matches_any:
                issues.append(
                    Issue(
                        check="second_opinion",
                        severity=WARN,
                        code="POSSIBLE_FALSE_POS",
                        image=stem,
                        message="dòng nhãn không khớp với dự đoán nào của model",
                        line=line_no,
                        extra={"conf": 0.0, "class": cls_id},
                    )
                )

    return issues
