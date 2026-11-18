"""CHECK 2 — Dòng nhãn YOLO: sai định dạng, ID lớp lạ, box tràn ảnh, box rỗng, box lặp.   (Script 2 trong slide)   ★★

Test:  pytest tests/test_label_format.py -q

Mỗi dòng nhãn: `id cx cy w h` (5 số, toạ độ 0–1). Trả về list[Issue]; `Issue.line` = số dòng (đếm từ 1).

| code             | severity | khi nào                                                                  |
|------------------|----------|--------------------------------------------------------------------------|
| BAD_FORMAT       | ERROR    | không đúng 5 cột số (dùng `parse_yolo_line`, bắt `ValueError`)           |
| BAD_CLASS_ID     | ERROR    | id lớp ngoài 0..len(ds.classes)-1                                        |
| BOX_ZERO         | ERROR    | w <= 0 hoặc h <= 0                                                       |
| BOX_OUT_OF_IMAGE | ERROR    | mép box vượt [0,1] quá `cfg["label_format"]["overflow_tol"]`; extra={"overflow": mức tràn} |
| DUP_BOX          | ERROR    | cùng lớp với một dòng TRƯỚC ĐÓ và IoU > `dup_box_iou`; báo ở dòng SAU    |
| BOX_TINY         | WARN     | cạnh ngắn của box (px) < `min_box_px`  (cần `ds.image_size(stem)`)       |
| BOX_HUGE         | WARN     | w*h > `max_box_frac`                                                     |
"""
from collections import defaultdict

from ..common import ERROR, WARN, Issue, iou_xywh, parse_yolo_line


def run(ds, cfg):
    issues = []
    lf_cfg = cfg.get("label_format", {})
    overflow_tol = lf_cfg.get("overflow_tol", 0.001)
    min_box_px = lf_cfg.get("min_box_px", 12)
    max_box_frac = lf_cfg.get("max_box_frac", 0.95)
    dup_box_iou = lf_cfg.get("dup_box_iou", 0.9)
    n_classes = len(ds.classes)

    for lab_file in ds.label_files():
        stem = lab_file.stem
        img_size = ds.image_size(stem)

        try:
            content = lab_file.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        lines = content.splitlines()
        prev_boxes = defaultdict(list)

        for line_no, raw_line in enumerate(lines, 1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                cls_id, cx, cy, w, h = parse_yolo_line(line)
            except ValueError:
                issues.append(
                    Issue(
                        check="label_format",
                        severity=ERROR,
                        code="BAD_FORMAT",
                        image=stem,
                        message="sai định dạng dòng YOLO",
                        line=line_no,
                    )
                )
                continue

            if not (0 <= cls_id < n_classes):
                issues.append(
                    Issue(
                        check="label_format",
                        severity=ERROR,
                        code="BAD_CLASS_ID",
                        image=stem,
                        message=f"id lớp {cls_id} không hợp lệ (ngoài 0..{n_classes - 1})",
                        line=line_no,
                    )
                )
                continue

            if w <= 0 or h <= 0:
                issues.append(
                    Issue(
                        check="label_format",
                        severity=ERROR,
                        code="BOX_ZERO",
                        image=stem,
                        message=f"kích thước box <= 0 (w={w}, h={h})",
                        line=line_no,
                    )
                )
                continue

            # Geometry checks
            x_min = cx - w / 2.0
            x_max = cx + w / 2.0
            y_min = cy - h / 2.0
            y_max = cy + h / 2.0
            overflow = max(0.0, -x_min, x_max - 1.0, -y_min, y_max - 1.0)

            if overflow > overflow_tol:
                issues.append(
                    Issue(
                        check="label_format",
                        severity=ERROR,
                        code="BOX_OUT_OF_IMAGE",
                        image=stem,
                        message=f"box tràn ảnh (overflow={overflow:.4f})",
                        line=line_no,
                        extra={"overflow": float(overflow)},
                    )
                )

            curr_box = (cx, cy, w, h)
            # Check duplicate box with earlier boxes of the same class in this file
            is_dup = False
            for prev_box in prev_boxes[cls_id]:
                if iou_xywh(curr_box, prev_box) > dup_box_iou:
                    issues.append(
                        Issue(
                            check="label_format",
                            severity=ERROR,
                            code="DUP_BOX",
                            image=stem,
                            message="box trùng lặp với dòng trước",
                            line=line_no,
                        )
                    )
                    is_dup = True
                    break

            prev_boxes[cls_id].append(curr_box)

            # Warnings: BOX_TINY and BOX_HUGE
            if img_size is not None:
                W, H = img_size
                if min(w * W, h * H) < min_box_px:
                    issues.append(
                        Issue(
                            check="label_format",
                            severity=WARN,
                            code="BOX_TINY",
                            image=stem,
                            message=f"cạnh ngắn < {min_box_px}px",
                            line=line_no,
                        )
                    )

            if w * h > max_box_frac:
                issues.append(
                    Issue(
                        check="label_format",
                        severity=WARN,
                        code="BOX_HUGE",
                        image=stem,
                        message=f"diện tích box > {max_box_frac:.0%}",
                        line=line_no,
                    )
                )

    return issues
