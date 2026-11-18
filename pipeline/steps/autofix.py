"""BƯỚC 2/4 — Sửa tự động những lỗi bắt buộc mà máy sửa được AN TOÀN.   ★★

Test:  pytest tests/test_autofix.py -q

`fix(ds, issues, cfg) -> list[dict]`: duyệt `issues` (chỉ lỗi severity ERROR), sửa thẳng vào thư mục `ds`,
và trả về danh sách thay đổi; mỗi phần tử có khoá: image, line, action, before, after
(đây chính là file `03_fixes.csv` — bằng chứng pipeline đã làm gì, người có thể kiểm lại).

| code lỗi                                       | cách sửa                                    | action           |
|------------------------------------------------|---------------------------------------------|------------------|
| BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO, DUP_BOX    | xoá dòng đó khỏi file nhãn                  | "drop_line"      |
| BOX_OUT_OF_IMAGE                               | cắt box về trong [0,1] (giữ nguyên lớp)     | "clip_box"       |
| MISSING_LABEL_FILE                             | KHÔNG sửa file (máy không thể bịa nhãn): ghi 1 dòng fix, before="(không có file)", after="chuyển người gán" | "needs_label" |
| BAD_CLASS_NAME                                 | ghi lại `names` trong data.yaml đúng `cfg["classes"]` | "rename_classes" |
| DUP_IMAGE_*, LEAK_TEST, CORRUPT_IMAGE, BAD_SIZE| xoá ảnh + nhãn khỏi `ds`                    | "remove_image"   |
"""
from collections import defaultdict

import yaml

from ..common import ERROR, parse_yolo_line


def fix(ds, issues, cfg):
    error_issues = [i for i in issues if i.severity == ERROR]
    if not error_issues:
        return []

    fixes = []
    removed_images = set()

    # 1. Image removal issues (DUP_IMAGE_*, LEAK_TEST, CORRUPT_IMAGE, BAD_SIZE)
    for i in error_issues:
        if i.code.startswith("DUP_IMAGE_") or i.code in (
            "LEAK_TEST",
            "CORRUPT_IMAGE",
            "BAD_SIZE",
        ):
            stem = i.image
            if stem not in removed_images:
                removed_images.add(stem)
                img_p = ds.image_path(stem)
                if img_p and img_p.exists():
                    img_p.unlink()
                lab_p = ds.label_path(stem)
                if lab_p and lab_p.exists():
                    lab_p.unlink()
                fixes.append(
                    {
                        "image": stem,
                        "line": "",
                        "action": "remove_image",
                        "before": i.code,
                        "after": "xoá ảnh và nhãn khỏi ds",
                    }
                )

    # 2. Dataset-level issues (BAD_CLASS_NAME)
    renamed_classes = False
    for i in error_issues:
        if i.code == "BAD_CLASS_NAME" and not renamed_classes:
            data = {}
            if ds.yaml_path.exists():
                try:
                    data = yaml.safe_load(ds.yaml_path.read_text(encoding="utf-8")) or {}
                except Exception:
                    data = {}
            old_names = data.get("names", "")
            data["names"] = list(cfg["classes"])
            ds.yaml_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
            fixes.append(
                {
                    "image": "data.yaml",
                    "line": "",
                    "action": "rename_classes",
                    "before": str(old_names),
                    "after": str(cfg["classes"]),
                }
            )
            renamed_classes = True

    # 3. MISSING_LABEL_FILE (needs_label)
    seen_missing = set()
    for i in error_issues:
        if i.code == "MISSING_LABEL_FILE":
            stem = i.image
            if stem not in removed_images and stem not in seen_missing:
                seen_missing.add(stem)
                fixes.append(
                    {
                        "image": stem,
                        "line": "",
                        "action": "needs_label",
                        "before": "(không có file)",
                        "after": "chuyển người gán",
                    }
                )

    # 4. Label line issues (BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO, DUP_BOX, BOX_OUT_OF_IMAGE)
    line_issues = defaultdict(lambda: defaultdict(list))
    for i in error_issues:
        if i.code in (
            "BAD_FORMAT",
            "BAD_CLASS_ID",
            "BOX_ZERO",
            "DUP_BOX",
            "BOX_OUT_OF_IMAGE",
        ):
            stem = i.image
            if stem not in removed_images and i.line is not None:
                line_issues[stem][i.line].append(i)

    drop_codes = {"BAD_FORMAT", "BAD_CLASS_ID", "BOX_ZERO", "DUP_BOX"}

    for stem in sorted(line_issues.keys()):
        lab_p = ds.label_path(stem)
        if not lab_p.exists():
            continue

        raw_lines = lab_p.read_text(encoding="utf-8", errors="replace").splitlines()
        dropped_lines = set()
        modified_lines = {}

        for line_no in sorted(line_issues[stem].keys()):
            if line_no < 1 or line_no > len(raw_lines):
                continue

            cur_issues = line_issues[stem][line_no]
            raw_line = raw_lines[line_no - 1]

            has_drop = any(iss.code in drop_codes for iss in cur_issues)
            has_overflow = any(iss.code == "BOX_OUT_OF_IMAGE" for iss in cur_issues)

            if has_drop:
                dropped_lines.add(line_no)
                fixes.append(
                    {
                        "image": stem,
                        "line": line_no,
                        "action": "drop_line",
                        "before": raw_line,
                        "after": "",
                    }
                )
            elif has_overflow:
                try:
                    cls_id, cx, cy, w, h = parse_yolo_line(raw_line)
                    x1 = cx - w / 2.0
                    y1 = cy - h / 2.0
                    x2 = cx + w / 2.0
                    y2 = cy + h / 2.0

                    cx1 = max(0.0, min(1.0, x1))
                    cy1 = max(0.0, min(1.0, y1))
                    cx2 = max(0.0, min(1.0, x2))
                    cy2 = max(0.0, min(1.0, y2))

                    new_w = cx2 - cx1
                    new_h = cy2 - cy1

                    if new_w <= 0.0 or new_h <= 0.0:
                        dropped_lines.add(line_no)
                        fixes.append(
                            {
                                "image": stem,
                                "line": line_no,
                                "action": "drop_line",
                                "before": raw_line,
                                "after": "",
                            }
                        )
                    else:
                        new_cx = (cx1 + cx2) / 2.0
                        new_cy = (cy1 + cy2) / 2.0
                        new_line = (
                            f"{cls_id} {new_cx:.6f} {new_cy:.6f} {new_w:.6f} {new_h:.6f}"
                        )
                        modified_lines[line_no] = new_line
                        fixes.append(
                            {
                                "image": stem,
                                "line": line_no,
                                "action": "clip_box",
                                "before": raw_line,
                                "after": new_line,
                            }
                        )
                except ValueError:
                    dropped_lines.add(line_no)
                    fixes.append(
                        {
                            "image": stem,
                            "line": line_no,
                            "action": "drop_line",
                            "before": raw_line,
                            "after": "",
                        }
                    )

        new_lines = []
        for idx, ln in enumerate(raw_lines, 1):
            if idx in dropped_lines:
                continue
            if idx in modified_lines:
                new_lines.append(modified_lines[idx])
            else:
                new_lines.append(ln)

        content = "\n".join(new_lines) + "\n" if new_lines else ""
        lab_p.write_text(content, encoding="utf-8")

    return fixes
