#!/usr/bin/env python3
"""扫描 docs/reading/ 下所有 .md 的 YAML frontmatter,重新生成 README.md 索引。

用法:
    python docs/reading/build_index.py

设计要点:
- **只用标准库**,不引入 pyyaml —— clipping 的 frontmatter 结构固定,
  手写十几行解析足够,不值得为它加一个依赖。
- **幂等**:连跑两次结果完全相同。
- README.md 顶部的说明段落是脚本内的常量,每次重写,不需要手工维护。
"""
from __future__ import annotations

import pathlib
import re

HERE = pathlib.Path(__file__).resolve().parent
README = HERE / "README.md"

HEADER = """# 参考读物

本目录收录**外部**量化 / 交易方法论文章,**不是本项目的研究产出**。

- 版权归各原作者所有;每篇的 YAML frontmatter 中保留了 `title` / `source` / `author`,
  **请勿删除**,那是署名与出处的唯一载体。
- 明确标注禁止转载的材料**不要放这里**。
- 本文件由 `build_index.py` 自动生成,**不要手工编辑** ——
  新增文章后跑一次 `python docs/reading/build_index.py` 即可。

"""


def parse_front_matter(text: str) -> dict[str, object]:
    """解析 clipping 的 frontmatter。只认 `key: value` 与紧随其后的 `- item` 列表。"""
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end == -1:
        return {}
    meta: dict[str, object] = {}
    key = None
    for raw in text[3:end].splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        if line.lstrip().startswith("- ") and key:
            meta.setdefault(key, [])
            if isinstance(meta[key], list):
                meta[key].append(_clean(line.lstrip()[2:]))
            continue
        m = re.match(r"^([A-Za-z_][\w-]*):\s*(.*)$", line)
        if not m:
            continue
        key, val = m.group(1), _clean(m.group(2))
        meta[key] = val if val else []
    return meta


def _clean(s: str) -> str:
    """去掉包裹引号与 Obsidian 的 [[wiki link]] 括号。"""
    s = s.strip().strip('"').strip("'").strip()
    return re.sub(r"^\[\[(.*)\]\]$", r"\1", s)


def as_text(v: object) -> str:
    if isinstance(v, list):
        return " / ".join(str(x) for x in v if str(x).strip())
    return str(v or "").strip()


def main() -> int:
    rows = []
    for p in sorted(HERE.glob("*.md")):
        if p.name == "README.md":
            continue
        meta = parse_front_matter(p.read_text(encoding="utf-8"))
        rows.append({
            "file": p.name,
            "title": as_text(meta.get("title")) or p.stem,
            "author": as_text(meta.get("author")) or "—",
            "source": as_text(meta.get("source")),
            "created": as_text(meta.get("created")) or "—",
        })
    rows.sort(key=lambda r: (r["created"], r["title"]), reverse=True)

    lines = [HEADER.rstrip(), "", f"共 **{len(rows)}** 篇。", "",
             "| 标题 | 作者 | 原文 | 收录 |", "|---|---|---|---|"]
    for r in rows:
        link = f"[{r['title']}]({r['file']})"
        src = f"[原文]({r['source']})" if r["source"] else "—"
        lines.append(f"| {link} | {r['author']} | {src} | {r['created']} |")
    lines.append("")

    README.write_text("\n".join(lines), encoding="utf-8")
    print(f"→ {README}  ({len(rows)} 篇)")
    for r in rows:
        print(f"   {r['created']}  {r['title'][:40]}  —  {r['author']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
