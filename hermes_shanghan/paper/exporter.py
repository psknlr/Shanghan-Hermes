"""論文導出（十九輪）：manuscript.md → .docx 與 .zip（純標準庫）。

- ``manuscript.docx``：最小合法 OOXML（標題層級/正文段落/列表；Markdown
  表格以等寬段落保形）。SVG 圖表不嵌入 docx（OOXML 需點陣/EMF），
  隨 zip 完整分發並在文中保留圖號引用。
- ``paper_bundle.zip``：修訂目錄全件打包（稿件 md/docx + SVG 圖 + CSV
  表 + 元數據 + Source Data），投稿即用。
"""
from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Dict, List
from xml.sax.saxutils import escape

_CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>"""

_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""

_STYLES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:style w:type="paragraph" w:styleId="Normal" w:default="1">
 <w:name w:val="Normal"/><w:rPr><w:sz w:val="21"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/>
 <w:basedOn w:val="Normal"/><w:rPr><w:b/><w:sz w:val="32"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/>
 <w:basedOn w:val="Normal"/><w:rPr><w:b/><w:sz w:val="27"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading3"><w:name w:val="heading 3"/>
 <w:basedOn w:val="Normal"/><w:rPr><w:b/><w:sz w:val="24"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Mono"><w:name w:val="Mono"/>
 <w:basedOn w:val="Normal"/>
 <w:rPr><w:rFonts w:ascii="Courier New" w:eastAsia="SimSun"/><w:sz w:val="18"/></w:rPr></w:style>
</w:styles>"""


def _para(text: str, style: str = "") -> str:
    st = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    runs = f'<w:r><w:t xml:space="preserve">{escape(text)}</w:t></w:r>' \
        if text else "<w:r/>"
    return f"<w:p>{st}{runs}</w:p>"


def markdown_to_docx_xml(md: str) -> str:
    """Markdown（本項目稿件方言）→ document.xml 主體。"""
    body: List[str] = []
    in_table = False
    for raw in md.splitlines():
        line = raw.rstrip()
        if not line.strip():
            if in_table:
                in_table = False
            body.append(_para(""))
            continue
        if line.startswith("|"):
            if re.fullmatch(r"\|[\s:\-|]+\|?", line):
                continue                     # 分隔行
            cells = [c.strip() for c in line.strip("|").split("|")]
            body.append(_para("　".join(cells), "Mono"))
            in_table = True
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            level = min(3, len(m.group(1)))
            body.append(_para(m.group(2), f"Heading{level}"))
            continue
        if line.startswith(("- ", "* ")):
            body.append(_para("• " + line[2:]))
            continue
        # 行內 markdown 標記剝離（**…**、`…`）
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", line)
        text = re.sub(r"`([^`]*)`", r"\1", text)
        body.append(_para(text))
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/'
            'wordprocessingml/2006/main"><w:body>'
            + "".join(body) +
            '<w:sectPr/></w:body></w:document>')


def write_docx(md_path: Path, out_path: Path) -> Path:
    md = md_path.read_text(encoding="utf-8")
    doc_xml = markdown_to_docx_xml(md)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", _CONTENT_TYPES)
        z.writestr("_rels/.rels", _RELS)
        z.writestr("word/styles.xml", _STYLES)
        z.writestr("word/document.xml", doc_xml)
    return out_path


def write_bundle_zip(rev_dir: Path, out_path: Path) -> Path:
    """修訂目錄全件打包（跳過自身與既有 zip）。"""
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(rev_dir.rglob("*")):
            if p.is_file() and p.suffix != ".zip":
                z.write(p, p.relative_to(rev_dir))
    return out_path


def export_bundle(manuscript_path: Path) -> Dict[str, str]:
    """為一篇已生成的稿件補齊 docx 與 zip，返回修訂目錄內文件名。"""
    rev = manuscript_path.parent
    docx = write_docx(manuscript_path, rev / "manuscript.docx")
    bundle = write_bundle_zip(rev, rev / "paper_bundle.zip")
    return {"md": manuscript_path.name, "docx": docx.name,
            "zip": bundle.name}
