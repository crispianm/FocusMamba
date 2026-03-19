import re
import sys
from pathlib import Path

MD_PATH = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else Path("bug_hunt_vda/supervisor_update.md")
)
OUT_PATH = Path(sys.argv[2]) if len(sys.argv) > 2 else MD_PATH.with_suffix(".pdf")

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Image,
        Preformatted,
    )
    from reportlab.lib import utils
except ImportError:
    import subprocess

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "reportlab"]
    )  # try to install
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Image,
        Preformatted,
    )
    from reportlab.lib import utils


def fit_image(path, max_width=6 * inch):
    img = utils.ImageReader(str(path))
    iw, ih = img.getSize()
    ratio = min(1, max_width / iw)
    return Image(str(path), width=iw * ratio, height=ih * ratio)


def parse_markdown(md_text):
    # Split into lines and simple blocks: headings, lists, paragraphs, images, pre blocks
    lines = md_text.splitlines()
    blocks = []
    buffer = []

    def flush_buf():
        nonlocal buffer
        if not buffer:
            return
        text = "\n".join(buffer).strip()
        if text:
            blocks.append(("para", text))
        buffer = []

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("# "):
            flush_buf()
            blocks.append(("h1", line[2:].strip()))
        elif line.startswith("## "):
            flush_buf()
            blocks.append(("h2", line[3:].strip()))
        elif re.match(r"^!\[.*\]\(.*\)", line.strip()):
            flush_buf()
            m = re.match(r"!\[(.*?)\]\((.*?)\)", line.strip())
            if m:
                alt, path = m.groups()
                blocks.append(("img", path))
        elif re.match(r"^\|.*\|", line):
            # collect table lines
            flush_buf()
            table_lines = [line]
            i += 1
            while i < len(lines) and re.match(r"^\|.*\|", lines[i]):
                table_lines.append(lines[i])
                i += 1
            blocks.append(("table", "\n".join(table_lines)))
            continue
        else:
            buffer.append(line)
        i += 1
    flush_buf()
    return blocks


def md_to_flowables(md_path):
    text = md_path.read_text(encoding="utf-8")
    blocks = parse_markdown(text)
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], spaceAfter=12)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], spaceAfter=8)
    normal = styles["BodyText"]
    pre = ParagraphStyle("pre", parent=styles["Code"], fontName="Courier", fontSize=8)

    flow = []
    for kind, content in blocks:
        if kind == "h1":
            flow.append(Paragraph(content, h1))
        elif kind == "h2":
            flow.append(Paragraph(content, h2))
        elif kind == "img":
            p = Path(content)
            # try relative to md file
            if not p.exists():
                p = md_path.parent / content
            if not p.exists():
                p = Path(content)
            if p.exists():
                try:
                    flow.append(fit_image(p))
                except Exception:
                    flow.append(
                        Paragraph(
                            f"[Image: {content} missing or failed to load]", normal
                        )
                    )
            else:
                flow.append(Paragraph(f"[Image not found: {content}]", normal))
        elif kind == "table":
            # simply include table as preformatted text
            flow.append(Preformatted(content, pre))
        elif kind == "para":
            # convert basic markdown emphasis markers to simple HTML for Paragraph
            html = (
                content.replace("**", "<b>")
                .replace("`", '<font face="Courier">')
                .replace("</b><b>", "")
            )
            # fix bold pairs: naive approach
            html = html.replace("<b>", "<b>").replace("<b>", "<b>")
            # Replace markdown list items with bullets
            if "\n- " in content or content.strip().startswith("- "):
                # convert to lines
                lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
                for ln in lines:
                    if ln.startswith("- "):
                        flow.append(Paragraph("• " + ln[2:], normal))
                    else:
                        flow.append(Paragraph(ln, normal))
            else:
                flow.append(Paragraph(content.replace("\n", "<br/>"), normal))
        flow.append(Spacer(1, 6))
    return flow


def main():
    flowables = md_to_flowables(MD_PATH)
    doc = SimpleDocTemplate(
        str(OUT_PATH),
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )
    doc.build(flowables)


if __name__ == "__main__":
    main()
