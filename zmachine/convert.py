"""
Converts the Z-Machine spec HTML (Section 11 and Appendix B) to a
structured plain text suitable for LLM triple extraction. The goal
is not to produce readable prose but to make the relational content
of the header tables legible to the extraction prompt while making
sure to preserve parent-child structure for flags registers.
"""

import re
import urllib.request
from bs4 import BeautifulSoup
from pathlib import Path

SECT11_URL = "https://inform-fiction.org/zmachine/standards/z1point1/sect11.html"
APPB_URL = "https://inform-fiction.org/zmachine/standards/z1point1/appb.html"


def fetch(url):
  with urllib.request.urlopen(url) as r:
    return r.read().decode("utf-8")


def expand_access(dyn, intr, rst):
  """Convert asterisk columns to explicit English phrases."""
  parts = []

  if dyn.strip() == "*":
    parts.append("game may mutate")
  if intr.strip() == "*":
    parts.append("interpreter may mutate")
  if rst.strip() == "*":
    parts.append("interpreter must reset on restore/restart")

  return "; ".join(parts) if parts else "read-only"


def clean(text):
  """Collapse whitespace and strip."""
  return re.sub(r"\s+", " ", text).strip()


def parse_sect11(html):
  soup = BeautifulSoup(html, "html.parser")
  lines = []

  lines.append("=" * 70)
  lines.append("SOURCE: Z-Machine Standard 1.1, Section 11")
  lines.append("AUTHORITY: normative")
  lines.append("TOPIC: The format of the Z-Machine header")
  lines.append("=" * 70)
  lines.append("")

  # --- Preamble prose before the table ---
  lines.append("PREAMBLE:")
  lines.append(
    "The header occupies the first 64 bytes of a Z-Machine story file. "
    "Fields marked Dyn may be changed by the game during play. Fields "
    "marked Int may be changed by the interpreter. Fields marked Rst must "
    "be set correctly by the interpreter after loading, after restore, and "
    "after restart."
  )
  lines.append("")

  # --- Main header table ---
  lines.append("MAIN HEADER TABLE:")
  lines.append("-" * 70)

  tables = soup.find_all("table")

  if not tables:
    raise ValueError("No tables found in Section 11 HTML")

  current_hex = None
  current_reg = None  # label of current flags register, if any
  skip_header = True  # skip the first header row

  for table in tables[:1]:  # first table is the main header
    for row in table.find_all("tr"):
      cells = [clean(td.get_text()) for td in row.find_all(["td", "th"])]

      if len(cells) < 6:
        continue

      hex_val, v_val, dyn, intr, rst, contents = (
        cells[0],
        cells[1],
        cells[2],
        cells[3],
        cells[4],
        cells[5],
      )

      # Skip repeated header rows
      if hex_val.lower() in ("hex", "**hex**"):
        skip_header = False
        continue

      if skip_header:
        continue

      access = expand_access(dyn, intr, rst)

      # Detect flags register declaration lines (Contents starts with
      # "Flags" and has no bit detail, or italicised flag label)
      is_flags_header = bool(re.match(r"\*?Flags\s+\d", contents, re.I))

      # Blank hex = continuation row (bit within current register,
      # or version-specific variant of the previous field)
      if hex_val == "":
        if is_flags_header:
          # Version-variant flags header (e.g. "Flags 1 from V4")
          current_reg = contents
          lines.append(
            f"  FLAGS REGISTER VARIANT (offset {current_hex}): "
            f"{contents} [applies from version {v_val or '?'}]"
          )
        elif current_reg and re.match(r"Bit\s+\d", contents, re.I):
          # Bit row within a register
          lines.append(
            f"    BIT {contents} [version: {v_val or 'all'}] [access: {access}]"
          )
        elif current_reg:
          # Bit row without explicit "Bit N:" prefix
          lines.append(
            f"    BIT: {contents} [version: {v_val or 'all'}] [access: {access}]"
          )
        else:
          # Version-conditional variant of a plain field
          lines.append(
            f"  VARIANT (offset {current_hex}): {contents} "
            f"[version: {v_val or 'all'}] "
            f"[access: {access}]"
          )
      else:
        # New field row
        current_hex = hex_val

        if is_flags_header:
          current_reg = contents
          lines.append("")
          lines.append(
            f"FIELD offset=0x{hex_val} [version: {v_val or 'all'}] [access: {access}]"
          )
          lines.append("  TYPE: flags register")
          lines.append(f"  LABEL: {contents}")
          lines.append("  BITS:")
        else:
          current_reg = None
          lines.append("")
          lines.append(
            f"FIELD offset=0x{hex_val} [version: {v_val or 'all'}] [access: {access}]"
          )
          lines.append(f"  CONTENTS: {contents}")

  lines.append("")
  lines.append("-" * 70)

  # --- Header extension table (second table in Section 11) ---
  lines.append("")
  lines.append("HEADER EXTENSION TABLE:")
  lines.append(
    "Pointed to by the field at offset 0x36. Word 0 contains the count "
    "of subsequent words. Reads beyond the table length return 0. "
    "Writes beyond the table length are silently ignored."
  )
  lines.append("-" * 70)

  if len(tables) >= 2:
    skip_header = True

    for row in tables[1].find_all("tr"):
      cells = [clean(td.get_text()) for td in row.find_all(["td", "th"])]

      if len(cells) < 6:
        continue

      word, v_val, dyn, intr, rst, contents = (
        cells[0],
        cells[1],
        cells[2],
        cells[3],
        cells[4],
        cells[5],
      )

      if word.lower() == "word":
        skip_header = False
        continue

      if skip_header:
        continue

      access = expand_access(dyn, intr, rst)
      is_flags = bool(re.match(r"\*?Flags\s+\d", contents, re.I))

      if word == "":
        lines.append(
          f"    BIT: {contents} [version: {v_val or 'all'}] [access: {access}]"
        )
      else:
        lines.append("")
        lines.append(
          f"EXTENSION WORD index={word} [version: {v_val or 'all'}] [access: {access}]"
        )
        lines.append(f"  CONTENTS: {contents}")

        if is_flags:
          lines.append("  BITS:")

  lines.append("")
  lines.append("-" * 70)

  # --- Numbered subsections (prose rules) ---
  lines.append("")
  lines.append("NORMATIVE RULES (subsections):")
  lines.append("")

  for tag in soup.find_all(["h2", "h3", "p", "pre"]):
    text = clean(tag.get_text())

    if not text or text.lower() in ("remarks",):
      continue

    # Skip navigation and image alt text
    if re.match(r"^(Contents|Section|Appendix|\[|\d+\s*/)", text):
      continue

    if tag.name in ("h2", "h3"):
      lines.append(f"\n[{text}]")
    elif tag.name == "pre":
      lines.append(f"  CODE/TABLE: {text}")
    else:
      if len(text) > 20:
        lines.append(f"  {text}")

  return "\n".join(lines)


def parse_appb(html):
  soup = BeautifulSoup(html, "html.parser")
  lines = []

  lines.append("=" * 70)
  lines.append("SOURCE: Z-Machine Standard 1.1, Appendix B")
  lines.append("AUTHORITY: conventional")
  lines.append("TOPIC: Conventional contents of the Z-Machine header")
  lines.append("=" * 70)
  lines.append("")
  lines.append(
    "PREAMBLE: These fields are not normatively required by Section 11 "
    "but are present by convention in Infocom and Inform story files. "
    "Interpreters that wish to handle real-world story files should "
    "be aware of them."
  )
  lines.append("")

  tables = soup.find_all("table")

  if not tables:
    raise ValueError("No tables found in Appendix B HTML")

  lines.append("CONVENTIONAL HEADER FIELDS:")
  lines.append("-" * 70)

  skip_header = True
  current_hex = None
  current_reg = None

  for row in tables[0].find_all("tr"):
    cells = [clean(td.get_text()) for td in row.find_all(["td", "th"])]

    if len(cells) < 5:
      continue

    # Appendix B table has 5 columns: Hex, V, Dyn, Int, Contents
    # (no Rst column)
    hex_val, v_val, dyn, intr, contents = (
      cells[0],
      cells[1],
      cells[2],
      cells[3],
      cells[4],
    )

    if hex_val.lower() == "hex":
      skip_header = False
      continue

    if skip_header:
      continue

    access_parts = []
    if dyn.strip() == "*":
      access_parts.append("game may mutate")

    if intr.strip() == "*":
      access_parts.append("interpreter may mutate")

    access = "; ".join(access_parts) if access_parts else "read-only"

    is_flags = bool(re.match(r"\*?Flags\s+\d", contents, re.I))

    if hex_val == "":
      if is_flags:
        current_reg = contents
        lines.append(
          f"  FLAGS REGISTER VARIANT (offset {current_hex}): "
          f"{contents} [version: {v_val or 'all'}]"
        )
      elif current_reg:
        lines.append(
          f"    BIT: {contents} [version: {v_val or 'all'}] [access: {access}]"
        )
      else:
        lines.append(
          f"  VARIANT (offset {current_hex}): {contents} [version: {v_val or 'all'}]"
        )
    else:
      current_hex = hex_val

      if is_flags:
        current_reg = contents
        lines.append("")
        lines.append(
          f"FIELD offset=0x{hex_val} "
          f"[version: {v_val or 'all'}] "
          f"[access: {access}] "
          f"[authority: conventional]"
        )
        lines.append("  TYPE: flags register")
        lines.append(f"  LABEL: {contents}")
        lines.append("  BITS:")
      else:
        current_reg = None
        lines.append("")
        lines.append(
          f"FIELD offset=0x{hex_val} "
          f"[version: {v_val or 'all'}] "
          f"[access: {access}] "
          f"[authority: conventional]"
        )
        lines.append(f"  CONTENTS: {contents}")

  lines.append("")
  lines.append("-" * 70)

  # --- Footnotes as prose ---
  lines.append("")
  lines.append("NOTES (conventional usage details):")
  lines.append("")

  for tag in soup.find_all(["p", "li", "blockquote"]):
    text = clean(tag.get_text())

    if len(text) > 30 and not re.match(r"^(Contents|Section|Appendix)", text):
      lines.append(f"  NOTE: {text}")

  return "\n".join(lines)


if __name__ == "__main__":
  script_dir = Path(__file__).parent

  print("Fetching Section 11...")

  sect11_html = fetch(SECT11_URL)
  sect11_text = parse_sect11(sect11_html)
  out_sect11 = script_dir / "extraction_input_sect11.txt"

  with open(out_sect11, "w") as f:
    f.write(sect11_text)

  print(f"  Written: extraction_input_sect11.txt ({len(sect11_text)} chars)")

  print("Fetching Appendix B...")

  appb_html = fetch(APPB_URL)
  appb_text = parse_appb(appb_html)
  out_appb = script_dir / "extraction_input_appb.txt"

  with open(out_appb, "w") as f:
    f.write(appb_text)

  print(f"  Written: extraction_input_appb.txt ({len(appb_text)} chars)")

  print()
  print("Preview: Section 11 (first 3000 chars)")
  print("-" * 70)
  print(sect11_text[:3000])
  print()
  print("Preview: Appendix B (first 2000 chars)")
  print("-" * 70)
  print(appb_text[:2000])
