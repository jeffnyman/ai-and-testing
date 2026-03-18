"""
zmachine_header_reference.py

Reference implementation of the Z-Machine header parser.

Written directly from:
  - Section 11 of the Z-Machine Standard 1.1 (normative)
  - Appendix B of the Z-Machine Standard 1.1 (conventional)
  - zmachine_header.ttl (ontology, used as consistency check)

Design goals:
  - Clarity over cleverness. Every decision traces to a spec citation.
  - Explicit version-gating: fields absent for a given version are
    absent from the output dict, not present as None or 0.
  - Conventional fields (Appendix B) grouped under a 'conventional' key
    to match the ontology's hasSourceAuthority zm:conventional distinction.
  - No third-party dependencies. Standard library only.

Output structure:
  {
    "version": int,
    "scalar_fields": { ... },   # normative scalar fields
    "flags1": { ... },          # version-conditional bit layout
    "flags2": { ... },          # per-bit version applicability
    "extension_table": { ... }, # V5+ only, None if absent/zero pointer
    "conventional": { ... },    # Appendix B fields
  }

Usage:
  python zmachine_header_reference.py path/to/story.z5
"""

import json
import struct
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Low-level readers
# These are the only functions that touch raw bytes. Everything else
# builds on them, making the byte layout easy to audit.
# ---------------------------------------------------------------------------


def _byte(data: bytes, offset: int) -> int:
  """Read one unsigned byte."""
  return data[offset]


def _word(data: bytes, offset: int) -> int:
  """Read one big-endian unsigned word (2 bytes). Spec §11: all words
  in the header are big-endian."""
  return struct.unpack_from(">H", data, offset)[0]


def _ascii(data: bytes, offset: int, length: int) -> str:
  """Read 'length' bytes as ASCII, replacing non-printable bytes with '.'."""
  raw = data[offset : offset + length]
  return "".join(chr(b) if 32 <= b < 127 else "." for b in raw)


def _ext_word(data: bytes, ext_base: int, ext_length: int, index: int) -> int:
  """
  Read word at 'index' from the header extension table.
  Spec §11.1.7.1: reads beyond the table length return 0.
  Word 0 is the length word itself; data words start at index 1.
  ext_length is the value stored in word 0 (count of subsequent words).
  """
  if index == 0:
    return ext_length
  if index > ext_length:
    return 0
  return _word(data, ext_base + index * 2)


# ---------------------------------------------------------------------------
# Flags 1 decoder
# Spec §11, offset 0x01. The bit layout is completely different between
# V1-3 and V4+. The ontology models this as two sets of BitField individuals
# sharing a single FlagsRegister parent (zm:field_flags1).
# ---------------------------------------------------------------------------


def _decode_flags1(raw: int, version: int) -> dict:
  """
  Decode Flags 1 (offset 0x01) according to version.

  V1-3 layout (spec §11, Flags 1 in Versions 1 to 3):
    bit 1: status line type (0=score/turns, 1=hours:mins)
    bit 2: story file split across two discs
    bit 3: [conventional] Tandy bit (Appendix B)
    bit 4: status line not available
    bit 5: screen-splitting available
    bit 6: variable-pitch font is default

  V4+ layout (spec §11, Flags 1 from Version 4):
    bit 0: colours available (V5+)
    bit 1: picture displaying available (V6)
    bit 2: boldface available (V4+)
    bit 3: italic available (V4+)
    bit 4: fixed-space style available (V4+)
    bit 5: sound effects available (V6)
    bit 7: timed keyboard input available (V4+)
  """

  def bit(n: int) -> bool:
    return bool(raw & (1 << n))

  if version <= 3:
    result = {
      "status_line_type_score_turns": not bit(1),  # 0=score/turns
      "story_split_across_two_discs": bit(2),
      "status_line_not_available": bit(4),
      "screen_splitting_available": bit(5),
      "variable_pitch_font_default": bit(6),
    }
    # Tandy bit: conventional (Appendix B), V3 only
    # Ontology: zm:flags1_v13_bit3_tandy, hasSourceAuthority zm:conventional
    if version == 3:
      result["tandy_bit"] = bit(3)
    return result
  else:
    # V4+ layout. Per-bit version applicability from ontology:
    # bit 0 (colours): applicableFrom zm:v5
    # bit 1 (pictures): applicableFrom zm:v6
    # bit 5 (sound): applicableFrom zm:v6
    result = {
      "boldface_available": bit(2),  # V4+
      "italic_available": bit(3),  # V4+
      "fixed_space_style_available": bit(4),  # V4+
      "timed_keyboard_input_available": bit(7),  # V4+
    }
    if version >= 5:
      result["colours_available"] = bit(0)
    if version >= 6:
      result["picture_displaying_available"] = bit(1)
      result["sound_effects_available"] = bit(5)
    return result


# ---------------------------------------------------------------------------
# Flags 2 decoder
# Spec §11, offset 0x10. Two-byte register; per-bit version applicability.
# Ontology: zm:field_flags2, containsBit per individual BitField.
# ---------------------------------------------------------------------------


def _decode_flags2(raw: int, version: int) -> dict:
  """
  Decode Flags 2 (offset 0x10, 2 bytes) with per-bit version gating.

  bit 0: transcripting on (V1+)
  bit 1: force fixed-pitch font (V3+)
  bit 2: screen redraw request (V6)
  bit 3: game wants pictures (V5+)
  bit 4: game wants UNDO opcodes (V5+)
  bit 5: game wants mouse (V5+)
  bit 6: game wants colours (V5+)
  bit 7: game wants sound effects (V5+)
  bit 8: game wants menus (V6)
  bit 10: [conventional] printer error (Appendix B)
  """

  def bit(n: int) -> bool:
    return bool(raw & (1 << n))

  result: dict = {}

  # V1+
  result["transcripting_on"] = bit(0)

  # V3+
  if version >= 3:
    result["force_fixed_pitch"] = bit(1)

  # V5+
  if version >= 5:
    result["game_wants_pictures"] = bit(3)
    result["game_wants_undo"] = bit(4)
    result["game_wants_mouse"] = bit(5)
    result["game_wants_colours"] = bit(6)
    result["game_wants_sound_effects"] = bit(7)

  # V6
  if version >= 6:
    result["screen_redraw_request"] = bit(2)
    result["game_wants_menus"] = bit(8)

  return result


# ---------------------------------------------------------------------------
# Extension table decoder
# Spec §11.1.7. Pointed to by offset 0x36 (V5+). Word 0 = count of
# subsequent words. Reads beyond table length return 0.
# Ontology: zm:headerExtensionTable, zm:ExtensionField individuals.
# ---------------------------------------------------------------------------


def _decode_extension_table(data: bytes, version: int, pointer: int) -> dict | None:
  """
  Decode the header extension table.
  Returns None if pointer is 0 or version < 5 (should not be called
  in that case, but defensive).
  Returns a dict of extension fields otherwise.
  """
  if version < 5 or pointer == 0:
    return None

  # Word 0: count of subsequent words (spec §11.1.7)
  ext_length = _word(data, pointer)

  def ew(index: int) -> int:
    return _ext_word(data, pointer, ext_length, index)

  result: dict = {
    "table_length": ext_length,
    "mouse_x": ew(1),
    "mouse_y": ew(2),
    "unicode_translation_table_addr": ew(3),
    "true_default_foreground_colour": ew(5),
    "true_default_background_colour": ew(6),
  }

  # Flags 3 (extension word 4): only bit 0 is defined, and only from V6.
  # Spec §11.1.7.4: bits set by game to request features; interpreter
  # clears if it cannot provide. §11.1.7.4.1: all unused bits must be
  # cleared by interpreter.
  flags3_raw = ew(4)
  flags3: dict = {}
  if version >= 6:
    flags3["game_wants_transparency"] = bool(flags3_raw & 0x01)
  result["flags3"] = flags3

  return result


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------


def parse_header(data: bytes) -> dict:
  """
  Parse the Z-Machine header from raw story file bytes.

  Raises ValueError if the data is shorter than 64 bytes (the minimum
  header size defined in spec §11).

  Returns a dict with keys:
    version, scalar_fields, flags1, flags2,
    extension_table (None if absent), conventional
  """
  # Spec §11: header occupies the first 64 bytes.
  if len(data) < 64:
    raise ValueError(
      f"Story file too short: {len(data)} bytes. "
      "A valid Z-Machine file must be at least 64 bytes."
    )

  # Version MUST be read first. Almost every other field is version-
  # conditional. Ontology: zm:field_version, applicableFrom zm:v1.
  version = _byte(data, 0x00)
  if version < 1 or version > 8:
    raise ValueError(
      f"Unexpected version byte: {version}. "
      "Z-Machine versions 1-8 are known; this file may be corrupt."
    )

  # -------------------------------------------------------------------
  # Scalar fields — normative (Section 11)
  # Each field is only included in the output if it is applicable for
  # the file's version. This matches applicableFrom/applicableUntil in
  # the ontology and avoids the null-field problem in the generated parser.
  # -------------------------------------------------------------------
  scalar: dict = {}

  # 0x04: High memory base. V1+. Ontology: zm:field_high_memory_base.
  scalar["high_memory_base"] = _word(data, 0x04)

  # 0x06: Initial PC (V1-5) or packed address of main routine (V6).
  # Spec §11, VARIANT: meaning changes at V6 boundary.
  # Ontology: zm:field_initial_pc_v1to5 supersededBy zm:field_initial_pc_v6.
  # Note: we store the raw word. For V6, the caller must multiply by 4
  # to get the byte address (packed address constant for routines in V6
  # is 8, but initial PC uses a different convention — raw value here).
  if version <= 5:
    scalar["initial_pc"] = _word(data, 0x06)
  else:
    scalar["initial_pc_packed"] = _word(data, 0x06)

  # 0x08: Dictionary location. V1+.
  scalar["dictionary_address"] = _word(data, 0x08)

  # 0x0A: Object table location. V1+.
  scalar["object_table_address"] = _word(data, 0x0A)

  # 0x0C: Global variables table. V1+.
  scalar["global_variables_address"] = _word(data, 0x0C)

  # 0x0E: Static memory base. V1+.
  scalar["static_memory_base"] = _word(data, 0x0E)

  # 0x18: Abbreviations table. V2+.
  if version >= 2:
    scalar["abbreviations_address"] = _word(data, 0x18)

  # 0x1A: File length (divided by version-dependent constant). V3+.
  # Spec §11.1.6: constant is 2 for V1-3, 4 for V4-5, 8 for V6+.
  # We store both the raw header word and the reconstructed byte length.
  if version >= 3:
    raw_length = _word(data, 0x1A)
    if version <= 3:
      multiplier = 2
    elif version <= 5:
      multiplier = 4
    else:
      multiplier = 8
    scalar["file_length_raw"] = raw_length
    scalar["file_length_bytes"] = raw_length * multiplier

  # 0x1C: Checksum. V3+.
  if version >= 3:
    scalar["checksum"] = _word(data, 0x1C)

  # 0x1E: Interpreter number. V4+.
  if version >= 4:
    scalar["interpreter_number"] = _byte(data, 0x1E)

  # 0x1F: Interpreter version. V4+.
  # Spec §11.1.3.1: conventionally ASCII uppercase letter in V4-5,
  # numeric in V6.
  if version >= 4:
    scalar["interpreter_version"] = _byte(data, 0x1F)

  # 0x20: Screen height in lines. V4+. 255 = infinite.
  if version >= 4:
    scalar["screen_height_lines"] = _byte(data, 0x20)

  # 0x21: Screen width in characters. V4+.
  if version >= 4:
    scalar["screen_width_chars"] = _byte(data, 0x21)

  # 0x22: Screen width in units. V5+.
  if version >= 5:
    scalar["screen_width_units"] = _word(data, 0x22)

  # 0x24: Screen height in units. V5+.
  if version >= 5:
    scalar["screen_height_units"] = _word(data, 0x24)

  # 0x26/0x27: Font dimensions. V5+.
  # CRITICAL VERSION-CONDITIONAL SWAP (spec §11, VARIANT at offsets 26/27):
  #   V5:  0x26 = font WIDTH,  0x27 = font HEIGHT
  #   V6+: 0x26 = font HEIGHT, 0x27 = font WIDTH
  # Ontology: zm:field_font_width_v5 supersededBy zm:field_font_height_v6_at26
  #           zm:field_font_height_v5 supersededBy zm:field_font_width_v6_at27
  if version == 5:
    scalar["font_width"] = _byte(data, 0x26)
    scalar["font_height"] = _byte(data, 0x27)
  elif version >= 6:
    scalar["font_height"] = _byte(data, 0x26)  # swapped in V6
    scalar["font_width"] = _byte(data, 0x27)  # swapped in V6

  # 0x28: Routines offset. V6+ only.
  if version >= 6:
    scalar["routines_offset"] = _word(data, 0x28)

  # 0x2A: Static strings offset. V6+ only.
  if version >= 6:
    scalar["static_strings_offset"] = _word(data, 0x2A)

  # 0x2C: Default background colour. V5+.
  if version >= 5:
    scalar["default_background_colour"] = _byte(data, 0x2C)

  # 0x2D: Default foreground colour. V5+.
  if version >= 5:
    scalar["default_foreground_colour"] = _byte(data, 0x2D)

  # 0x2E: Terminating characters table address. V5+.
  if version >= 5:
    scalar["terminating_chars_address"] = _word(data, 0x2E)

  # 0x30: Total width of output stream 3 text (pixels). V6+.
  if version >= 6:
    scalar["stream3_width"] = _word(data, 0x30)

  # 0x32: Standard revision number (2 bytes: major, minor). V1+.
  # Spec §11.1.5: interpreter writes n.m here if it conforms to
  # revision n.m. Non-standard interpreters leave as 0.
  scalar["standard_revision"] = {
    "major": _byte(data, 0x32),
    "minor": _byte(data, 0x33),
  }

  # 0x34: Alphabet table address. V5+. 0 = use default alphabet.
  if version >= 5:
    scalar["alphabet_table_address"] = _word(data, 0x34)

  # 0x36: Header extension table pointer. V5+.
  ext_pointer = 0
  if version >= 5:
    ext_pointer = _word(data, 0x36)
    scalar["header_extension_table_address"] = ext_pointer

  # -------------------------------------------------------------------
  # Flags 1 (offset 0x01) — version-conditional register layout
  # -------------------------------------------------------------------
  flags1_raw = _byte(data, 0x01)
  flags1 = _decode_flags1(flags1_raw, version)

  # -------------------------------------------------------------------
  # Flags 2 (offset 0x10, 2 bytes) — per-bit version applicability
  # -------------------------------------------------------------------
  flags2_raw = _word(data, 0x10)
  flags2 = _decode_flags2(flags2_raw, version)

  # -------------------------------------------------------------------
  # Header extension table (V5+, only if pointer is non-zero)
  # -------------------------------------------------------------------
  extension_table = _decode_extension_table(data, version, ext_pointer)

  # -------------------------------------------------------------------
  # Conventional fields (Appendix B)
  # These are not normatively required by Section 11 but are present
  # by convention in all real Infocom and Inform story files.
  # Ontology: hasSourceAuthority zm:conventional.
  # -------------------------------------------------------------------
  conventional: dict = {}

  # 0x02: Release number. V1+. Ontology: zm:field_release_number.
  conventional["release_number"] = _word(data, 0x02)

  # 0x12: Serial code (V2: general ASCII; V3+: YYMMDD compilation date).
  # Ontology: zm:field_serial_v2 supersededBy zm:field_serial_v3plus.
  serial_raw = _ascii(data, 0x12, 6)
  conventional["serial_code"] = serial_raw
  if version >= 3:
    # Attempt to parse as YYMMDD; leave as raw string if not numeric.
    if serial_raw.isdigit():
      yy = int(serial_raw[:2])
      century = "19" if yy >= 80 else "20"
      conventional["serial_code_date"] = (
        f"{century}{serial_raw[:2]}-{serial_raw[2:4]}-{serial_raw[4:6]}"
      )

  # 0x38: Player username (8 bytes ASCII). V6 only.
  # Ontology: zm:field_player_username. All zero in shipped story files.
  if version >= 6:
    username_raw = data[0x38:0x40]
    conventional["player_username"] = (
      None if all(b == 0 for b in username_raw) else _ascii(data, 0x38, 8)
    )

  # 0x3C: Inform compiler version (4 bytes ASCII). All versions.
  # Ontology: zm:field_inform_version. Present only in Inform-compiled
  # files; zero bytes in Infocom originals.
  inform_raw = data[0x3C:0x40]
  if all(b == 0 for b in inform_raw):
    conventional["inform_compiler_version"] = None  # Infocom file, not Inform
  else:
    conventional["inform_compiler_version"] = _ascii(data, 0x3C, 4)

  # -------------------------------------------------------------------
  # Assemble and return
  # -------------------------------------------------------------------
  return {
    "version": version,
    "scalar_fields": scalar,
    "flags1": flags1,
    "flags2": flags2,
    "extension_table": extension_table,
    "conventional": conventional,
  }


# ---------------------------------------------------------------------------
# __main__: pretty-print the header for a story file
# ---------------------------------------------------------------------------

if __name__ == "__main__":
  if len(sys.argv) < 2:
    print("Usage: python zmachine_header_reference.py <story_file>")
    sys.exit(1)

  path = Path(sys.argv[1])
  if not path.exists():
    print(f"Error: file not found: {path}", file=sys.stderr)
    sys.exit(1)

  data = path.read_bytes()

  try:
    result = parse_header(data)
  except ValueError as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)

  print(json.dumps(result, indent=2))
