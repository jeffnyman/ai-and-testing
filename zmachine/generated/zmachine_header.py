import sys


def parse_zmachine_header(story_file):
  if len(story_file) < 64:
    raise ValueError("Story file must be at least 64 bytes long.")

  version = story_file[0x00]

  def read_byte(offset):
    return story_file[offset]

  def read_word(offset):
    return (story_file[offset] << 8) | story_file[offset + 1]

  def decode_flags(flags, bit_definitions):
    return {name: bool(flags & (1 << pos)) for pos, name in bit_definitions}

  # Scalar fields
  header = {
    "version": version,
    "high_memory_base": read_word(0x04),
    "initial_pc": read_word(0x06)
    if version < 6
    else read_word(0x06) * 8,  # Packed address for V6
    "dictionary_address": read_word(0x08),
    "object_table_address": read_word(0x0A),
    "global_variables_address": read_word(0x0C),
    "static_memory_base": read_word(0x0E),
    "abbreviations_address": read_word(0x18) if version >= 2 else None,
    "file_length": read_word(0x1A) if version >= 3 else None,
    "checksum": read_word(0x1C) if version >= 3 else None,
    "interpreter_number": read_byte(0x1E) if version >= 4 else None,
    "interpreter_version": read_byte(0x1F) if version >= 4 else None,
    "screen_height": read_byte(0x20) if version >= 4 else None,
    "screen_width_chars": read_byte(0x21) if version >= 4 else None,
    "screen_width_units": read_word(0x22) if version >= 5 else None,
    "screen_height_units": read_word(0x24) if version >= 5 else None,
    "font_width": read_byte(0x26)
    if version == 5
    else read_byte(0x27),  # Font width in V5, height in V6
    "font_height": read_byte(0x27)
    if version == 5
    else read_byte(0x26),  # Font height in V5, width in V6
    "routines_offset": read_word(0x28) * 8 if version >= 6 else None,
    "static_strings_offset": read_word(0x2A) * 8 if version >= 6 else None,
    "default_background_colour": read_byte(0x2C) if version >= 5 else None,
    "default_foreground_colour": read_byte(0x2D) if version >= 5 else None,
    "terminating_chars_address": read_word(0x2E) if version >= 5 else None,
    "stream3_width": read_word(0x30) if version >= 6 else None,
    "standard_revision": (read_byte(0x32), read_byte(0x33)),
    "alphabet_table_address": read_word(0x34) if version >= 5 else None,
  }

  # Flags 1
  flags1 = read_byte(0x01)
  if version <= 3:
    header["flags1"] = decode_flags(
      flags1,
      [
        (1, "status_line_type_score_turns"),
        (2, "story_split_across_two_discs"),
        (4, "status_line_not_available"),
        (5, "screen_splitting_available"),
        (6, "variable_pitch_font_default"),
      ],
    )
  else:
    header["flags1"] = decode_flags(
      flags1,
      [
        (0, "colours_available"),
        (1, "picture_displaying_available"),
        (2, "boldface_available"),
        (3, "italic_available"),
        (4, "fixed_space_style_available"),
        (5, "sound_effects_available"),
        (7, "timed_keyboard_input_available"),
      ],
    )

  # Flags 2
  flags2 = read_word(0x10)
  header["flags2"] = decode_flags(
    flags2,
    [
      (0, "transcripting_on"),
      (1, "force_fixed_pitch"),
      (2, "screen_redraw_request"),
      (3, "game_wants_pictures"),
      (4, "game_wants_undo"),
      (5, "game_wants_mouse"),
      (6, "game_wants_colours"),  # No Int column
      (7, "game_wants_sound_effects"),
      (8, "game_wants_menus"),
    ],
  )

  # Conventional fields
  header["conventional"] = {
    "release_number": read_word(0x02),
    "serial_code": story_file[0x12:0x18].decode("ascii"),
    "player_username": story_file[0x38:0x40].decode("ascii") if version >= 6 else None,
    "inform_version": story_file[0x3C:0x40].decode("ascii"),
  }

  # Header extension table
  if version >= 5:
    ext_ptr = read_word(0x36)
    if ext_ptr != 0 and len(story_file) > ext_ptr + 2:
      ext_count = read_word(ext_ptr)
      header["extension_table"] = {
        "count": ext_count,
        "mouse_x_coordinate": read_word(ext_ptr + 2) if ext_count > 1 else None,
        "mouse_y_coordinate": read_word(ext_ptr + 4) if ext_count > 2 else None,
        "unicode_translation_table_address": read_word(ext_ptr + 6)
        if ext_count > 3
        else None,
      }

      # Flags 3
      flags3 = read_word(ext_ptr + 8) if ext_count > 4 else 0
      header["extension_table"]["flags3"] = decode_flags(
        flags3, [(0, "game_wants_transparency")]
      )

      # True default colours
      header["extension_table"]["true_default_foreground_colour"] = (
        read_word(ext_ptr + 10) if ext_count > 5 else None
      )
      header["extension_table"]["true_default_background_colour"] = (
        read_word(ext_ptr + 12) if ext_count > 6 else None
      )

  return header


if __name__ == "__main__":
  import json
  import sys

  if len(sys.argv) != 2:
    print("Usage: python zmachine_header.py <story_file_path>")
    sys.exit(1)

  with open(sys.argv[1], "rb") as f:
    story_file = f.read()

  header = parse_zmachine_header(story_file)
  print(json.dumps(header, indent=4))
