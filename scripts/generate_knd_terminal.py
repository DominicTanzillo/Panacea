"""KND-style PANACEA intro — terminal cursor version.

Phase 1: PANACEA appears horizontally centered
Phase 2: Letters migrate one by one to vertical
Phase 3: A blinking cursor types out each word character by character,
         moving down to the next line after each word completes.
Phase 4: Subtitle, hold
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

# --- Config ---
W, H = 1280, 720
FPS = 30
BG_COLOR = (12, 12, 18)
ACCENT = (59, 130, 246)
TEXT_COLOR = (232, 232, 240)
DIM_COLOR = (85, 85, 106)
RED = (239, 68, 68)
CURSOR_COLOR = (59, 130, 246)  # Blue cursor

ACRONYM = [
    ("P", "REDICTIVE"),
    ("A", "NALYSIS"),
    ("N", "ETWORK FOR"),
    ("A", "UTOMATED"),
    ("C", "ONJUNCTION"),
    ("E", "SCALATION"),
    ("A", "LERTS"),
]

LETTERS = "PANACEA"

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "webapp-react", "public", "panacea_intro_terminal.mp4")


def get_font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/cour.ttf",
        "C:/Windows/Fonts/lucon.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def ease_in_out_cubic(t: float) -> float:
    if t < 0.5:
        return 4 * t * t * t
    return 1 - (-2 * t + 2) ** 3 / 2


def ease_out_expo(t: float) -> float:
    return 1.0 if t >= 1.0 else 1.0 - 2.0 ** (-10.0 * t)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def measure_char(font: ImageFont.FreeTypeFont, ch: str) -> int:
    bbox = font.getbbox(ch)
    return bbox[2] - bbox[0]


def measure_text(font: ImageFont.FreeTypeFont, text: str) -> int:
    bbox = font.getbbox(text)
    return bbox[2] - bbox[0]


def draw_background(img: Image.Image):
    draw = ImageDraw.Draw(img)
    for y in range(0, H, 4):
        if y % 8 == 0:
            draw.line([(0, y), (W, y)], fill=(18, 18, 26), width=1)
    draw.rectangle([(0, 0), (W, 3)], fill=ACCENT)
    header_font = get_font(16)
    draw.text((40, 20), "OPERATION DESIGNATION //", fill=DIM_COLOR, font=header_font)
    draw.rectangle([(30, 50), (W - 30, H - 50)], outline=(30, 30, 45), width=1)
    draw.rectangle([(34, 54), (W - 34, H - 54)], outline=(25, 25, 38), width=1)
    wm_font = get_font(14)
    draw.text((40, H - 30), "CLASSIFIED // SPACE DEFENSE // PANACEA PROJECT", fill=(30, 30, 42), font=wm_font)


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    letter_font = get_font(52)
    word_font = get_font(28)

    # Horizontal positions (centered PANACEA)
    char_widths = [measure_char(letter_font, ch) for ch in LETTERS]
    total_word_w = sum(char_widths) + (len(LETTERS) - 1) * 8
    horiz_y = H // 2 - 30
    horiz_xs = []
    cx = (W - total_word_w) // 2
    for cw in char_widths:
        horiz_xs.append(cx)
        cx += cw + 8

    # Vertical (final) positions
    vert_x = 100
    start_y = 95
    line_height = 72
    vert_positions = [(vert_x, start_y + i * line_height) for i in range(len(LETTERS))]

    # Word x-offsets and character widths
    word_offsets = []
    for letter, rest in ACRONYM:
        lw = measure_char(letter_font, letter)
        word_offsets.append(lw + 2)

    dot_w = measure_char(word_font, ".")

    # Cursor dimensions
    cursor_w = measure_char(word_font, "M")
    cursor_h = 28

    # Blink rate: cursor visible for half of each cycle
    CURSOR_BLINK_FRAMES = int(FPS * 0.5)  # full blink cycle

    frames = []

    # ======== PHASE 1: Fade in PANACEA horizontally (1.0s + 0.5s hold) ========
    for f in range(int(FPS * 1.0)):
        t = ease_out_expo(f / (FPS * 1.0))
        img = Image.new("RGB", (W, H), BG_COLOR)
        draw_background(img)
        draw = ImageDraw.Draw(img)
        color = tuple(int(c * t) for c in TEXT_COLOR)
        for i, ch in enumerate(LETTERS):
            draw.text((horiz_xs[i], horiz_y), ch, fill=color, font=letter_font)
        frames.append(np.array(img)[:, :, ::-1])

    for f in range(int(FPS * 0.5)):
        img = Image.new("RGB", (W, H), BG_COLOR)
        draw_background(img)
        draw = ImageDraw.Draw(img)
        for i, ch in enumerate(LETTERS):
            draw.text((horiz_xs[i], horiz_y), ch, fill=TEXT_COLOR, font=letter_font)
        frames.append(np.array(img)[:, :, ::-1])

    # ======== PHASE 2: Letters migrate one by one to vertical (0.5s each) ========
    migrate_dur = int(FPS * 0.5)

    for target_idx in range(len(LETTERS)):
        for f in range(migrate_dur):
            t = ease_in_out_cubic(f / migrate_dur)
            img = Image.new("RGB", (W, H), BG_COLOR)
            draw_background(img)
            draw = ImageDraw.Draw(img)

            for i, ch in enumerate(LETTERS):
                if i < target_idx:
                    vx, vy = vert_positions[i]
                    draw.text((vx, vy), ch, fill=TEXT_COLOR, font=letter_font)
                elif i == target_idx:
                    sx, sy = horiz_xs[i], horiz_y
                    ex, ey = vert_positions[i]
                    x = int(lerp(sx, ex, t))
                    y = int(lerp(sy, ey, t))
                    draw.text((x, y), ch, fill=TEXT_COLOR, font=letter_font)
                else:
                    draw.text((horiz_xs[i], horiz_y), ch, fill=TEXT_COLOR, font=letter_font)

            frames.append(np.array(img)[:, :, ::-1])

    # Brief hold (0.3s)
    for f in range(int(FPS * 0.3)):
        img = Image.new("RGB", (W, H), BG_COLOR)
        draw_background(img)
        draw = ImageDraw.Draw(img)
        for i, ch in enumerate(LETTERS):
            vx, vy = vert_positions[i]
            draw.text((vx, vy), ch, fill=TEXT_COLOR, font=letter_font)
        frames.append(np.array(img)[:, :, ::-1])

    # ======== PHASE 3: Terminal cursor types each word ========
    CHAR_RATE = 15  # chars per second (slowed ~15%)
    frames_per_char = max(1, int(FPS / CHAR_RATE))
    global_frame = 0  # for consistent cursor blink

    for target_idx in range(len(LETTERS)):
        letter_str, rest = ACRONYM[target_idx]
        # Full text to type: ".REST"
        full_text = "." + rest
        n_chars = len(full_text)

        # Typing phase: one char at a time
        for ci in range(n_chars):
            for cf in range(frames_per_char):
                img = Image.new("RGB", (W, H), BG_COLOR)
                draw_background(img)
                draw = ImageDraw.Draw(img)

                for i, ch in enumerate(LETTERS):
                    vx, vy = vert_positions[i]
                    draw.text((vx, vy), ch, fill=TEXT_COLOR, font=letter_font)

                    if i < target_idx:
                        _, r = ACRONYM[i]
                        draw.text((vx + word_offsets[i], vy + 16), ".", fill=RED, font=word_font)
                        draw.text((vx + word_offsets[i] + dot_w, vy + 16), r, fill=TEXT_COLOR, font=word_font)
                    elif i == target_idx:
                        typed_so_far = full_text[:ci + 1]
                        # Draw dot in red, rest in white
                        if typed_so_far:
                            draw.text((vx + word_offsets[i], vy + 16), typed_so_far[0], fill=RED, font=word_font)
                            if len(typed_so_far) > 1:
                                draw.text((vx + word_offsets[i] + dot_w, vy + 16), typed_so_far[1:], fill=TEXT_COLOR, font=word_font)

                        # Blinking cursor after typed text
                        typed_w = measure_text(word_font, typed_so_far) if typed_so_far else 0
                        cursor_x = vx + word_offsets[i] + typed_w + 2
                        cursor_y = vy + 16

                        cursor_visible = (global_frame % CURSOR_BLINK_FRAMES) < (CURSOR_BLINK_FRAMES // 2)
                        if cursor_visible:
                            draw.rectangle(
                                [(cursor_x, cursor_y + 2), (cursor_x + cursor_w, cursor_y + cursor_h)],
                                fill=CURSOR_COLOR,
                            )

                global_frame += 1
                frames.append(np.array(img)[:, :, ::-1])

        # After last char typed, hold briefly with blinking cursor (0.15s)
        for f in range(int(FPS * 0.15)):
            img = Image.new("RGB", (W, H), BG_COLOR)
            draw_background(img)
            draw = ImageDraw.Draw(img)

            for i, ch in enumerate(LETTERS):
                vx, vy = vert_positions[i]
                draw.text((vx, vy), ch, fill=TEXT_COLOR, font=letter_font)

                if i <= target_idx:
                    _, r = ACRONYM[i]
                    draw.text((vx + word_offsets[i], vy + 16), ".", fill=RED, font=word_font)
                    draw.text((vx + word_offsets[i] + dot_w, vy + 16), r, fill=TEXT_COLOR, font=word_font)

                    # Cursor on current line only
                    if i == target_idx:
                        full = "." + r
                        typed_w = measure_text(word_font, full)
                        cursor_x = vx + word_offsets[i] + typed_w + 2
                        cursor_y = vy + 16
                        cursor_visible = (global_frame % CURSOR_BLINK_FRAMES) < (CURSOR_BLINK_FRAMES // 2)
                        if cursor_visible:
                            draw.rectangle(
                                [(cursor_x, cursor_y + 2), (cursor_x + cursor_w, cursor_y + cursor_h)],
                                fill=CURSOR_COLOR,
                            )

            global_frame += 1
            frames.append(np.array(img)[:, :, ::-1])

    # ======== PHASE 4: Cursor disappears, subtitle fades in (1s) ========
    sub_font = get_font(20)
    sub_text = "Satellite Collision Avoidance Through Deep Learning"
    sub_bbox = sub_font.getbbox(sub_text)
    sub_tw = sub_bbox[2] - sub_bbox[0]
    sub_x = (W - sub_tw) // 2
    sub_y = start_y + len(ACRONYM) * line_height + 25

    def draw_final(draw: ImageDraw.ImageDraw, sub_alpha: float = 0.0):
        for i, ch in enumerate(LETTERS):
            vx, vy = vert_positions[i]
            draw.text((vx, vy), ch, fill=TEXT_COLOR, font=letter_font)
            _, rest = ACRONYM[i]
            draw.text((vx + word_offsets[i], vy + 16), ".", fill=RED, font=word_font)
            draw.text((vx + word_offsets[i] + dot_w, vy + 16), rest, fill=TEXT_COLOR, font=word_font)
        if sub_alpha > 0:
            color = tuple(int(c * sub_alpha) for c in ACCENT)
            draw.text((sub_x, sub_y), sub_text, fill=color, font=sub_font)
            if sub_alpha > 0.5:
                lp = (sub_alpha - 0.5) * 2
                lc = tuple(int(c * sub_alpha * 0.5) for c in ACCENT)
                draw.line([(sub_x, sub_y + 28), (sub_x + int(sub_tw * lp), sub_y + 28)], fill=lc, width=1)

    for f in range(int(FPS * 1.0)):
        alpha = ease_out_expo(f / (FPS * 1.0))
        img = Image.new("RGB", (W, H), BG_COLOR)
        draw_background(img)
        draw = ImageDraw.Draw(img)
        draw_final(draw, sub_alpha=alpha)
        frames.append(np.array(img)[:, :, ::-1])

    # ======== PHASE 5: Hold (2s) ========
    for f in range(int(FPS * 2.0)):
        img = Image.new("RGB", (W, H), BG_COLOR)
        draw_background(img)
        draw = ImageDraw.Draw(img)
        draw_final(draw, sub_alpha=1.0)
        frames.append(np.array(img)[:, :, ::-1])

    # ======== Write video ========
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, FPS, (W, H))
    for frame in frames:
        out.write(frame)
    out.release()

    total_seconds = len(frames) / FPS
    print(f"Generated {OUTPUT_PATH}")
    print(f"  {len(frames)} frames, {total_seconds:.1f}s at {FPS} fps")

    # Re-encode H.264
    h264_path = OUTPUT_PATH.replace(".mp4", "_h264.mp4")
    ret = os.system(
        f'ffmpeg -y -i "{OUTPUT_PATH}" -c:v libx264 -pix_fmt yuv420p '
        f'-crf 23 -preset medium -movflags +faststart "{h264_path}" 2>/dev/null'
    )
    if ret == 0:
        os.replace(h264_path, OUTPUT_PATH)
        size_kb = os.path.getsize(OUTPUT_PATH) / 1024
        print(f"  H.264: {size_kb:.0f} KB")
    else:
        print("  Warning: ffmpeg re-encode failed")


if __name__ == "__main__":
    main()
