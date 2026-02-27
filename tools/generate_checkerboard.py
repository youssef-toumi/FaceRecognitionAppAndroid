"""
Generate a printable checkerboard calibration pattern.

Standard: 9x6 inner corners (10x7 squares)
Print on A4 paper, glue to a flat rigid surface (cardboard, clipboard, etc.)

Usage:
    pip install Pillow
    python generate_checkerboard.py

Output: checkerboard_9x6.png  (ready to print)
"""

from PIL import Image, ImageDraw, ImageFont
import os

# ── Configuration ──────────────────────────────────────────────
INNER_COLS = 9          # inner corners horizontally
INNER_ROWS = 6          # inner corners vertically
SQUARE_SIZE_MM = 25     # each square = 25mm (measure after printing to verify)

# Derived
NUM_COLS = INNER_COLS + 1   # 10 squares wide
NUM_ROWS = INNER_ROWS + 1   # 7 squares tall

# A4 at 300 DPI
DPI = 300
A4_WIDTH_MM = 210
A4_HEIGHT_MM = 297
A4_WIDTH_PX = int(A4_WIDTH_MM / 25.4 * DPI)    # 2480
A4_HEIGHT_PX = int(A4_HEIGHT_MM / 25.4 * DPI)  # 3508

SQUARE_SIZE_PX = int(SQUARE_SIZE_MM / 25.4 * DPI)  # ~295 px per square

# ── Generate ───────────────────────────────────────────────────
img = Image.new("RGB", (A4_WIDTH_PX, A4_HEIGHT_PX), "white")
draw = ImageDraw.Draw(img)

# Center the board on the page
board_width = NUM_COLS * SQUARE_SIZE_PX
board_height = NUM_ROWS * SQUARE_SIZE_PX
offset_x = (A4_WIDTH_PX - board_width) // 2
offset_y = (A4_HEIGHT_PX - board_height) // 2

for row in range(NUM_ROWS):
    for col in range(NUM_COLS):
        if (row + col) % 2 == 0:
            color = "black"
        else:
            color = "white"
        x0 = offset_x + col * SQUARE_SIZE_PX
        y0 = offset_y + row * SQUARE_SIZE_PX
        x1 = x0 + SQUARE_SIZE_PX
        y1 = y0 + SQUARE_SIZE_PX
        draw.rectangle([x0, y0, x1, y1], fill=color)

# Draw border around the board for easy cutting
draw.rectangle(
    [offset_x - 2, offset_y - 2,
     offset_x + board_width + 2, offset_y + board_height + 2],
    outline="gray", width=2
)

# Add label text at the bottom
label = (
    f"Stereo Calibration Board  |  "
    f"{INNER_COLS}x{INNER_ROWS} inner corners  |  "
    f"Square = {SQUARE_SIZE_MM}mm  |  "
    f"Print at 100% scale on A4 (no fit-to-page)"
)
try:
    font = ImageFont.truetype("arial.ttf", 28)
except OSError:
    font = ImageFont.load_default()

text_y = offset_y + board_height + 40
draw.text((offset_x, text_y), label, fill="black", font=font)

# ── Save ───────────────────────────────────────────────────────
output_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(output_dir, "checkerboard_9x6.png")
img.save(output_path, dpi=(DPI, DPI))
print(f"✅ Saved: {output_path}")
print(f"   Board: {INNER_COLS}x{INNER_ROWS} inner corners, {SQUARE_SIZE_MM}mm squares")
print(f"   Image: {A4_WIDTH_PX}x{A4_HEIGHT_PX} px @ {DPI} DPI")
print()
print("⚠️  IMPORTANT when printing:")
print("   1. Print at 100% scale (disable 'Fit to page')")
print("   2. After printing, measure a square with a ruler")
print(f"   3. It should be exactly {SQUARE_SIZE_MM}mm")
print("   4. If not, note the ACTUAL size — you'll need it for calibration")
print("   5. Glue the print to something flat & rigid (cardboard, clipboard)")
