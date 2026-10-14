#!/usr/bin/env python3
"""Plot paired 7/10-sensor nonlinear MAP RMSE and matched PCRLB."""

from pathlib import Path
import math

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
RESULT_ROOT = ROOT / "results" / "nonlinear_map_5mc_cr50_cr500"
OUTPUT_STEM = RESULT_ROOT / "nonlinear_map_rmse_pcrlb_7_vs_10"
SNRS = (20, 30, 40)
STATE_LABELS = ("x position (m)", "y position (m)",
                "vx velocity (m/s)", "vy velocity (m/s)")
BLUE, ORANGE, INK, GRID = "#1769aa", "#d66b19", "#202124", "#d7dadd"


def font(size, bold=False):
    names = ("Times New Roman Bold.ttf" if bold else "Times New Roman.ttf",
             "Arial Bold.ttf" if bold else "Arial.ttf")
    for name in names:
        path = Path("/System/Library/Fonts/Supplemental") / name
        if path.exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def latest_mat(snr):
    files = sorted((RESULT_ROOT / f"snr{snr}").glob("*.mat"))
    if not files:
        raise FileNotFoundError(f"No MAT artifact found for {snr} dB")
    return files[-1]


def read_curves(path):
    with h5py.File(path, "r") as handle:
        study = handle["Study"]
        time = np.asarray(study["plot_time"]).reshape(-1)
        rmse = np.asarray(study["rmse_D_plot"]).transpose(3, 2, 1, 0)
        pcrlb = np.asarray(study["pcrlb_D_plot"]).transpose(3, 2, 1, 0)
    return time, rmse[:, :, 0, :], pcrlb[:, :, 0, :]


def dashed(draw, points, fill, width=3, dash=10, gap=7):
    for a, b in zip(points[:-1], points[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy)
        if length == 0:
            continue
        cursor = 0.0
        while cursor < length:
            end = min(cursor + dash, length)
            p0 = (a[0] + dx * cursor / length, a[1] + dy * cursor / length)
            p1 = (a[0] + dx * end / length, a[1] + dy * end / length)
            draw.line([p0, p1], fill=fill, width=width)
            cursor += dash + gap


def main():
    width, height = 2400, 1500
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    regular, small, heading = font(25), font(22), font(29, True)
    left, right, top, bottom = 155, 35, 70, 145
    col_gap, row_gap = 32, 50
    panel_w = (width - left - right - 3 * col_gap) / 4
    panel_h = (height - top - bottom - 2 * row_gap) / 3

    for row, snr in enumerate(SNRS):
        time, rmse, pcrlb = read_curves(latest_mat(snr))
        for state in range(4):
            x0 = left + state * (panel_w + col_gap)
            y0 = top + row * (panel_h + row_gap)
            x1, y1 = x0 + panel_w, y0 + panel_h
            px0, px1, py0, py1 = x0 + 74, x1 - 14, y0 + 18, y1 - 48
            series = (rmse[state, :, 0], rmse[state, :, 1],
                      pcrlb[state, :, 0], pcrlb[state, :, 1])
            positive = np.concatenate(series)
            positive = positive[positive > 0]
            lo = 10 ** math.floor(math.log10(float(positive.min())))
            hi = 10 ** math.ceil(math.log10(float(positive.max())))
            log_lo, log_hi = math.log10(lo), math.log10(hi)

            for exponent in range(math.ceil(log_lo), math.floor(log_hi) + 1):
                value = 10 ** exponent
                yy = py1 - (math.log10(value)-log_lo)/(log_hi-log_lo)*(py1-py0)
                draw.line([(px0, yy), (px1, yy)], fill=GRID, width=1)
                label = f"1e{exponent}"
                box = draw.textbbox((0, 0), label, font=small)
                draw.text((px0-10-(box[2]-box[0]), yy-(box[3]-box[1])/2),
                          label, fill=INK, font=small)
            for tick in (0.0, 0.2, 0.4, 0.6):
                xx = px0 + tick/time[-1]*(px1-px0)
                draw.line([(xx, py0), (xx, py1)], fill=GRID, width=1)
                label = f"{tick:.1f}"
                box = draw.textbbox((0, 0), label, font=small)
                draw.text((xx-(box[2]-box[0])/2, py1+8), label,
                          fill=INK, font=small)
            draw.rectangle([(px0, py0), (px1, py1)], outline=INK, width=2)

            def points(values):
                result = []
                for tt, value in zip(time, values):
                    xx = px0 + tt/time[-1]*(px1-px0)
                    yy = py1-(math.log10(max(float(value), 1e-14))-log_lo)/(log_hi-log_lo)*(py1-py0)
                    result.append((xx, yy))
                return result

            draw.line(points(series[0]), fill=BLUE, width=4, joint="curve")
            draw.line(points(series[1]), fill=ORANGE, width=4, joint="curve")
            dashed(draw, points(series[2]), BLUE)
            dashed(draw, points(series[3]), ORANGE)
            if row == 0:
                box = draw.textbbox((0, 0), STATE_LABELS[state], font=heading)
                draw.text(((x0+x1-(box[2]-box[0]))/2, 22),
                          STATE_LABELS[state], fill=INK, font=heading)
            if state == 0:
                draw.text((16, (y0+y1)/2-18), f"{snr} dB", fill=INK, font=heading)
            if row == 2:
                label = "Time (s)"
                box = draw.textbbox((0, 0), label, font=regular)
                draw.text(((x0+x1-(box[2]-box[0]))/2, y1+48),
                          label, fill=INK, font=regular)

    legend = ((BLUE, False, "7-sensor RMSE"),
              (ORANGE, False, "10-sensor RMSE"),
              (BLUE, True, "7-sensor PCRLB"),
              (ORANGE, True, "10-sensor PCRLB"))
    start_x, y = 500, height - 55
    for color, is_dashed, label in legend:
        segment = [(start_x, y), (start_x + 75, y)]
        if is_dashed:
            dashed(draw, segment, color, width=4, dash=14, gap=8)
        else:
            draw.line(segment, fill=color, width=4)
        draw.text((start_x + 88, y - 15), label, fill=INK, font=regular)
        start_x += 390

    OUTPUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    png, pdf = OUTPUT_STEM.with_suffix(".png"), OUTPUT_STEM.with_suffix(".pdf")
    image.save(png, dpi=(300, 300))
    image.save(pdf, "PDF", resolution=300)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
