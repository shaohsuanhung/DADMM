#!/usr/bin/env python3
"""Plot 3/7/10-sensor nonlinear MAP position/velocity RMSE and C-PCRLB."""

from pathlib import Path
import math

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "results" / "nonlinear_map_10mc_cr30_cr50_cr500"
OUTPUT_STEM = DATA_ROOT / "nonlinear_map_pos_vel_rmse_3_7_10_cpcrlb"
SNRS, CRS = (20, 30, 40), (30, 50, 500)
COLORS = ("#1769aa", "#d66b19", "#25845b")
INK, GRID = "#202124", "#d7dadd"


def font(size, bold=False):
    names = ("Times New Roman Bold.ttf" if bold else "Times New Roman.ttf",
             "Arial Bold.ttf" if bold else "Arial.ttf")
    for name in names:
        path = Path("/System/Library/Fonts/Supplemental") / name
        if path.exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def only_mat(folder):
    files = sorted(folder.glob("*.mat"))
    if len(files) != 1:
        raise RuntimeError(f"Expected exactly one MAT file in {folder}, got {len(files)}")
    return files[0]


def read_study(path):
    with h5py.File(path, "r") as handle:
        study = handle["Study"]
        time = np.asarray(study["plot_time"]).reshape(-1)
        rmse_raw = np.asarray(study["rmse_D_plot"])
        if rmse_raw.ndim == 2:
            rmse = rmse_raw.transpose(1, 0)[:, :, None, None]
        else:
            rmse = rmse_raw.transpose(3, 2, 1, 0)
        central_raw = np.asarray(study["pcrlb_C_plot"])
        if central_raw.ndim == 2:
            central = central_raw.transpose(1, 0)[:, :, None]
        else:
            central = central_raw.transpose(2, 1, 0)
    return time, rmse, central


def load_curves():
    result = {}
    for snr in (20, 30):
        time, rmse, central = read_study(only_mat(DATA_ROOT / f"snr{snr}"))
        result[snr] = (time, rmse[:, :, 0, :], central[:, :, 0])
    rmse40 = []
    central40 = None
    time40 = None
    for cr in CRS:
        time, rmse, central = read_study(
            only_mat(DATA_ROOT / f"snr40_cr{cr}"))
        if time40 is None:
            time40, central40 = time, central[:, :, 0]
        else:
            if not np.array_equal(time40, time):
                raise RuntimeError("40 dB time vectors differ across CR runs")
            if np.max(np.abs(central40 - central[:, :, 0])) > 1e-12:
                raise RuntimeError("40 dB centralized PCRLB differs across CR runs")
        rmse40.append(rmse[:, :, 0, 0])
    result[40] = (time40, np.stack(rmse40, axis=2), central40)
    return result


def dashed(draw, points, fill, width=4, dash=14, gap=9):
    if len(points) > 2:
        for index in range(0, len(points)-1, 2):
            draw.line([points[index], points[index+1]], fill=fill, width=width)
        return
    for a, b in zip(points[:-1], points[1:]):
        dx, dy = b[0]-a[0], b[1]-a[1]
        length = math.hypot(dx, dy)
        if length == 0:
            continue
        cursor = 0.0
        while cursor < length:
            end = min(cursor+dash, length)
            draw.line([(a[0]+dx*cursor/length, a[1]+dy*cursor/length),
                       (a[0]+dx*end/length, a[1]+dy*end/length)],
                      fill=fill, width=width)
            cursor += dash+gap


def aggregate(values, row):
    indices = (0, 1) if row == 0 else (2, 3)
    return np.sqrt(values[indices[0]]**2 + values[indices[1]]**2)


def main():
    curves = load_curves()
    width, height = 2100, 1050
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    normal, small, heading = font(27), font(23), font(31, True)
    left, right, top, bottom = 190, 40, 85, 150
    col_gap, row_gap = 52, 70
    panel_w = (width-left-right-2*col_gap)/3
    panel_h = (height-top-bottom-row_gap)/2

    row_ranges = []
    for row in range(2):
        all_values = []
        for snr in SNRS:
            _, rmse, central = curves[snr]
            all_values.extend(aggregate(rmse, row).reshape(-1))
            all_values.extend(aggregate(central, row).reshape(-1))
        positive = np.asarray(all_values)
        positive = positive[positive > 0]
        row_ranges.append((10**math.floor(math.log10(float(positive.min()))),
                           10**math.ceil(math.log10(float(positive.max())))))

    for col, snr in enumerate(SNRS):
        time, rmse, central = curves[snr]
        for row in range(2):
            x0, y0 = left+col*(panel_w+col_gap), top+row*(panel_h+row_gap)
            x1, y1 = x0+panel_w, y0+panel_h
            px0, px1, py0, py1 = x0+82, x1-16, y0+18, y1-52
            lo, hi = row_ranges[row]
            log_lo, log_hi = math.log10(lo), math.log10(hi)
            ymap = lambda value: py1-(math.log10(max(float(value),1e-14))-log_lo)/(log_hi-log_lo)*(py1-py0)
            for exponent in range(math.ceil(log_lo), math.floor(log_hi)+1):
                value, yy = 10**exponent, ymap(10**exponent)
                draw.line([(px0,yy),(px1,yy)],fill=GRID,width=1)
                label=f"1e{exponent}"; box=draw.textbbox((0,0),label,font=small)
                draw.text((px0-11-(box[2]-box[0]),yy-(box[3]-box[1])/2),label,fill=INK,font=small)
            for tick in (0.0,0.2,0.4,0.6):
                xx=px0+tick/time[-1]*(px1-px0)
                draw.line([(xx,py0),(xx,py1)],fill=GRID,width=1)
                label=f"{tick:.1f}"; box=draw.textbbox((0,0),label,font=small)
                draw.text((xx-(box[2]-box[0])/2,py1+8),label,fill=INK,font=small)
            draw.rectangle([(px0,py0),(px1,py1)],outline=INK,width=2)

            def points(values):
                return [(px0+tt/time[-1]*(px1-px0),ymap(value))
                        for tt,value in zip(time,values)]

            aggregate_rmse = aggregate(rmse, row)
            for cr_idx, color in enumerate(COLORS):
                draw.line(points(aggregate_rmse[:,cr_idx]),fill=color,width=4,joint="curve")
            dashed(draw,points(aggregate(central,row)),INK,width=4)
            if row == 0:
                label=f"{snr} dB"; box=draw.textbbox((0,0),label,font=heading)
                draw.text(((x0+x1-(box[2]-box[0]))/2,25),label,fill=INK,font=heading)
            if col == 0:
                label="Position RMSE (m)" if row==0 else "Velocity RMSE (m/s)"
                box=draw.textbbox((0,0),label,font=heading)
                draw.text((18,(y0+y1-(box[3]-box[1]))/2),label,fill=INK,font=heading)
            if row == 1:
                label="Time (s)"; box=draw.textbbox((0,0),label,font=normal)
                draw.text(((x0+x1-(box[2]-box[0]))/2,y1+48),label,fill=INK,font=normal)

    legend=((COLORS[0],False,"D-MAP, 3 sensors"),(COLORS[1],False,"D-MAP, 7 sensors"),(COLORS[2],False,"D-MAP, 10 sensors"),(INK,True,"Centralized PCRLB"))
    start_x,y=255,height-55
    for color,is_dashed,label in legend:
        segment=[(start_x,y),(start_x+70,y)]
        dashed(draw,segment,color,width=4) if is_dashed else draw.line(segment,fill=color,width=4)
        draw.text((start_x+82,y-16),label,fill=INK,font=normal)
        start_x+=465

    png,pdf=OUTPUT_STEM.with_suffix(".png"),OUTPUT_STEM.with_suffix(".pdf")
    image.save(png,dpi=(300,300)); image.save(pdf,"PDF",resolution=300)
    print(png); print(pdf)


if __name__ == "__main__":
    main()
