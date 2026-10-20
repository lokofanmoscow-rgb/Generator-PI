"""Распознавание мелких номеров с выносными линиями на синтетическом снимке."""
import math
import os
import random

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from generator_pi.ocr import NumberReader, connected_components

FONT_CANDIDATES = [
    r'C:\Windows\Fonts\calibri.ttf', r'C:\Windows\Fonts\arial.ttf',
    '/usr/share/fonts/truetype/crosextra/Carlito-Regular.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
]
FONT = next((f for f in FONT_CANDIDATES if os.path.exists(f)), None)


def _make_photo(path, digit_h, seed=5):
    rnd = random.Random(seed)
    w, h = 3200, 2400
    rng = np.random.default_rng(seed)
    base = rng.normal(150, 18, (h // 8, w // 8, 3)).clip(0, 255).astype(np.uint8)
    im = Image.fromarray(base).resize((w, h), Image.BICUBIC)       # «фон» снимка
    d = ImageDraw.Draw(im)
    font = ImageFont.truetype(FONT, int(digit_h / 0.7))
    gt, placed = [], []
    for i in range(24):
        num = 101 + i
        for _ in range(500):
            x, y = rnd.randint(100, w - 300), rnd.randint(100, h - 200)
            bb = d.textbbox((x, y), str(num), font=font)
            box = (bb[0] - 40, bb[1] - 40, bb[2] + 40, bb[3] + 40)
            if all(box[2] < p[0] or box[0] > p[2] or box[3] < p[1] or box[1] > p[3] for p in placed):
                break
        placed.append(box)
        d.text((x, y), str(num), font=font, fill=(10, 10, 10))
        # выносная линия, вплотную к цифрам
        cy = (bb[1] + bb[3]) / 2
        if i % 3 == 0:
            p0, ang = (bb[2], cy), 0
        elif i % 3 == 1:
            p0, ang = ((bb[0] + bb[2]) / 2, bb[3]), 90
        else:
            p0, ang = (bb[0], cy), 200
        p1 = (p0[0] + 4 * digit_h * math.cos(math.radians(ang)),
              p0[1] + 4 * digit_h * math.sin(math.radians(ang)))
        d.line([p0, p1], fill=(0, 0, 0), width=max(1, int(digit_h * 0.1)))
        gt.append(num)
    im.save(path, quality=88)
    return set(gt)


@pytest.fixture(scope='module')
def reader():
    return NumberReader()


@pytest.mark.skipif(FONT is None, reason='нет шрифта для синтетического снимка')
@pytest.mark.parametrize('digit_h', [16, 28])
def test_small_labels_with_touching_lines(tmp_path, reader, digit_h):
    path = str(tmp_path / f'labels_{digit_h}.jpg')
    gt = _make_photo(path, digit_h)
    found = set(reader.read_numbers(path))
    assert len(found & gt) >= 0.85 * len(gt), sorted(gt - found)
    assert len(found - gt) <= 1, sorted(found - gt)


def test_connected_components_matches_naive():
    rng = np.random.default_rng(0)
    mask = rng.random((60, 80)) > 0.7
    comps = connected_components(mask)
    assert sum(c.area for c in comps) == int(mask.sum())
    # каждая точка маски принадлежит ровно одной области
    seen = np.zeros_like(mask, dtype=int)
    for c in comps:
        for y, a, b in c.runs:
            seen[y, a:b] += 1
    assert (seen == mask.astype(int)).all()


def _make_dense_rows(path, digit_h=47):
    """Как на реальном фото: два плотных ряда подписей; выноски верхнего ряда подходят
    к цифрам сверху, выноски нижнего ряда проходят вверх в промежутках между подписями
    верхнего; рядом — тёмные прорези (не должны читаться как «1»)."""
    w, h = 3000, 1600
    rng = np.random.default_rng(3)
    base = rng.normal(120, 12, (h // 8, w // 8, 3)).clip(0, 255).astype(np.uint8)
    im = Image.fromarray(base).resize((w, h), Image.BICUBIC)
    d = ImageDraw.Draw(im)
    font = ImageFont.truetype(FONT, int(digit_h / 0.7))
    lw = max(2, int(digit_h * 0.25))
    pitch = int(3.2 * digit_h)
    top_y, low_y = 700, 700 + int(2.0 * digit_h)
    for i, (top, low) in enumerate([(108, 107), (110, 109), (112, 111)]):
        tx = 700 + i * pitch
        lx = tx - pitch // 2
        for num, x, y in ((top, tx, top_y), (low, lx, low_y)):
            bb = d.textbbox((x, y), str(num), font=font)
            d.text((x, y), str(num), font=font, fill=(5, 5, 5))
            cx = (bb[0] + bb[2]) / 2
            d.line([(cx, bb[1] + 3), (cx, 250)], fill=(0, 0, 0), width=lw)
    for k in range(6):                                  # прорези в металлической пластине
        x = 400 + k * 60
        d.rounded_rectangle((x, 1250, x + 18, 1250 + int(2.4 * digit_h)), radius=8, fill=(25, 25, 25))
    im.save(path, quality=90)
    return {107, 108, 109, 110, 111, 112}


@pytest.mark.skipif(FONT is None, reason='нет шрифта для синтетического снимка')
def test_dense_rows_with_crossing_lines_and_slots(tmp_path, reader):
    path = str(tmp_path / 'dense.jpg')
    gt = _make_dense_rows(path)
    found = set(reader.read_numbers(path))
    assert gt <= found, sorted(gt - found)
    assert not found - gt, sorted(found - gt)


@pytest.mark.skipif(FONT is None, reason='нет шрифта для синтетического снимка')
def test_labels_at_photo_edges(tmp_path, reader):
    """Подписи у самого края снимка, к крайней цифре которых вплотную подходит линия,
    не должны ронять распознавание."""
    w, h = 1600, 1200
    im = Image.new('RGB', (w, h), (60, 140, 120))
    d = ImageDraw.Draw(im)
    font = ImageFont.truetype(FONT, 40)
    for num, (x, y, right) in {21: (2, 0, True), 22: (w - 50, 0, False),
                               23: (2, h - 27, True), 24: (w - 50, h - 27, False)}.items():
        bb = d.textbbox((0, 0), str(num), font=font)
        d.text((x - bb[0], y - bb[1]), str(num), font=font, fill=(0, 0, 0))
        bb = d.textbbox((x - bb[0], y - bb[1]), str(num), font=font)
        cy = (bb[1] + bb[3]) // 2
        line = [(bb[2] - 4, cy), (bb[2] + 80, cy)] if right else [(bb[0] - 80, cy), (bb[0] + 4, cy)]
        d.line(line, fill=(0, 0, 0), width=4)
    path = str(tmp_path / 'edges.jpg')
    im.save(path, quality=90)
    found = set(reader.read_numbers(path))
    assert not found - {21, 22, 23, 24}


@pytest.mark.skipif(FONT is None, reason='нет шрифта для синтетического снимка')
def test_white_digits_on_dark_parts_are_ignored(tmp_path, reader):
    """Белая маркировка на тёмных корпусах деталей («101» на SMD-резисторе) — не подписи."""
    w, h = 2000, 1400
    im = Image.new('RGB', (w, h), (30, 110, 70))            # зелёная плата
    d = ImageDraw.Draw(im)
    small = ImageFont.truetype(FONT, 34)
    for k in range(8):                                        # резисторы и микросхемы с маркировкой
        x, y = 300 + (k % 4) * 350, 300 + (k // 4) * 400
        d.rectangle((x, y, x + 150, y + 70), fill=(15, 15, 15))
        d.text((x + 25, y + 12), ('101', '730', '103', '472')[k % 4], font=small, fill=(235, 235, 235))
    font = ImageFont.truetype(FONT, 48)
    for i, num in enumerate((265, 266, 267)):                 # настоящие подписи — на светлом поле
        x, y = 200 + i * 500, 1150
        d.rectangle((x - 30, y - 30, x + 160, y + 80), fill=(200, 215, 205))
        d.text((x, y), str(num), font=font, fill=(0, 0, 0))
    path = str(tmp_path / 'board.jpg')
    im.save(path, quality=90)
    found = set(reader.read_numbers(path))
    assert {265, 266, 267} <= found, sorted(found)
    assert not found - {265, 266, 267}, sorted(found)
