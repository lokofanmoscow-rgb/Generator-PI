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
