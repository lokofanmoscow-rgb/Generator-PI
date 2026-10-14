"""Распознавание номеров компонентов на фотографиях.

Номера на фото — чёрные цифры, нанесённые поверх снимка (с выносными линиями).
Алгоритм:
  1. Маска «чернил»: пиксели, у которых все каналы тёмные.
  2. Связные области, похожие по форме на цифры, группируются в числа.
  3. Каждое число распознаётся CRNN-моделью (PP-OCRv3 rec, ONNX) с декодированием,
     ограниченным только цифрами.
  4. Отсев ложных срабатываний: по «чистоте» чёрного и по высоте цифр
     (все подписи на одном фото одного размера).

Затем assign_ranges() по всем фото сразу подбирает непрерывные диапазоны
(1…a, a+1…b, …), которые лучше всего согласуются с найденными числами.
"""
from __future__ import annotations

import bisect
import os
import sys
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps

INK_THRESHOLD = 90        # пиксель «чёрный», если все каналы < порога
MAX_SIDE = 1600           # анализ ведётся на уменьшенной копии
LABEL_DARKNESS_MAX = 50   # медиана яркости цифр подписи не выше
MIN_SCORE = 0.5
MAX_NUMBER = 999


@dataclass
class Detection:
    number: int
    score: float
    box: tuple[int, int, int, int]   # x, y, w, h на уменьшенной копии
    darkness: float


def resource_path(*parts: str) -> str:
    base = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    if getattr(sys, '_MEIPASS', None):
        base = os.path.join(base, 'generator_pi')
    return os.path.join(base, *parts)


def load_image(path: str) -> Image.Image:
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)
    return im.convert('RGB')


class NumberReader:
    def __init__(self, model_path: str | None = None):
        import onnxruntime as ort

        model_path = model_path or resource_path('models', 'digits_rec.onnx')
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(model_path, sess_options=opts,
                                            providers=['CPUExecutionProvider'])
        chars = self.session.get_modelmeta().custom_metadata_map['character'].splitlines()
        chars = ['<blank>'] + chars + [' ']
        self.digit_idx = np.array([0] + [chars.index(str(d)) for d in range(10)])
        self.input_name = self.session.get_inputs()[0].name

    # -- распознавание одного фрагмента ------------------------------------
    def _recognize(self, crop: np.ndarray) -> tuple[str, float]:
        """crop: uint8 HxW (0 — чернила, 255 — фон). Возвращает (цифры, уверенность)."""
        h, w = crop.shape
        new_w = max(8, int(round(48 * w / h)))
        img = Image.fromarray(crop).resize((new_w, 48), Image.BILINEAR)
        arr = (np.asarray(img, dtype=np.float32) / 255.0 - 0.5) / 0.5
        inp = np.zeros((1, 3, 48, max(new_w, 48)), np.float32)
        inp[0, :, :, :new_w] = arr[None, :, :]
        probs = self.session.run(None, {self.input_name: inp})[0][0]   # T x C
        sub = probs[:, self.digit_idx]
        best = sub.argmax(1)
        text, conf, prev = '', [], 0
        for t, k in enumerate(best):
            if k != 0 and k != prev:
                text += str(k - 1)
                conf.append(float(sub[t, k]))
            prev = k
        return text, (float(np.mean(conf)) if conf else 0.0)

    # -- поиск подписей ----------------------------------------------------
    def read(self, path: str) -> list[Detection]:
        im = load_image(path)
        im.thumbnail((MAX_SIDE, MAX_SIDE))
        a = np.asarray(im)
        brightness = a.max(axis=2)
        ink = brightness < INK_THRESHOLD
        side = max(a.shape[:2])

        glyphs = []
        for comp in connected_components(ink):
            x, y, w, h, area = comp.x, comp.y, comp.w, comp.h, comp.area
            if h < 0.012 * side or h > 0.12 * side:
                continue
            if w > 1.1 * h or w < 0.12 * h:
                continue
            fill = area / float(w * h)
            if fill < 0.12 or fill > 0.8:
                continue
            pixels = np.concatenate([brightness[ry, rx0:rx1] for ry, rx0, rx1 in comp.runs])
            dark = float(np.median(pixels))
            if dark > LABEL_DARKNESS_MAX:
                continue
            glyphs.append((x, y, w, h, dark))
        ink = ink.astype(np.uint8)

        dets: list[Detection] = []
        for group in _group_glyphs(glyphs):
            x1 = min(g[0] for g in group)
            y1 = min(g[1] for g in group)
            x2 = max(g[0] + g[2] for g in group)
            y2 = max(g[1] + g[3] for g in group)
            pad = int(0.2 * (y2 - y1)) + 2
            sub = ink[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
            crop = np.where(sub > 0, 0, 255).astype(np.uint8)
            text, score = self._recognize(crop)
            if not text or score < MIN_SCORE or len(text) > 3:
                continue
            n = int(text)
            if not 1 <= n <= MAX_NUMBER:
                continue
            dets.append(Detection(n, score, (x1, y1, x2 - x1, y2 - y1),
                                  max(g[4] for g in group)))
        return _filter_by_height(dets)

    def read_numbers(self, path: str) -> list[int]:
        return sorted({d.number for d in self.read(path)})


@dataclass
class Component:
    x: int
    y: int
    w: int
    h: int
    area: int
    runs: list[tuple[int, int, int]]   # (y, x0, x1) — отрезки строк, x1 не включительно


def connected_components(mask: np.ndarray) -> list[Component]:
    """Связные области (8-связность) булевой маски через отрезки строк и union-find."""
    h, w = mask.shape
    padded = np.zeros((h, w + 2), dtype=np.int8)
    padded[:, 1:-1] = mask
    diff = np.diff(padded, axis=1)
    sy, sx = np.nonzero(diff == 1)
    ey, ex = np.nonzero(diff == -1)
    n = len(sy)
    if n == 0:
        return []
    # обе выборки идут построчно слева направо, поэтому начала и концы совпадают по порядку
    ys, x0s, x1s = sy, sx, ex
    row_ptr = np.searchsorted(ys, np.arange(h + 1))
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    x0l, x1l = x0s.tolist(), x1s.tolist()
    rp = row_ptr.tolist()
    for y in range(1, h):
        i, i_end = rp[y - 1], rp[y]
        j, j_end = rp[y], rp[y + 1]
        while i < i_end and j < j_end:
            if x0l[i] <= x1l[j] and x0l[j] <= x1l[i]:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
            if x1l[i] < x1l[j]:
                i += 1
            else:
                j += 1
    roots = np.array([find(i) for i in range(n)])
    _, labels = np.unique(roots, return_inverse=True)
    k = labels.max() + 1
    xmin = np.full(k, w)
    np.minimum.at(xmin, labels, x0s)
    xmax = np.zeros(k, dtype=np.int64)
    np.maximum.at(xmax, labels, x1s)
    ymin = np.full(k, h)
    np.minimum.at(ymin, labels, ys)
    ymax = np.zeros(k, dtype=np.int64)
    np.maximum.at(ymax, labels, ys)
    area = np.bincount(labels, weights=(x1s - x0s), minlength=k).astype(np.int64)

    order = np.argsort(labels, kind='stable')
    bounds = np.searchsorted(labels[order], np.arange(k + 1))
    comps = []
    for c in range(k):
        idx = order[bounds[c]:bounds[c + 1]]
        runs = list(zip(ys[idx].tolist(), x0s[idx].tolist(), x1s[idx].tolist()))
        comps.append(Component(int(xmin[c]), int(ymin[c]), int(xmax[c] - xmin[c]),
                               int(ymax[c] - ymin[c] + 1), int(area[c]), runs))
    return comps


def _group_glyphs(glyphs):
    """Склеивает соседние цифры одной высоты в числа (не длиннее 3 знаков)."""
    glyphs = sorted(glyphs, key=lambda g: g[0])
    used = [False] * len(glyphs)
    groups = []
    for i, g in enumerate(glyphs):
        if used[i]:
            continue
        group = [g]
        used[i] = True
        changed = True
        while changed:
            changed = False
            gx1 = min(c[0] for c in group)
            gx2 = max(c[0] + c[2] for c in group)
            gy1 = min(c[1] for c in group)
            gy2 = max(c[1] + c[3] for c in group)
            gh = gy2 - gy1
            for j, c in enumerate(glyphs):
                if used[j]:
                    continue
                if not 0.65 < c[3] / gh < 1.5:
                    continue
                overlap = min(gy2, c[1] + c[3]) - max(gy1, c[1])
                if overlap < 0.6 * min(gh, c[3]):
                    continue
                gap = max(c[0] - gx2, gx1 - (c[0] + c[2]))
                if gap > 0.45 * gh:
                    continue
                group.append(c)
                used[j] = True
                changed = True
        if len(group) <= 3:
            groups.append(group)
    return groups


def _filter_by_height(dets: list[Detection]) -> list[Detection]:
    """Подписи на одном фото одного размера — убираем выбросы по высоте."""
    if len(dets) < 2:
        return dets
    weights = np.array([d.score for d in dets])
    heights = np.array([d.box[3] for d in dets], dtype=float)
    order = np.argsort(heights)
    cum = np.cumsum(weights[order])
    median_h = heights[order][np.searchsorted(cum, cum[-1] / 2.0)]
    return [d for d in dets if 0.7 * median_h <= d.box[3] <= 1.4 * median_h]


# --- распределение диапазонов по фото -------------------------------------

MISS_PENALTY = 0.3     # штраф за номер внутри диапазона, которого не нашли на фото
SMALL_GAP = 3          # пропуск до 3 номеров между фото — скорее всего, не распознались
BIG_GAP_COST = 0.5     # большой пропуск — вероятно, не добавлено фото
EPS = 0.01


def _gap_cost(gap: int) -> float:
    """Штраф за номера между диапазонами соседних фото.

    Маленький пропуск выгоднее «закрыть» соседним фото (номер просто не
    распознался). Большой — оставить как есть: тогда программа покажет,
    что каких-то фото не хватает, а не растянет диапазон на чужие номера.
    """
    if gap <= 0:
        return 0.0
    if gap <= SMALL_GAP:
        return gap * (MISS_PENALTY + EPS)
    return BIG_GAP_COST


def assign_ranges(numbers_per_photo: list[list[int]]) -> list[tuple[int, int] | None]:
    """Подбирает диапазоны номеров для каждого фото.

    numbers_per_photo — найденные на каждом фото числа (возможны пропуски и
    ложные срабатывания). Диапазоны идут по возрастанию и не пересекаются;
    номер, найденный «не на своём» фото, отбрасывается. Фото без чисел —
    общий вид (None). Возвращает для каждого фото (с, по) или None.
    """
    result: list[tuple[int, int] | None] = [None] * len(numbers_per_photo)
    photos = []
    for i, ns in enumerate(numbers_per_photo):
        clean = sorted({n for n in ns if 1 <= n <= MAX_NUMBER})
        if clean:
            photos.append((i, clean))
    if not photos:
        return result
    photos.sort(key=lambda p: (float(np.median(p[1])), p[0]))

    # dp: конец диапазона текущего фото -> (счёт, (конец предыдущего, с, по))
    history: list[dict[int, tuple[float, tuple[int, int, int] | None]]] = []
    dp: dict[int, tuple[float, tuple[int, int, int] | None]] = {0: (0.0, None)}
    for _, ns in photos:
        new: dict[int, tuple[float, tuple[int, int, int] | None]] = {}
        for prev_end, (score, _) in dp.items():
            los = {prev_end + 1} | {n for n in ns if n > prev_end}
            for lo in sorted(los):
                first = bisect.bisect_left(ns, lo)
                for hi in sorted({lo} | set(ns[first:])):
                    hits = bisect.bisect_right(ns, hi) - first
                    s = (score + hits - MISS_PENALTY * ((hi - lo + 1) - hits)
                         - _gap_cost(lo - prev_end - 1))
                    if hi not in new or s > new[hi][0] + 1e-9:
                        new[hi] = (s, (prev_end, lo, hi))
        if len(new) > 400:       # оставляем лучшие состояния
            new = dict(sorted(new.items(), key=lambda kv: -kv[1][0])[:400])
        history.append(new)
        dp = new

    end = max(dp, key=lambda hi: (dp[hi][0], -hi))
    for k in range(len(photos) - 1, -1, -1):
        prev_end, lo, hi = history[k][end][1]
        result[photos[k][0]] = (lo, hi)
        end = prev_end
    return result
