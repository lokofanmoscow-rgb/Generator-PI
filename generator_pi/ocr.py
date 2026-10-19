"""Распознавание номеров компонентов на фотографиях.

Номера на фото — чёрные цифры, нанесённые поверх снимка (с выносными линиями).
Алгоритм (на фото в полном разрешении, до 4096 px — мелкие подписи нельзя уменьшать):
  1. Маска «чернил»: пиксели, у которых все каналы тёмные.
  2. Кандидаты на подписи из двух источников:
     а) связные области, похожие по форме на цифры, склеенные в числа;
     б) нейросетевой детектор текста PP-OCRv3 det по маске — находит числа,
        к которым вплотную подходит выносная линия.
  3. Каждый кандидат распознаётся моделью PP-OCRv3 rec с декодированием только
     по цифрам; неуверенная «лишняя» цифра с края (кусок линии) отбрасывается.
  4. Из пересекающихся находок остаётся лучшая; отсев по «чистоте» чёрного
     и по высоте цифр (все подписи на одном фото одного размера).

Затем assign_ranges() по всем фото сразу подбирает диапазоны номеров
(1…a, a+1…b, …), которые лучше всего согласуются с найденными числами.
"""
from __future__ import annotations

import bisect
import os
import sys
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageOps

INK_THRESHOLD = 90        # пиксель «чёрный», если все каналы < порога
MAX_SIDE = 4096           # больше — уменьшаем
DET_SIDE = 3072           # размер входа детектора текста
MIN_GLYPH_PX = 8          # минимальная высота цифры, px
MAX_GLYPH_REL = 0.12      # максимальная высота цифры от большей стороны фото
LABEL_DARKNESS_MAX = 50   # яркость цифр подписи (25-й процентиль) не выше…
LABEL_DARKNESS_MAX_BIG = 35   # …а у цифр от 24 px — строже: их края JPEG почти не высветляет
DARK_PERCENTILE = 25
MIN_SCORE = 0.5           # средняя уверенность по цифрам числа
MIN_CHAR_SCORE = 0.3      # цифра с краю с меньшей уверенностью — кусок линии
MAX_NUMBER = 999


@dataclass
class Detection:
    number: int
    score: float
    box: tuple[int, int, int, int]   # x, y, w, h (в координатах анализа)
    darkness: float
    char_scores: list[float] = field(default_factory=list)
    height: float = 0.0              # высота цифр

    @property
    def min_char(self) -> float:
        return min(self.char_scores) if self.char_scores else self.score


def resource_path(*parts: str) -> str:
    base = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    if getattr(sys, '_MEIPASS', None):
        base = os.path.join(base, 'generator_pi')
    return os.path.join(base, *parts)


def load_image(path: str) -> Image.Image:
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)
    return im.convert('RGB')


_DET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_DET_STD = np.array([0.229, 0.224, 0.225], np.float32)


class NumberReader:
    def __init__(self, model_path: str | None = None, det_model_path: str | None = None):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        model_path = model_path or resource_path('models', 'digits_rec.onnx')
        self.session = ort.InferenceSession(model_path, sess_options=opts,
                                            providers=['CPUExecutionProvider'])
        chars = self.session.get_modelmeta().custom_metadata_map['character'].splitlines()
        chars = ['<blank>'] + chars + [' ']
        self.digit_idx = np.array([0] + [chars.index(str(d)) for d in range(10)])
        self.input_name = self.session.get_inputs()[0].name

        det_model_path = det_model_path or resource_path('models', 'text_det.onnx')
        self.det = None
        if os.path.exists(det_model_path):
            self.det = ort.InferenceSession(det_model_path, sess_options=opts,
                                            providers=['CPUExecutionProvider'])

    # -- распознавание одного фрагмента ------------------------------------
    def _recognize(self, crop: np.ndarray) -> tuple[str, list[float]]:
        """crop: uint8 HxW (0 — чернила, 255 — фон). Возвращает (цифры, уверенность по цифрам)."""
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
        # неуверенная цифра с края — обычно конец выносной линии
        while len(text) > 1 and conf[0] < MIN_CHAR_SCORE:
            text, conf = text[1:], conf[1:]
        while len(text) > 1 and conf[-1] < MIN_CHAR_SCORE:
            text, conf = text[:-1], conf[:-1]
        return text, conf

    # -- кандидаты ----------------------------------------------------------
    def _glyph_boxes(self, ink: np.ndarray, brightness: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Числа из связных областей, похожих на цифры."""
        side = max(ink.shape)
        glyphs, rejected = [], []
        for comp in connected_components(ink):
            x, y, w, h, area = comp.x, comp.y, comp.w, comp.h, comp.area
            if h < MIN_GLYPH_PX and w < MIN_GLYPH_PX:
                continue
            fill = area / float(w * h)
            if (MIN_GLYPH_PX <= h <= MAX_GLYPH_REL * side and 0.12 * h <= w <= 1.1 * h
                    and 0.12 <= fill <= 0.8):
                if _dark_enough(brightness, comp.runs, h):
                    glyphs.append(((x, y, w, h), comp))
            elif max(w, h) <= 0.5 * side:
                rejected.append(comp)      # может быть цифрой, к которой прилипла линия
        if len(glyphs) >= 3:
            # цифра со сросшейся вертикальной линией намного выше остальных
            typical = float(np.median([g[0][3] for g in glyphs]))
            rejected += [c for g, c in glyphs if g[3] > 1.6 * typical]
            glyphs = [(g, c) for g, c in glyphs if g[3] <= 1.6 * typical]
        glyphs = [g for g, _ in glyphs]
        glyphs += _peel_touching(glyphs, rejected, brightness)
        boxes = []
        for group in _group_glyphs(glyphs):
            x1 = min(g[0] for g in group)
            y1 = min(g[1] for g in group)
            x2 = max(g[0] + g[2] for g in group)
            y2 = max(g[1] + g[3] for g in group)
            pad = int(0.2 * (y2 - y1)) + 2
            boxes.append((x1 - pad, y1 - pad, x2 + pad, y2 + pad))
        return boxes

    def _text_boxes(self, ink: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Области текста по нейросетевому детектору (DB, PP-OCRv3 det)."""
        if self.det is None:
            return []
        h, w = ink.shape
        s = min(1.0, DET_SIDE / max(h, w))
        nh = max(32, int(round(h * s / 32)) * 32)
        nw = max(32, int(round(w * s / 32)) * 32)
        img = Image.fromarray(np.where(ink, 0, 255).astype(np.uint8)).resize((nw, nh), Image.BILINEAR)
        gray = np.asarray(img, np.float32) / 255.0
        x = ((gray[:, :, None] - _DET_MEAN) / _DET_STD).transpose(2, 0, 1)[None].astype(np.float32)
        pred = self.det.run(None, {self.det.get_inputs()[0].name: x})[0][0, 0]
        binm = pred > 0.3
        binm[:-1, :] |= binm[1:, :]          # расширение 2x2, как в PaddleOCR
        binm[:, :-1] |= binm[:, 1:]
        boxes = []
        for c in connected_components(binm):
            if c.w < 3 or c.h < 3:
                continue
            score = float(np.mean(np.concatenate([pred[ry, a:b] for ry, a, b in c.runs])))
            if score < 0.5:
                continue
            d = c.w * c.h * 1.6 / (2 * (c.w + c.h))   # «unclip» прямоугольника
            boxes.append((int((c.x - d) * w / nw), int((c.y - d) * h / nh),
                          int((c.x + c.w + d) * w / nw) + 1, int((c.y + c.h + d) * h / nh) + 1))
        return boxes

    # -- поиск подписей ----------------------------------------------------
    def read(self, path: str) -> list[Detection]:
        return self.read_full(path)[0]

    def read_full(self, path: str) -> tuple[list[Detection], tuple[int, int]]:
        """Найденные подписи и размер (ширина, высота) кадра, в котором даны их рамки."""
        im = load_image(path)
        im.thumbnail((MAX_SIDE, MAX_SIDE))
        a = np.asarray(im)
        brightness = a.max(axis=2)
        ink = brightness < INK_THRESHOLD
        H, W = ink.shape
        rec_src = np.where(ink, 0, 255).astype(np.uint8)

        dets: list[Detection] = []
        seen = set()
        for box in self._glyph_boxes(ink, brightness) + self._text_boxes(ink):
            x1, y1 = max(0, box[0]), max(0, box[1])
            x2, y2 = min(W, box[2]), min(H, box[3])
            if y2 - y1 < MIN_GLYPH_PX or x2 - x1 < 4 or (x1, y1, x2, y2) in seen:
                continue
            seen.add((x1, y1, x2, y2))
            if not ink[y1:y2, x1:x2].any():
                continue
            x1, y1, x2, y2 = _trim_lines(ink, x1, y1, x2, y2)
            sub_ink = ink[y1:y2, x1:x2]
            if y2 - y1 < MIN_GLYPH_PX or not sub_ink.any():
                continue
            dark = float(np.percentile(brightness[y1:y2, x1:x2][sub_ink], DARK_PERCENTILE))
            text_h = _text_height(sub_ink)
            if dark > _dark_limit(text_h):
                continue
            text, conf = self._recognize(rec_src[y1:y2, x1:x2])
            if not text or len(text) > 3 or float(np.mean(conf)) < MIN_SCORE:
                continue
            n = int(text)
            if not 1 <= n <= MAX_NUMBER:
                continue
            dets.append(Detection(n, float(np.mean(conf)), (x1, y1, x2 - x1, y2 - y1), dark,
                                  conf, text_h))
        return _filter_by_height(_dedupe(dets)), (W, H)

    def read_numbers(self, path: str) -> list[int]:
        return sorted({d.number for d in self.read(path)})


def _trim_lines(ink: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> tuple[int, int, int, int]:
    """Срезает с краёв рамки выносную линию, чтобы её конец не прочитался как «1».

    По высоте — оставляем полосу цифр (строки, где чернил заметно больше, чем у
    тонкой линии). По ширине — с каждого края убираем длинный (больше 0,6 высоты
    цифр) участок тонких столбцов; короткий тонкий участок — это часть цифры
    (например, перекладина «7»), его не трогаем.
    """
    sub = ink[y1:y2, x1:x2]
    rows = sub.sum(axis=1)
    if rows.max() == 0:
        return x1, y1, x2, y2
    # 80-й процентиль, а не максимум: строки горизонтальной линии не должны задавать полосу
    band = np.nonzero(rows >= 0.4 * np.percentile(rows[rows > 0], 80))[0]
    th = float(band[-1] - band[0] + 1)
    pad = int(round(0.25 * th)) + 1
    ny1, ny2 = max(0, band[0] - pad), min(sub.shape[0], band[-1] + 1 + pad)
    cols = sub[ny1:ny2].sum(axis=0)
    thin = max(2.0, 0.2 * th)
    nz = np.nonzero(cols)[0]
    if len(nz) == 0:
        return x1, y1, x2, y2
    c0, c1 = int(nz[0]), int(nz[-1])
    left = c0
    while left <= c1 and cols[left] <= thin:
        left += 1
    right = c1
    while right >= c0 and cols[right] <= thin:
        right -= 1
    if left > right:                      # одни тонкие столбцы — это линия, не число
        return x1, y1, x1, y1
    nx1 = left - pad if left - c0 >= 0.6 * th else 0
    nx2 = right + 1 + pad if c1 - right >= 0.6 * th else sub.shape[1]
    nx1, nx2 = max(0, nx1), min(sub.shape[1], nx2)
    return x1 + nx1, y1 + ny1, x1 + nx2, y1 + ny2


def _text_height(sub_ink: np.ndarray) -> float:
    """Высота цифр в рамке: строки, где чернил заметно больше, чем у тонкой линии."""
    rows = sub_ink.sum(axis=1)
    if rows.max() == 0:
        return 0.0
    idx = np.nonzero(rows >= 0.4 * np.percentile(rows[rows > 0], 80))[0]
    return float(idx[-1] - idx[0] + 1)


def _overlap(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = min(ax + aw, bx + bw) - max(ax, bx)
    ih = min(ay + ah, by + bh) - max(ay, by)
    if iw <= 0 or ih <= 0:
        return 0.0
    return iw * ih / float(min(aw * ah, bw * bh))


def _priority(d: Detection) -> tuple:
    """Порядок отбора: сначала уверенно прочитанные длинные числа, затем по уверенности.

    Обрывок числа («5» от «105», к средней цифре которого подходит линия) читается
    увереннее целого, поэтому длина важнее средней уверенности."""
    digits = len(str(d.number)) if d.min_char >= 0.35 else 1
    return (-digits, -d.score)


def _dedupe(dets: list[Detection]) -> list[Detection]:
    """Из находок на одном месте (разные кандидаты одной подписи) оставляем лучшую."""
    kept: list[Detection] = []
    for d in sorted(dets, key=_priority):
        if all(_overlap(d.box, k.box) <= 0.5 for k in kept):
            kept.append(d)
    return kept


class Component:
    """Связная область: рамка, площадь и отрезки строк (y, x0, x1), x1 не включительно.

    Отрезки собираются лениво — их нужно лишь немногим областям."""

    __slots__ = ('x', 'y', 'w', 'h', 'area', '_table', '_idx')

    def __init__(self, x, y, w, h, area, table, idx):
        self.x, self.y, self.w, self.h, self.area = x, y, w, h, area
        self._table, self._idx = table, idx

    @property
    def runs(self) -> list[tuple[int, int, int]]:
        ys, x0s, x1s = self._table
        i = self._idx
        return list(zip(ys[i].tolist(), x0s[i].tolist(), x1s[i].tolist()))


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
    bounds = np.searchsorted(labels[order], np.arange(k + 1)).tolist()
    table = (ys, x0s, x1s)
    xmin, xmax, ymin, ymax, area = (v.tolist() for v in (xmin, xmax, ymin, ymax, area))
    return [Component(xmin[c], ymin[c], xmax[c] - xmin[c], ymax[c] - ymin[c] + 1, area[c],
                      table, order[bounds[c]:bounds[c + 1]]) for c in range(k)]


def _dark_limit(height: float) -> float:
    return LABEL_DARKNESS_MAX_BIG if height >= 24 else LABEL_DARKNESS_MAX


def _dark_enough(brightness: np.ndarray, runs, height: float) -> bool:
    pixels = np.concatenate([brightness[ry, rx0:rx1] for ry, rx0, rx1 in runs])
    return float(np.percentile(pixels, DARK_PERCENTILE)) <= _dark_limit(height)


def _peel_touching(glyphs, rejected, brightness) -> list[tuple[int, int, int, int]]:
    """Цифра, которой касается выносная линия, сливается с ней в одно пятно.

    Рядом с уже найденными цифрами (слева и справа, в полосе их высоты) берём
    такие пятна, срезаем с дальнего края тонкие столбцы (линию) и, если остаток
    похож на цифру той же высоты, добавляем его как цифру.
    """
    if not glyphs or not rejected:
        return []
    bx0 = np.array([c.x for c in rejected])
    by0 = np.array([c.y for c in rejected])
    bx1 = bx0 + np.array([c.w for c in rejected])
    by1 = by0 + np.array([c.h for c in rejected])
    found = set()
    for gx, gy, gw, gh in glyphs:
        y0, y1 = gy - int(0.1 * gh), gy + gh + int(0.1 * gh) + 1
        reach = int(1.3 * gh)
        for side in (-1, 1):
            wx0, wx1 = (gx + gw, gx + gw + reach) if side > 0 else (gx - reach, gx)
            hit = np.nonzero((bx0 < wx1) & (bx1 > wx0) & (by0 < y1) & (by1 > y0))[0]
            for i in hit.tolist():
                margin = 0.3 * gh
                if by0[i] < y0 - margin and by1[i] > y1 + margin:
                    continue                # линия проходит полосу цифр насквозь — не цифра
                mask = np.zeros((y1 - y0, wx1 - wx0), bool)
                for ry, a, b in rejected[i].runs:
                    if y0 <= ry < y1:
                        a, b = max(a, wx0), min(b, wx1)
                        if a < b:
                            mask[ry - y0, a - wx0:b - wx0] = True
                cols = mask.sum(axis=0)
                nz = np.nonzero(cols)[0]
                if len(nz) == 0:
                    continue
                c0, c1 = int(nz[0]), int(nz[-1])
                thin = max(2.0, 0.22 * gh)
                if side > 0:
                    while c1 > c0 and cols[c1] <= thin:
                        c1 -= 1
                else:
                    while c0 < c1 and cols[c0] <= thin:
                        c0 += 1
                rows = np.nonzero(mask[:, c0:c1 + 1].any(axis=1))[0]
                if len(rows) == 0:
                    continue
                r0, r1 = int(rows[0]), int(rows[-1])
                w2, h2 = c1 - c0 + 1, r1 - r0 + 1
                if not (0.6 * gh <= h2 <= 1.3 * gh and 0.25 * gh <= w2 <= 1.1 * gh):
                    continue
                pix = brightness[y0:y1, wx0:wx1][mask]
                if float(np.percentile(pix, DARK_PERCENTILE)) > _dark_limit(h2):
                    continue
                found.add((wx0 + c0, y0 + r0, w2, h2))
    known = set(glyphs)
    return [g for g in found if g not in known]


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
    """Подписи на одном фото одного размера — убираем выбросы по высоте цифр."""
    good = [d for d in dets if d.height > 0]
    if len(good) < 3:
        return dets
    weights = np.array([d.score for d in good])
    heights = np.array([d.height for d in good], dtype=float)
    order = np.argsort(heights)
    cum = np.cumsum(weights[order])
    median_h = heights[order][np.searchsorted(cum, cum[-1] / 2.0)]
    return [d for d in dets if 0.6 * median_h <= d.height <= 1.6 * median_h]


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
