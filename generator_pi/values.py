"""Типы компонентов, разбор введённых значений и интерпретация результатов.

Значения вводит пользователь. Модуль только приводит их к виду, принятому
в Таблице №4 («120 ± 12» / «0,0120 ± 0,0012»), и определяет интерпретацию
(BL / X / OL) по пределам скрининга ГОСТ IEC 62321-3-1-2016, Приложение А.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

# --- элементы -------------------------------------------------------------

PB, HG, CD, CR, BR = 'Pb', 'Hg', 'Cd', 'Cr', 'Br'
ALL_ELEMENTS = (PB, HG, CD, CR, BR)

ELEMENT_TITLES = {
    PB: 'Свинец (Pb)',
    HG: 'Ртуть (Hg)',
    CD: 'Кадмий (Cd)',
    CR: 'Общий хром',
    BR: 'Общий бром',
}

# --- типы компонентов -----------------------------------------------------

POLYMER, METAL, WIRE, BOARD, SMD = 'polymer', 'metal', 'wire', 'board', 'smd'
CATEGORIES = (POLYMER, METAL, WIRE, BOARD, SMD)

CATEGORY_TITLES = {
    POLYMER: 'Чистый полимер',
    METAL: 'Чистый металл',
    WIRE: 'Провод',
    BOARD: 'Плата',
    SMD: 'СМД компонент',
}

# Цвета квадратиков (полимер — без выделения).
CATEGORY_COLORS = {
    POLYMER: '#FFFFFF',
    METAL: '#9AA5B1',
    WIRE: '#F5A623',
    BOARD: '#4CAF50',
    SMD: '#B05CD6',
}

# Какие элементы попадают в таблицу для каждого типа.
CATEGORY_ELEMENTS = {
    POLYMER: ALL_ELEMENTS,
    METAL: (PB, HG, CD, CR),  # без брома
    WIRE: ALL_ELEMENTS,
    BOARD: ALL_ELEMENTS,
    SMD: ALL_ELEMENTS,
}

# Где программа требует ввести значение вручную.
REQUIRED_ELEMENTS = {
    POLYMER: (),
    METAL: (),
    WIRE: (BR,),
    BOARD: (PB, BR),
    SMD: (PB, BR),
}

# Матрица для интерпретации по Таблице №5.
MATRIX_POLYMER, MATRIX_METAL, MATRIX_COMPOSITE = 'polymer', 'metal', 'composite'
CATEGORY_MATRIX = {
    POLYMER: MATRIX_POLYMER,
    METAL: MATRIX_METAL,
    WIRE: MATRIX_COMPOSITE,
    BOARD: MATRIX_COMPOSITE,
    SMD: MATRIX_COMPOSITE,
}

# Пределы скрининга, мг/кг: (нижний, верхний); None — граница не задана.
# BL ≤ (нижний − 3σ) < X < (верхний + 3σ) ≤ OL
SCREENING_LIMITS = {
    MATRIX_POLYMER: {CD: (70, 130), PB: (700, 1300), HG: (700, 1300), BR: (300, None), CR: (700, None)},
    MATRIX_METAL: {CD: (70, 130), PB: (700, 1300), HG: (700, 1300), BR: None, CR: (700, None)},
    # Для кадмия в композитах BL не предусмотрено: LOD < X < (150 + 3σ) ≤ OL.
    MATRIX_COMPOSITE: {CD: (None, 150), PB: (500, 1500), HG: (500, 1500), BR: (250, None), CR: (500, None)},
}

DEFAULT_TEXT = 'Менее 10'


class ValueError_(ValueError):
    """Неверный формат введённого значения."""


@dataclass(frozen=True)
class Result:
    """Результат по одному элементу одного компонента."""

    below: bool              # «Менее X»
    value: Decimal           # X для «Менее X», иначе измеренное значение
    uncertainty: Decimal | None

    @property
    def mg_text(self) -> str:
        if self.below:
            return f'Менее {_fmt(self.value)}'
        if self.uncertainty is None:
            return _fmt(self.value)
        return f'{_fmt(self.value)} ± {_fmt(self.uncertainty)}'

    @property
    def percent_text(self) -> str:
        if self.below:
            return f'Менее {_fmt_percent(self.value, minimal=True)}'
        if self.uncertainty is None:
            return _fmt_percent(self.value)
        return f'{_fmt_percent(self.value)} ± {_fmt_percent(self.uncertainty)}'


_NUM = r'(\d+(?:[.,]\d+)?)'
_BELOW_RE = re.compile(r'^(?:менее|меньше|<|&lt;)\s*' + _NUM + r'$', re.IGNORECASE)
_PM_RE = re.compile(r'^' + _NUM + r'\s*(?:±|\+-|\+/-|\+\\-|\s|/)\s*' + _NUM + r'$')
_SINGLE_RE = re.compile(r'^' + _NUM + r'$')


def _dec(s: str) -> Decimal:
    try:
        return Decimal(s.replace(',', '.'))
    except InvalidOperation as e:  # pragma: no cover - regex guarantees digits
        raise ValueError_(s) from e


def _fmt(d: Decimal) -> str:
    s = format(d.normalize(), 'f') if d == d.to_integral() else format(d, 'f')
    if '.' in s:
        s = s.rstrip('0').rstrip('.')
    return s.replace('.', ',')


def _fmt_percent(mg: Decimal, minimal: bool = False) -> str:
    """мг/кг → % (1 % = 10 000 мг/кг).

    Обычные значения — 4 знака после запятой, как в образце (120 → 0,0120).
    Для «Менее X» — без лишних нулей (10 → 0,001).
    """
    pct = mg / Decimal(10000)
    if minimal:
        s = format(pct.normalize(), 'f')
    else:
        exp = -mg.as_tuple().exponent if mg.as_tuple().exponent < 0 else 0
        s = format(pct.quantize(Decimal(1).scaleb(-(4 + exp))), 'f')
    return s.replace('.', ',')


def parse(text: str) -> Result:
    """Разбирает ввод пользователя.

    Поддерживается: «Менее 10», «<10», «120 ± 12», «120±12», «120 +- 12»,
    «120 12», «120/12», «120».
    """
    t = (text or '').strip().replace(' ', ' ')
    t = re.sub(r'\s+', ' ', t)
    if not t:
        raise ValueError_('пустое значение')
    m = _BELOW_RE.match(t)
    if m:
        return Result(True, _dec(m.group(1)), None)
    m = _PM_RE.match(t)
    if m:
        return Result(False, _dec(m.group(1)), _dec(m.group(2)))
    m = _SINGLE_RE.match(t)
    if m:
        return Result(False, _dec(m.group(1)), None)
    raise ValueError_(f'не понимаю значение «{text}»')


def interpret(result: Result, element: str, category: str) -> str:
    """BL / X / OL по Таблице №5 (3σ берётся равным указанной погрешности)."""
    limits = SCREENING_LIMITS[CATEGORY_MATRIX[category]].get(element)
    if result.below or limits is None:
        return 'BL'
    low, high = limits
    sigma3 = result.uncertainty or Decimal(0)
    v = result.value
    if high is not None and v >= high + sigma3:
        return 'OL'
    if low is None:
        return 'X'
    if v <= low - sigma3:
        return 'BL'
    return 'X'


def elements_for(category: str) -> tuple[str, ...]:
    return CATEGORY_ELEMENTS[category]


def required_for(category: str) -> tuple[str, ...]:
    return REQUIRED_ELEMENTS[category]
