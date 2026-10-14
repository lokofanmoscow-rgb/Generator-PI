import pytest

from generator_pi import values as V
from generator_pi.ocr import assign_ranges


@pytest.mark.parametrize('text,mg,pct', [
    ('Менее 10', 'Менее 10', 'Менее 0,001'),
    ('<10', 'Менее 10', 'Менее 0,001'),
    ('менее 5', 'Менее 5', 'Менее 0,0005'),
    ('120 ± 12', '120 ± 12', '0,0120 ± 0,0012'),
    ('120±12', '120 ± 12', '0,0120 ± 0,0012'),
    ('53 +- 8', '53 ± 8', '0,0053 ± 0,0008'),
    ('151 20', '151 ± 20', '0,0151 ± 0,0020'),
    ('151/20', '151 ± 20', '0,0151 ± 0,0020'),
    ('130', '130', '0,0130'),
    ('12,5 ± 1,5', '12,5 ± 1,5', '0,00125 ± 0,00015'),
])
def test_parse_and_format(text, mg, pct):
    r = V.parse(text)
    assert r.mg_text == mg
    assert r.percent_text == pct


@pytest.mark.parametrize('text', ['', 'abc', '12 ± ', '± 3'])
def test_parse_rejects_garbage(text):
    with pytest.raises(ValueError):
        V.parse(text)


def test_interpretation():
    assert V.interpret(V.parse('Менее 10'), V.CD, V.SMD) == 'BL'
    assert V.interpret(V.parse('135 ± 21'), V.PB, V.SMD) == 'BL'
    assert V.interpret(V.parse('160 ± 30'), V.BR, V.BOARD) == 'BL'
    assert V.interpret(V.parse('240 ± 20'), V.BR, V.BOARD) == 'X'
    assert V.interpret(V.parse('1600 ± 50'), V.PB, V.SMD) == 'OL'
    assert V.interpret(V.parse('20 ± 3'), V.CD, V.SMD) == 'X'      # композит: LOD < X
    assert V.interpret(V.parse('20 ± 3'), V.CD, V.POLYMER) == 'BL'


def test_categories():
    assert V.elements_for(V.METAL) == (V.PB, V.HG, V.CD, V.CR)
    assert len(V.elements_for(V.POLYMER)) == 5
    assert V.required_for(V.WIRE) == (V.BR,)
    assert V.required_for(V.BOARD) == (V.PB, V.BR)
    assert V.required_for(V.SMD) == (V.PB, V.BR)
    assert V.required_for(V.POLYMER) == ()


SAMPLE = [[], [1], [2], [3], [4, 5, 6, 7], [8, 9, 10, 11, 12, 13],
          list(range(14, 22)), list(range(22, 42)), list(range(42, 48)), list(range(48, 54))]
EXPECTED = [None, (1, 1), (2, 2), (3, 3), (4, 7), (8, 13), (14, 21), (22, 41), (42, 47), (48, 53)]


def test_assign_ranges_exact():
    assert assign_ranges(SAMPLE) == EXPECTED


def test_assign_ranges_any_order():
    order = [3, 9, 0, 5, 1, 8, 2, 7, 4, 6]
    got = assign_ranges([SAMPLE[i] for i in order])
    assert got == [EXPECTED[i] for i in order]


def test_assign_ranges_with_misses_and_noise():
    noisy = [[], [1], [2], [3], [1, 4, 6, 7], [9, 10, 11, 12, 13],
             [14, 15, 17, 18, 20, 21], list(range(22, 42)), [42, 43, 45, 47], [48, 49, 50, 51, 53, 103]]
    got = assign_ranges(noisy)
    assert got[0] is None
    assert got[1:4] == [(1, 1), (2, 2), (3, 3)]
    assert got[4][0] == 4 and got[5][1] == 13       # 8 мог уйти в любое из соседних фото
    assert got[6:] == [(14, 21), (22, 41), (42, 47), (48, 53)]
    # диапазоны непрерывны
    rs = [r for r in got if r]
    assert all(b[0] == a[1] + 1 for a, b in zip(rs, rs[1:]))


def test_assign_ranges_missing_photos_keep_detected_ranges():
    got = assign_ranges([list(range(14, 22)), [1], [], list(range(48, 54)), [4, 5, 6, 7]])
    assert got == [(14, 21), (1, 1), None, (48, 53), (2, 7)]


def test_assign_ranges_small_gap_at_start_is_closed():
    assert assign_ranges([[2, 3], [4, 5, 6]]) == [(1, 3), (4, 6)]


def test_assign_ranges_manual_range_constrains_neighbours():
    # исправленный вручную диапазон программа передаёт как полный список номеров
    got = assign_ranges([list(range(1, 11)), [12, 13, 14], [15, 16, 18]])
    assert got == [(1, 10), (11, 14), (15, 18)]


def test_assign_ranges_noise_beyond_last_photo_is_dropped():
    assert assign_ranges([[1, 2, 3], [4, 5, 7, 103]]) == [(1, 3), (4, 7)]
