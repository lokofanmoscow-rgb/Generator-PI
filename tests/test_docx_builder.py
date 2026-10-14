"""Интеграционный тест сборки ПИ.

Нужен реальный шаблон ПИ: путь в переменной окружения GPI_SAMPLE_DOCX
(шаблоны в репозиторий не кладём). Фото берутся из самого шаблона.
"""
import os
import re
import zipfile

import pytest
from lxml import etree

from generator_pi import values as V
from generator_pi.docx_builder import (NS, ComponentSpec, PhotoSpec, TemplateError, build,
                                       find_table4, text_of, Template)

SAMPLE = os.environ.get('GPI_SAMPLE_DOCX')
pytestmark = pytest.mark.skipif(not SAMPLE or not os.path.exists(SAMPLE),
                                reason='GPI_SAMPLE_DOCX не задан')


@pytest.fixture(scope='module')
def photos(tmp_path_factory):
    d = tmp_path_factory.mktemp('photos')
    out = []
    with zipfile.ZipFile(SAMPLE) as z:
        media = sorted(n for n in z.namelist() if n.startswith('word/media/'))[:3]
        for n in media:
            p = d / os.path.basename(n)
            p.write_bytes(z.read(n))
            out.append(str(p))
    return out


def _components():
    cats = {2: V.METAL, 3: V.WIRE, 4: V.BOARD, 5: V.SMD}
    vals = {3: {V.BR: '118 ± 10'}, 4: {V.PB: '53 ± 8', V.BR: '139 ± 18'}, 5: {V.PB: '124±12', V.BR: '155 27'}}
    return [ComponentSpec(n, cats.get(n, V.POLYMER), vals.get(n, {})) for n in range(1, 7)]


def test_build(tmp_path, photos):
    out = tmp_path / 'out.docx'
    name = 'Ночник светодиодный, марка "Lumio", модель NL-15'
    specs = [PhotoSpec(photos[0]), PhotoSpec(photos[1], (1, 1)), PhotoSpec(photos[2], (2, 6))]
    rep = build(SAMPLE, str(out), name, specs, _components())
    assert rep.name_replaced and rep.count_replaced and not rep.warnings

    tpl = Template(str(out))
    body_text = '\n'.join(text_of(p) for p in tpl.body.iter('{%s}p' % NS['w']))
    assert 'Ночник светодиодный, марка "Lumio", модель NL-15' in body_text
    assert 'Количество компонентов составило 6 шт' in body_text
    assert 'Компоненты. Ночник светодиодный, марка «Lumio», модель NL-15' in body_text
    assert '2÷6 компоненты' in body_text and '1 компонент' in body_text
    assert rep.old_name and rep.old_name not in body_text

    tbl = find_table4(tpl)
    rows = tbl.findall('w:tr', NS)
    data = [r for r in rows if r.find('w:trPr/w:tblHeader', NS) is None]
    assert len(data) == 5 * 5 + 4          # один металл — 4 строки
    texts = [[text_of(c) for c in r.findall('w:tc', NS)] for r in data]
    assert ['', 'Общий бром', '155 ± 27', '0,0155 ± 0,0027', 'BL'] in texts
    assert ['', 'Общий бром', '118 ± 10', '0,0118 ± 0,0010', 'BL'] in texts

    with zipfile.ZipFile(out) as z:
        names = z.namelist()
        assert names[0] == '[Content_Types].xml'
        media = [n for n in names if n.startswith('word/media/')]
        assert len(media) == 3
        with zipfile.ZipFile(SAMPLE) as src:
            for n in src.namelist():          # колонтитулы и прочие части не тронуты
                if re.match(r'word/(header|footer)\d+\.xml', n) or n == 'word/styles.xml':
                    assert z.read(n) == src.read(n)
        for n in names:
            if n.endswith('.xml') or n.endswith('.rels'):
                etree.fromstring(z.read(n))


def test_missing_required_value(tmp_path, photos):
    comps = _components()
    comps[3].values = {V.PB: '53 ± 8'}             # у платы нет брома
    with pytest.raises(TemplateError, match='Общий бром'):
        build(SAMPLE, str(tmp_path / 'x.docx'), 'Образец', [PhotoSpec(photos[0])], comps)
