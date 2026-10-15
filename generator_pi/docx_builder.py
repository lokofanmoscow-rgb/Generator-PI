"""Сборка нового ПИ на основе шаблона (.docx).

Меняется только:
  * наименование образца в п. 3.1;
  * количество компонентов в п. 5 («Количество компонентов составило N шт.»);
  * Таблица №4 (строится заново по компонентам и введённым значениям);
  * Приложение №1 (фото и подписи).
Всё остальное в документе остаётся байт-в-байт как было (остальные части
пакета не перезаписываются).
"""
from __future__ import annotations

import copy
import io
import posixpath
import re
import zipfile
from dataclasses import dataclass, field

from lxml import etree
from PIL import Image, ImageOps

from . import values as V

W_NS = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
R_NS = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
WP_NS = 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'
A_NS = 'http://schemas.openxmlformats.org/drawingml/2006/main'
PIC_NS = 'http://schemas.openxmlformats.org/drawingml/2006/picture'
PKG_REL_NS = 'http://schemas.openxmlformats.org/package/2006/relationships'
CT_NS = 'http://schemas.openxmlformats.org/package/2006/content-types'
XML_NS = 'http://www.w3.org/XML/1998/namespace'

IMAGE_REL = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships/image'

NS = {'w': W_NS, 'r': R_NS, 'wp': WP_NS, 'a': A_NS, 'pic': PIC_NS}

EMU_PER_TWIP = 635
EMU_PER_CM = 360000
PHOTO_MAX_HEIGHT_CM = 8.5   # два фото с подписями на страницу A4
PHOTO_MAX_PIXELS = 2400


def w(tag: str) -> str:
    return f'{{{W_NS}}}{tag}'


class TemplateError(Exception):
    """Шаблон ПИ не похож на ожидаемый."""


@dataclass
class PhotoSpec:
    path: str
    range: tuple[int, int] | None = None   # None — общий вид образца


@dataclass
class ComponentSpec:
    number: int
    category: str = V.POLYMER
    values: dict[str, str] = field(default_factory=dict)   # элемент -> ввод


@dataclass
class BuildReport:
    old_name: str | None = None
    name_replaced: bool = False
    count_replaced: bool = False
    warnings: list[str] = field(default_factory=list)


# --- работа с текстом -----------------------------------------------------

def text_of(el) -> str:
    return ''.join(t.text or '' for t in el.iter(w('t')))


def _set_t(t_el, text: str) -> None:
    t_el.text = text
    if text != text.strip() or '  ' in text:
        t_el.set(f'{{{XML_NS}}}space', 'preserve')


def replace_in_paragraph(p, pattern: re.Pattern, repl) -> bool:
    """Заменяет первое совпадение pattern в тексте абзаца, даже если текст
    разбит на несколько w:t. Форматирование берётся от первого затронутого куска."""
    ts = list(p.iter(w('t')))
    full = ''.join(t.text or '' for t in ts)
    m = pattern.search(full)
    if not m:
        return False
    new = repl(m) if callable(repl) else m.expand(repl)
    start, end = m.span()
    pos = 0
    placed = False
    for t in ts:
        s = t.text or ''
        a, b = pos, pos + len(s)
        pos = b
        if b <= start or a >= end:
            continue
        left = s[:start - a] if start > a else ''
        right = s[end - a:] if end < b else ''
        if not placed:
            _set_t(t, left + new + right)
            placed = True
        else:
            _set_t(t, right)
    return placed


def _flex_name_regex(name: str) -> re.Pattern:
    """Регулярка для поиска наименования без учёта вида кавычек и пробелов."""
    parts = []
    for ch in name.strip():
        if ch in '"«»“”„\'':
            parts.append('["«»“”„\']')
        elif ch.isspace():
            parts.append(r'\s+')
        else:
            parts.append(re.escape(ch))
    return re.compile(''.join(parts))


def clean_name(name: str) -> str:
    """Наименование для п. 3.1: без точки в конце — после него в тексте идёт своя точка."""
    return re.sub(r'\s+', ' ', name).strip().rstrip('.').strip()


def appendix_name(name: str) -> str:
    """В подписях к фото — как введено, но кавычки «ёлочки», как в образце."""
    name = re.sub(r'"([^"]*)"', '«\\1»', re.sub(r'\s+', ' ', name).strip())
    name = re.sub(r'“([^”]*)”', '«\\1»', name)
    return name


# --- документ -------------------------------------------------------------

class Template:
    def __init__(self, path: str):
        with zipfile.ZipFile(path) as z:
            self.infos = z.infolist()
            self.parts = {i.filename: z.read(i.filename) for i in self.infos}
        if 'word/document.xml' not in self.parts:
            raise TemplateError('Это не документ Word (.docx).')
        self.doc = etree.fromstring(self.parts['word/document.xml'])
        self.body = self.doc.find('w:body', NS)
        self.rels = etree.fromstring(self.parts['word/_rels/document.xml.rels'])
        self.ctypes = etree.fromstring(self.parts['[Content_Types].xml'])

    # -- поиск ------------------------------------------------------------
    def body_children(self):
        return list(self.body)

    def find_paragraph(self, regex: str, start: int = 0, last: bool = False):
        rx = re.compile(regex)
        found = None
        for i, el in enumerate(self.body_children()):
            if i < start or el.tag != w('p'):
                continue
            if rx.search(text_of(el).strip()):
                if not last:
                    return i, el
                found = (i, el)
        return found if found else (None, None)

    # -- сохранение -------------------------------------------------------
    def save(self, out_path: str) -> None:
        self.parts['word/document.xml'] = _ser(self.doc)
        self.parts['word/_rels/document.xml.rels'] = _ser(self.rels)
        self.parts['[Content_Types].xml'] = _ser(self.ctypes)
        names_in_order = [i.filename for i in self.infos if i.filename in self.parts]
        names_in_order += [n for n in self.parts if n not in names_in_order]
        with zipfile.ZipFile(out_path, 'w', zipfile.ZIP_DEFLATED) as z:
            # [Content_Types].xml должен идти первым
            for n in sorted(names_in_order, key=lambda n: n != '[Content_Types].xml'):
                z.writestr(n, self.parts[n])


def _ser(root) -> bytes:
    return etree.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)


# --- п. 3.1 и п. 5 ----------------------------------------------------------

def detect_old_name(tpl: Template) -> str | None:
    idx, _ = tpl.find_paragraph(r'^Приложение\s*№\s*1\b', last=True)
    if idx is None:
        return None
    region = tpl.body_children()[idx + 1:]
    for el in region:
        for p in el.iter(w('p')):
            t = text_of(p).strip()
            m = re.match(r'^Компонент(?:ы)?\.\s*(.+)$', t)
            if m:
                return m.group(1).strip().rstrip('.').strip()
    # иначе — подпись под «Фото №1»
    after_photo1 = False
    for el in region:
        for p in el.iter(w('p')):
            t = text_of(p).strip()
            if re.match(r'^Фото\s*№\s*1$', t):
                after_photo1 = True
                continue
            if after_photo1 and t:
                return t.rstrip('.').strip()
    return None


def replace_name(tpl: Template, new_name: str, report: BuildReport) -> None:
    idx, _ = tpl.find_paragraph(r'^Наименование образца испытаний')
    if idx is None:
        report.warnings.append('Не найден пункт «Наименование образца испытаний» — наименование в п. 3.1 не заменено.')
        return
    candidates = [el for el in tpl.body_children()[idx + 1: idx + 6] if el.tag == w('p')]
    old = report.old_name
    if old:
        rx = _flex_name_regex(old)
        for p in candidates:
            if replace_in_paragraph(p, rx, lambda m: new_name):
                report.name_replaced = True
                return
    # запасной вариант: всё до «. Образец…» / «. Шифр…» в первом непустом абзаце
    for p in candidates:
        if text_of(p).strip():
            rx = re.compile(r'^\s*(.+?)(?=\.\s+(?:Образец|Шифр)|\.?\s*$)', re.S)
            if replace_in_paragraph(p, rx, lambda m: new_name.rstrip('.')):
                report.name_replaced = True
                report.warnings.append('Старое наименование в п. 3.1 найдено приблизительно — проверьте текст.')
            return
    report.warnings.append('Не удалось заменить наименование в п. 3.1.')


def replace_count(tpl: Template, count: int, report: BuildReport) -> None:
    rx = re.compile(r'(Количество\s+компонентов\s+составило\s*)(\d+)')
    for el in tpl.body_children():
        if el.tag == w('p') and replace_in_paragraph(el, rx, lambda m: m.group(1) + str(count)):
            report.count_replaced = True
            return
    report.warnings.append('Не найдена фраза «Количество компонентов составило N шт.» — количество не заменено.')


# --- Таблица №4 ---------------------------------------------------------------

def find_table4(tpl: Template):
    children = tpl.body_children()
    idx, _ = tpl.find_paragraph(r'^Таблица\s*№\s*4$')
    if idx is not None:
        for el in children[idx + 1: idx + 4]:
            if el.tag == w('tbl'):
                return el
    for el in children:
        if el.tag == w('tbl') and 'компонента' in text_of(el)[:200] and 'Свинец' in text_of(el):
            return el
    raise TemplateError('Не найдена Таблица №4 в шаблоне.')


def _cells(tr):
    return tr.findall('w:tc', NS)


def _is_vmerge_continue(tc) -> bool:
    vm = tc.find('w:tcPr/w:vMerge', NS)
    return vm is not None and vm.get(w('val')) in (None, 'continue')


def _set_cell_text(tc, text: str) -> None:
    ps = tc.findall('w:p', NS)
    p = ps[0]
    for extra in ps[1:]:
        tc.remove(extra)
    runs = p.findall('w:r', NS)
    for child in list(p):
        if child.tag not in (w('pPr'), w('r')):
            p.remove(child)          # закладки, проверка правописания и т.п.
    if runs:
        run = runs[0]
        for r in runs[1:]:
            p.remove(r)
        for child in list(run):
            if child.tag != w('rPr'):
                run.remove(child)
    else:
        run = etree.SubElement(p, w('r'))
        prpr = p.find('w:pPr/w:rPr', NS)
        if prpr is not None:
            run.append(copy.deepcopy(prpr))
    if text:
        t = etree.SubElement(run, w('t'))
        _set_t(t, text)


def _strip_ids(el) -> None:
    """Убираем w14:paraId/textId у клонов, чтобы не было дублей."""
    for node in el.iter():
        for attr in list(node.attrib):
            if attr.endswith('}paraId') or attr.endswith('}textId'):
                del node.attrib[attr]


def _set_keep_next(p, on: bool) -> None:
    """«Не отрывать от следующего» для абзаца (w:keepNext идёт сразу после w:pStyle)."""
    ppr = p.find('w:pPr', NS)
    if ppr is None:
        ppr = etree.Element(w('pPr'))
        p.insert(0, ppr)
    kn = ppr.find('w:keepNext', NS)
    if on and kn is None:
        kn = etree.Element(w('keepNext'))
        style = ppr.find('w:pStyle', NS)
        if style is not None:
            style.addnext(kn)
        else:
            ppr.insert(0, kn)
    elif not on and kn is not None:
        ppr.remove(kn)


def _keep_row(row, keep_with_next: bool) -> None:
    """Строка не разрывается между страницами; по желанию держится вместе со следующей.

    Так компонент целиком переносится на следующую страницу, если не помещается:
    у всех строк компонента, кроме последней, включено «не отрывать от следующего».
    """
    trpr = row.find('w:trPr', NS)
    if trpr is None:
        trpr = etree.Element(w('trPr'))
        prex = row.find('w:tblPrEx', NS)
        if prex is not None:
            prex.addnext(trpr)
        else:
            row.insert(0, trpr)
    if trpr.find('w:cantSplit', NS) is None:
        trpr.insert(0, etree.Element(w('cantSplit')))
    for p in row.iter(w('p')):
        _set_keep_next(p, keep_with_next)


def rebuild_table4(tpl: Template, components: list[ComponentSpec]) -> None:
    tbl = find_table4(tpl)
    rows = tbl.findall('w:tr', NS)
    header = [r for r in rows if r.find('w:trPr/w:tblHeader', NS) is not None]
    if not header:
        header = rows[:2]
    data = [r for r in rows if r not in header]
    if not data:
        raise TemplateError('В Таблице №4 нет строк с результатами — не из чего взять оформление.')
    start_tpl = data[0]
    cont_tpl = next((r for r in data[1:] if _is_vmerge_continue(_cells(r)[0])), None)
    if cont_tpl is None:
        cont_tpl = copy.deepcopy(start_tpl)
        first = _cells(cont_tpl)[0]
        vm = first.find('w:tcPr/w:vMerge', NS)
        if vm is not None:
            vm.attrib.pop(w('val'), None)
        for p in first.findall('w:p', NS):
            num = p.find('w:pPr/w:numPr', NS)
            if num is not None:
                num.getparent().remove(num)
        _set_cell_text(first, '')
    start_tpl = copy.deepcopy(start_tpl)
    cont_tpl = copy.deepcopy(cont_tpl)
    for r in data:
        tbl.remove(r)

    first_cell_numbered = start_tpl.find('w:tc[1]/w:p/w:pPr/w:numPr', NS) is not None
    for comp in components:
        elements = V.elements_for(comp.category)
        for k, element in enumerate(elements):
            row = copy.deepcopy(start_tpl if k == 0 else cont_tpl)
            _strip_ids(row)
            _keep_row(row, keep_with_next=k < len(elements) - 1)
            cells = _cells(row)
            if k == 0 and not first_cell_numbered:
                _set_cell_text(cells[0], f'{comp.number}.')
            raw = comp.values.get(element) or V.DEFAULT_TEXT
            res = V.parse(raw)
            _set_cell_text(cells[1], V.ELEMENT_TITLES[element])
            _set_cell_text(cells[2], res.mg_text)
            _set_cell_text(cells[3], res.percent_text)
            _set_cell_text(cells[4], V.interpret(res, element, comp.category))
            tbl.append(row)


# --- Приложение №1 ------------------------------------------------------------

def _text_width_twips(tpl: Template) -> int:
    sect = tpl.body.find('w:sectPr', NS)
    try:
        pg = sect.find('w:pgSz', NS)
        mar = sect.find('w:pgMar', NS)
        return int(pg.get(w('w'))) - int(mar.get(w('left'))) - int(mar.get(w('right')))
    except (AttributeError, TypeError, ValueError):
        return 10205


def _prepare_image(path: str) -> tuple[bytes, int, int]:
    im = ImageOps.exif_transpose(Image.open(path))
    if im.mode != 'RGB':
        bg = Image.new('RGB', im.size, 'white')
        rgba = im.convert('RGBA')
        bg.paste(rgba, mask=rgba.split()[3])
        im = bg
    im.thumbnail((PHOTO_MAX_PIXELS, PHOTO_MAX_PIXELS), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, 'JPEG', quality=90, optimize=True, dpi=(96, 96))
    return buf.getvalue(), im.width, im.height


def _drawing_run(rel_id: str, cx: int, cy: int, doc_pr_id: int, name: str):
    xml = f'''<w:r xmlns:w="{W_NS}" xmlns:wp="{WP_NS}" xmlns:a="{A_NS}" xmlns:pic="{PIC_NS}" xmlns:r="{R_NS}">
  <w:rPr><w:noProof/></w:rPr>
  <w:drawing>
    <wp:inline distT="0" distB="0" distL="0" distR="0">
      <wp:extent cx="{cx}" cy="{cy}"/>
      <wp:effectExtent l="0" t="0" r="0" b="0"/>
      <wp:docPr id="{doc_pr_id}" name="Рисунок {doc_pr_id}"/>
      <wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr>
      <a:graphic>
        <a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
          <pic:pic>
            <pic:nvPicPr>
              <pic:cNvPr id="0" name="{name}"/>
              <pic:cNvPicPr><a:picLocks noChangeAspect="1" noChangeArrowheads="1"/></pic:cNvPicPr>
            </pic:nvPicPr>
            <pic:blipFill>
              <a:blip r:embed="{rel_id}"/>
              <a:stretch><a:fillRect/></a:stretch>
            </pic:blipFill>
            <pic:spPr bwMode="auto">
              <a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>
              <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
              <a:noFill/>
            </pic:spPr>
          </pic:pic>
        </a:graphicData>
      </a:graphic>
    </wp:inline>
  </w:drawing>
</w:r>'''
    return etree.fromstring(xml, etree.XMLParser(remove_blank_text=True))


def _para(text_runs=(), size=None, extra_runs=()):
    """Абзац по центру, как подписи в образце (w:bCs, по желанию размер).
    Фото с подписью не разрываются за счёт cantSplit у строки таблицы."""
    p = etree.Element(w('p'))
    ppr = etree.SubElement(p, w('pPr'))
    etree.SubElement(ppr, w('jc')).set(w('val'), 'center')
    prpr = etree.SubElement(ppr, w('rPr'))
    etree.SubElement(prpr, w('bCs'))
    for r in extra_runs:
        p.append(r)
    for text in text_runs:
        r = etree.SubElement(p, w('r'))
        rpr = etree.SubElement(r, w('rPr'))
        etree.SubElement(rpr, w('bCs'))
        if size:
            etree.SubElement(rpr, w('sz')).set(w('val'), str(size))
            etree.SubElement(rpr, w('szCs')).set(w('val'), str(size))
        _set_t(etree.SubElement(r, w('t')), text)
    return p


def _photo_table(tbl_pr, grid_w: int):
    tbl = etree.Element(w('tbl'))
    if tbl_pr is not None:
        tbl.append(copy.deepcopy(tbl_pr))
    else:
        tbl.append(etree.fromstring(
            f'<w:tblPr xmlns:w="{W_NS}"><w:tblW w:w="5000" w:type="pct"/><w:jc w:val="center"/>'
            '<w:tblBorders><w:top w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
            '<w:left w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
            '<w:bottom w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
            '<w:right w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
            '<w:insideH w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
            '<w:insideV w:val="none" w:sz="0" w:space="0" w:color="auto"/></w:tblBorders>'
            '<w:tblLook w:val="04A0"/></w:tblPr>'))
    grid = etree.SubElement(tbl, w('tblGrid'))
    etree.SubElement(grid, w('gridCol')).set(w('w'), str(grid_w))
    return tbl


def range_caption(rng: tuple[int, int]) -> str:
    a, b = rng
    return f'{a} компонент' if a == b else f'{a}÷{b} компоненты'


def rebuild_appendix(tpl: Template, name: str, photos: list[PhotoSpec], report: BuildReport) -> None:
    children = tpl.body_children()
    h_idx, _ = tpl.find_paragraph(r'^Приложение\s*№\s*1\b', last=True)
    if h_idx is None:
        raise TemplateError('Не найден заголовок «Приложение №1».')
    end_idx, _ = tpl.find_paragraph(r'конец протокола', start=h_idx + 1)
    if end_idx is None:
        end_idx = len(children) - (1 if children[-1].tag == w('sectPr') else 0)

    region = children[h_idx + 1:end_idx]
    old_tbl = next((el for el in region if el.tag == w('tbl')), None)
    tbl_pr = old_tbl.find('w:tblPr', NS) if old_tbl is not None else None
    for el in region:
        tpl.body.remove(el)

    text_w = _text_width_twips(tpl)
    max_w = (text_w - 2 * 108 - 60) * EMU_PER_TWIP
    max_h = int(PHOTO_MAX_HEIGHT_CM * EMU_PER_CM)
    doc_pr_id = max([int(d.get('id')) for d in tpl.doc.iter(f'{{{WP_NS}}}docPr')
                     if (d.get('id') or '').isdigit()] + [0]) + 1
    cap_name = appendix_name(name)

    tbl = _photo_table(tbl_pr, text_w)
    for k, photo in enumerate(photos, start=1):
        data, px_w, px_h = _prepare_image(photo.path)
        media_name = _unique_media_name(tpl, f'gpi_photo{k}.jpeg')
        tpl.parts[f'word/media/{media_name}'] = data
        rel_id = _add_rel(tpl, IMAGE_REL, f'media/{media_name}')
        scale = min(max_w / px_w, max_h / px_h)
        cx, cy = int(px_w * scale), int(px_h * scale)

        tr = etree.SubElement(tbl, w('tr'))
        trpr = etree.SubElement(tr, w('trPr'))
        etree.SubElement(trpr, w('cantSplit'))
        etree.SubElement(trpr, w('jc')).set(w('val'), 'center')
        tc = etree.SubElement(tr, w('tc'))
        tcpr = etree.SubElement(tc, w('tcPr'))
        tcw = etree.SubElement(tcpr, w('tcW'))
        tcw.set(w('w'), '5000')
        tcw.set(w('type'), 'pct')
        # фото, пустая строка, «Фото №k», наименование, пустая строка, «N÷M компоненты»
        tc.append(_para(extra_runs=[_drawing_run(rel_id, cx, cy, doc_pr_id, media_name)]))
        doc_pr_id += 1
        tc.append(_para())
        tc.append(_para([f'Фото №{k}']))
        if photo.range is None:
            tc.append(_para([cap_name]))
        else:
            single = photo.range[0] == photo.range[1]
            tc.append(_para(['Компонент. ' if single else 'Компоненты. ', cap_name]))
            tc.append(_para())
            tc.append(_para([range_caption(photo.range)], size=20))
        tc.append(_para())

    h_el = children[h_idx]
    h_el.addnext(_para())
    h_el.getnext().addnext(tbl)
    tbl.addnext(_para())


# --- связи, типы, медиа ---------------------------------------------------

def _unique_media_name(tpl: Template, name: str) -> str:
    base, ext = posixpath.splitext(name)
    n, cand = 1, name
    while f'word/media/{cand}' in tpl.parts:
        n += 1
        cand = f'{base}_{n}{ext}'
    return cand


def _add_rel(tpl: Template, rel_type: str, target: str) -> str:
    ids = {r.get('Id') for r in tpl.rels}
    n = 1
    while f'rIdGpi{n}' in ids:
        n += 1
    rid = f'rIdGpi{n}'
    el = etree.SubElement(tpl.rels, f'{{{PKG_REL_NS}}}Relationship')
    el.set('Id', rid)
    el.set('Type', rel_type)
    el.set('Target', target)
    return rid


def _ensure_default_ct(tpl: Template, ext: str, ctype: str) -> None:
    for d in tpl.ctypes.findall(f'{{{CT_NS}}}Default'):
        if d.get('Extension', '').lower() == ext:
            return
    d = etree.Element(f'{{{CT_NS}}}Default')
    d.set('Extension', ext)
    d.set('ContentType', ctype)
    tpl.ctypes.insert(0, d)


def _cleanup_unused_images(tpl: Template) -> None:
    used = set()
    for el in tpl.doc.iter():
        for k, v in el.attrib.items():
            if k.startswith(f'{{{R_NS}}}'):
                used.add(v)
    for rel in list(tpl.rels):
        if rel.get('Type') == IMAGE_REL and rel.get('Id') not in used and rel.get('TargetMode') != 'External':
            tpl.rels.remove(rel)
    # медиа, на которые больше никто не ссылается
    referenced = set()
    for name, data in tpl.parts.items():
        if not name.endswith('.rels'):
            continue
        root = tpl.rels if name == 'word/_rels/document.xml.rels' else etree.fromstring(data)
        src_dir = posixpath.dirname(posixpath.dirname(name))
        for rel in root:
            if rel.get('TargetMode') == 'External':
                continue
            target = rel.get('Target', '')
            path = target[1:] if target.startswith('/') else posixpath.join(src_dir, target)
            referenced.add(posixpath.normpath(path))
    for name in list(tpl.parts):
        if name.startswith('word/media/') and name not in referenced:
            del tpl.parts[name]
            for ov in tpl.ctypes.findall(f'{{{CT_NS}}}Override'):
                if ov.get('PartName') == '/' + name:
                    tpl.ctypes.remove(ov)


# --- точка входа ----------------------------------------------------------

def validate(name: str, photos: list[PhotoSpec], components: list[ComponentSpec]) -> list[str]:
    """Ошибки, при которых генерировать нельзя."""
    errors = []
    if not name.strip():
        errors.append('Не указано наименование образца.')
    if not photos:
        errors.append('Не добавлены фотографии.')
    if not components:
        errors.append('Нет ни одного компонента — укажите номера компонентов на фото.')
    for comp in components:
        for el in V.required_for(comp.category):
            if not (comp.values.get(el) or '').strip():
                errors.append(f'Компонент {comp.number} ({V.CATEGORY_TITLES[comp.category]}): '
                              f'не введено значение «{V.ELEMENT_TITLES[el]}».')
        for el, raw in comp.values.items():
            if el not in V.elements_for(comp.category) or not (raw or '').strip():
                continue
            try:
                V.parse(raw)
            except ValueError as e:
                errors.append(f'Компонент {comp.number}, {V.ELEMENT_TITLES[el]}: {e}.')
    return errors


def build(template_path: str, out_path: str, name: str, photos: list[PhotoSpec],
          components: list[ComponentSpec]) -> BuildReport:
    errors = validate(name, photos, components)
    if errors:
        raise TemplateError('\n'.join(errors))
    report = BuildReport()
    tpl = Template(template_path)
    report.old_name = detect_old_name(tpl)
    replace_name(tpl, clean_name(name), report)
    replace_count(tpl, len(components), report)
    rebuild_table4(tpl, sorted(components, key=lambda c: c.number))
    rebuild_appendix(tpl, name, photos, report)
    _ensure_default_ct(tpl, 'jpeg', 'image/jpeg')
    _cleanup_unused_images(tpl)
    tpl.save(out_path)
    return report
