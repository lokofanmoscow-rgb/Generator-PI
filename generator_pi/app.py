"""Окно программы «Генератор ПИ» (wxPython)."""
from __future__ import annotations

import itertools
import os
import queue
import sys
import threading
import traceback

import wx
import wx.grid as gridlib
import wx.lib.scrolledpanel as scrolled
from PIL import Image, ImageDraw, ImageFont

from . import values as V
from .docx_builder import ComponentSpec, PhotoSpec, TemplateError, build, validate
from .ocr import NumberReader, assign_ranges, load_image

APP_TITLE = 'Генератор ПИ'
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')
THUMB = 170

COL_ELEMENTS = list(V.ALL_ELEMENTS)
GRID_COLS = ['№', 'Тип'] + [V.ELEMENT_TITLES[e] for e in COL_ELEMENTS]



def _rgb(hex_color: str) -> wx.Colour:
    h = hex_color.lstrip('#')
    return wx.Colour(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


C_REQUIRED = _rgb('#FFF1A8')
C_FILLED = _rgb('#D6F2CF')
C_INVALID = _rgb('#F7C1C1')
C_NA = _rgb('#E4E4E4')
C_NORMAL = _rgb('#FFFFFF')


class Photo:
    _ids = itertools.count(1)

    def __init__(self, path: str):
        self.id = next(self._ids)
        self.path = path
        self.numbers: list[int] | None = None   # None — ещё распознаётся
        self.error: str | None = None
        self.range: tuple[int, int] | None = None
        self.manual = False
        self.bitmap: wx.Bitmap | None = None
        self.detections = []                    # найденные подписи (для просмотра)
        self.analysis_size = (1, 1)             # размер кадра, в котором даны их рамки


# --- плашки выбора типа -------------------------------------------------------

class CategoryChip(wx.Panel):
    def __init__(self, parent, category: str, on_select):
        super().__init__(parent, style=wx.BORDER_NONE | wx.FULL_REPAINT_ON_RESIZE)
        self.category = category
        self.selected = False
        self.on_select = on_select
        title = V.CATEGORY_TITLES[category]
        if category == V.POLYMER:
            title += ' (снять)'
        self.title = title
        self.SetFont(self._font())
        tw, th = self.GetTextExtent(title)
        self.SetMinSize(wx.Size(tw + self.FromDIP(40), th + self.FromDIP(16)))
        self.Bind(wx.EVT_PAINT, self._paint)
        self.Bind(wx.EVT_LEFT_DOWN, lambda e: self.on_select(self.category))
        self.SetCursor(wx.Cursor(wx.CURSOR_HAND))
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)

    def _font(self):
        f = self.GetFont()
        f.SetWeight(wx.FONTWEIGHT_BOLD)
        return f

    def _paint(self, _evt):
        dc = wx.AutoBufferedPaintDC(self)
        dc.SetBackground(wx.Brush(self.GetParent().GetBackgroundColour()))
        dc.Clear()
        w, h = self.GetClientSize()
        border = self.FromDIP(3) if self.selected else 1
        dc.SetPen(wx.Pen(_rgb('#1E5AA8') if self.selected else _rgb('#888888'), border))
        dc.SetBrush(wx.Brush(_rgb(V.CATEGORY_COLORS[self.category])))
        dc.DrawRoundedRectangle(border // 2, border // 2, w - border, h - border, self.FromDIP(6))
        dc.SetFont(self._font())
        tw, th = dc.GetTextExtent(self.title)
        dc.SetTextForeground(_rgb('#111111'))
        dc.DrawText(self.title, (w - tw) // 2, (h - th) // 2)

    def set_selected(self, sel: bool):
        self.selected = sel
        self.Refresh()


# --- квадратики компонентов ------------------------------------------------------

class SquaresPanel(wx.ScrolledWindow):
    """Квадратики с номерами; щелчок или протяжка мышью — закрасить выбранным типом,
    правая кнопка — вернуть «полимер»."""

    def __init__(self, parent, frame):
        super().__init__(parent, style=wx.BORDER_THEME | wx.VSCROLL)
        self.frame = frame
        self.count = 0
        self.size = self.FromDIP(46)
        self.gap = self.FromDIP(6)
        self.SetScrollRate(0, self.FromDIP(10))
        self.SetBackgroundColour(_rgb('#F7F7F7'))
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self._painting = None
        self.Bind(wx.EVT_PAINT, self._paint)
        self.Bind(wx.EVT_SIZE, lambda e: (self._update_virtual(), self.Refresh(), e.Skip()))
        self.Bind(wx.EVT_LEFT_DOWN, self._down)
        self.Bind(wx.EVT_RIGHT_DOWN, self._down)
        self.Bind(wx.EVT_MOTION, self._motion)
        self.Bind(wx.EVT_LEFT_UP, self._up)
        self.Bind(wx.EVT_RIGHT_UP, self._up)
        self.Bind(wx.EVT_MOUSE_CAPTURE_LOST, lambda e: setattr(self, '_painting', None))

    def set_count(self, n: int):
        self.count = n
        self._update_virtual()
        self.Refresh()

    def _cols(self):
        w = self.GetClientSize().width
        return max(1, (w - self.gap) // (self.size + self.gap))

    def _update_virtual(self):
        cols = self._cols()
        rows = (self.count + cols - 1) // cols
        self.SetVirtualSize(-1, rows * (self.size + self.gap) + self.gap)

    def _rect(self, n):
        cols = self._cols()
        i = n - 1
        x = self.gap + (i % cols) * (self.size + self.gap)
        y = self.gap + (i // cols) * (self.size + self.gap)
        return wx.Rect(x, y, self.size, self.size)

    def _hit(self, pos):
        x, y = self.CalcUnscrolledPosition(pos.x, pos.y)
        for n in range(1, self.count + 1):
            if self._rect(n).Contains(x, y):
                return n
        return None

    def _paint(self, _evt):
        dc = wx.AutoBufferedPaintDC(self)
        self.DoPrepareDC(dc)
        dc.SetBackground(wx.Brush(self.GetBackgroundColour()))
        dc.Clear()
        if not self.count:
            dc.SetTextForeground(_rgb('#777777'))
            dc.DrawText('Добавьте фото — здесь появятся квадратики компонентов.', self.gap * 2, self.gap * 2)
            return
        font = self.GetFont()
        font.SetWeight(wx.FONTWEIGHT_BOLD)
        dc.SetFont(font)
        for n in range(1, self.count + 1):
            r = self._rect(n)
            cat = self.frame.categories.get(n, V.POLYMER)
            dc.SetBrush(wx.Brush(_rgb(V.CATEGORY_COLORS[cat])))
            dc.SetPen(wx.Pen(_rgb('#6E6E6E'), 1))
            dc.DrawRoundedRectangle(r, self.FromDIP(5))
            if self.frame.needs_input(n):
                dc.SetPen(wx.Pen(_rgb('#D93025'), self.FromDIP(3)))
                dc.SetBrush(wx.TRANSPARENT_BRUSH)
                dc.DrawRoundedRectangle(wx.Rect(r).Deflate(self.FromDIP(2)), self.FromDIP(4))
            s = str(n)
            tw, th = dc.GetTextExtent(s)
            dc.SetTextForeground(_rgb('#111111'))
            dc.DrawText(s, r.x + (r.width - tw) // 2, r.y + (r.height - th) // 2)

    def _apply(self, n, category):
        if n is None:
            return
        if self.frame.categories.get(n, V.POLYMER) != category:
            if category == V.POLYMER:
                self.frame.categories.pop(n, None)
            else:
                self.frame.categories[n] = category
            self.frame.on_categories_changed()

    def _down(self, evt):
        self._painting = V.POLYMER if evt.RightDown() else self.frame.current_category
        if not self.HasCapture():
            self.CaptureMouse()
        self._apply(self._hit(evt.GetPosition()), self._painting)

    def _motion(self, evt):
        if self._painting and evt.Dragging():
            self._apply(self._hit(evt.GetPosition()), self._painting)

    def _up(self, _evt):
        self._painting = None
        if self.HasCapture():
            self.ReleaseMouse()


# --- карточка фото ---------------------------------------------------------------

class PhotoCard(wx.Panel):
    def __init__(self, parent, frame, photo: Photo):
        super().__init__(parent, style=wx.BORDER_THEME)
        self.frame = frame
        self.photo = photo
        self._mute = False
        self.SetBackgroundColour(_rgb('#FFFFFF'))

        self.thumb = wx.StaticBitmap(self, size=self.FromDIP(wx.Size(THUMB, THUMB * 3 // 4)))
        if photo.bitmap:
            self.thumb.SetBitmap(photo.bitmap)
        self.thumb.SetToolTip('Щёлкните, чтобы посмотреть, какие номера найдены на фото')
        self.thumb.SetCursor(wx.Cursor(wx.CURSOR_MAGNIFIER))
        self.thumb.Bind(wx.EVT_LEFT_DOWN, lambda e: PreviewDialog(self.frame, self.photo).ShowModal())
        self.title = wx.StaticText(self, label='Фото')
        f = self.title.GetFont()
        f.SetWeight(wx.FONTWEIGHT_BOLD)
        f.SetPointSize(f.GetPointSize() + 1)
        self.title.SetFont(f)
        self.file = wx.StaticText(self, label=os.path.basename(photo.path), style=wx.ST_ELLIPSIZE_MIDDLE)
        self.file.SetForegroundColour(_rgb('#666666'))
        self.status = wx.StaticText(self, label='')
        self.overview = wx.CheckBox(self, label='Общий вид образца (без номеров)')
        self.from_ = wx.SpinCtrl(self, min=1, max=999, initial=1, size=self.FromDIP(wx.Size(70, -1)))
        self.to = wx.SpinCtrl(self, min=1, max=999, initial=1, size=self.FromDIP(wx.Size(70, -1)))
        remove = wx.Button(self, label='Удалить', style=wx.BU_EXACTFIT)

        rng = wx.BoxSizer(wx.HORIZONTAL)
        rng.Add(wx.StaticText(self, label='Компоненты с'), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 4)
        rng.Add(self.from_, 0, wx.RIGHT, 4)
        rng.Add(wx.StaticText(self, label='по'), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 4)
        rng.Add(self.to, 0)

        head = wx.BoxSizer(wx.HORIZONTAL)
        head.Add(self.title, 1, wx.ALIGN_CENTER_VERTICAL)
        head.Add(remove, 0)

        right = wx.BoxSizer(wx.VERTICAL)
        right.Add(head, 0, wx.EXPAND | wx.BOTTOM, 2)
        right.Add(self.file, 0, wx.EXPAND | wx.BOTTOM, 4)
        right.Add(self.status, 0, wx.EXPAND | wx.BOTTOM, 6)
        right.Add(rng, 0, wx.BOTTOM, 6)
        right.Add(self.overview, 0)

        s = wx.BoxSizer(wx.HORIZONTAL)
        s.Add(self.thumb, 0, wx.ALL, 6)
        s.Add(right, 1, wx.ALL | wx.EXPAND, 6)
        self.SetSizer(s)

        self.overview.Bind(wx.EVT_CHECKBOX, self._on_overview)
        self.from_.Bind(wx.EVT_SPINCTRL, self._on_range)
        self.to.Bind(wx.EVT_SPINCTRL, self._on_range)
        self.from_.Bind(wx.EVT_TEXT, self._on_range)
        self.to.Bind(wx.EVT_TEXT, self._on_range)
        remove.Bind(wx.EVT_BUTTON, lambda e: self.frame.remove_photo(self.photo))
        self.refresh()

    def set_number(self, k: int):
        self.title.SetLabel(f'Фото №{k}')

    def refresh(self):
        p = self.photo
        self._mute = True
        if p.bitmap and not self.thumb.GetBitmap().IsOk():
            self.thumb.SetBitmap(p.bitmap)
        if p.error:
            self.status.SetLabel(f'Ошибка: {p.error}')
            self.status.SetForegroundColour(_rgb('#C62828'))
        elif p.numbers is None:
            self.status.SetLabel('Распознаю номера на фото…')
            self.status.SetForegroundColour(_rgb('#1E5AA8'))
        elif p.numbers:
            shown = ', '.join(map(str, p.numbers[:12])) + (' …' if len(p.numbers) > 12 else '')
            self.status.SetLabel(f'Найдены номера: {shown}')
            self.status.SetForegroundColour(_rgb('#2E7D32'))
        else:
            self.status.SetLabel('Номера не найдены')
            self.status.SetForegroundColour(_rgb('#666666'))
        busy = p.numbers is None and not p.error
        self.overview.SetValue(p.range is None and not busy)
        self.overview.Enable(not busy)
        on = p.range is not None
        self.from_.Enable(on and not busy)
        self.to.Enable(on and not busy)
        if p.range:
            self.from_.SetValue(p.range[0])
            self.to.SetValue(p.range[1])
        self._mute = False
        self.Layout()

    def _on_overview(self, _evt):
        if self._mute:
            return
        p = self.photo
        p.manual = True
        if self.overview.GetValue():
            p.range = None
        else:
            a = max(1, self.frame.component_count() + 1)
            p.range = (a, a)
        self.refresh()
        self.frame.on_ranges_changed()

    def _on_range(self, _evt):
        if self._mute or self.photo.range is None:
            return
        a, b = self.from_.GetValue(), self.to.GetValue()
        if b < a:
            b = a
            self._mute = True
            self.to.SetValue(b)
            self._mute = False
        if self.photo.range != (a, b):
            self.photo.range = (a, b)
            self.photo.manual = True
            self.frame.on_ranges_changed()


class PreviewDialog(wx.Dialog):
    """Фото крупно; найденные номера обведены зелёными рамками."""

    def __init__(self, parent, photo: Photo):
        super().__init__(parent, title=os.path.basename(photo.path),
                         style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER | wx.MAXIMIZE_BOX)
        dw, dh = wx.GetDisplaySize()
        max_w, max_h = int(dw * 0.9), int(dh * 0.82)
        try:
            img = load_image(photo.path)
        except Exception as e:  # noqa: BLE001
            img = Image.new('RGB', (400, 300), 'white')
            photo.error = photo.error or str(e)
        aw, ah = photo.analysis_size
        img.thumbnail((max_w, max_h), Image.LANCZOS)
        k = img.width / float(aw) if aw > 1 else 1.0
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.load_default(size=max(14, int(18 * self.GetDPIScaleFactor())))
        except TypeError:          # старый Pillow
            font = ImageFont.load_default()
        for d in photo.detections:
            x, y, w, h = (v * k for v in d.box)
            draw.rectangle((x, y, x + w, y + h), outline=(0, 200, 0), width=3)
            label = str(d.number)
            tb = draw.textbbox((0, 0), label, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
            ty = y - th - 6 if y - th - 6 > 0 else y + h + 2
            draw.rectangle((x, ty, x + tw + 6, ty + th + 6), fill=(0, 160, 0))
            draw.text((x + 3, ty + 3 - tb[1]), label, fill='white', font=font)
        wimg = wx.Image(img.width, img.height)
        wimg.SetData(img.convert('RGB').tobytes())

        if photo.numbers is None:
            info = 'Номера ещё распознаются…'
        elif photo.detections:
            info = (f'Найдены номера: {", ".join(map(str, sorted({d.number for d in photo.detections})))}. '
                    'Если номер не обведён или прочитан неверно — поправьте диапазон на карточке фото.')
        else:
            info = 'Номера на фото не найдены.'
        text = wx.StaticText(self, label=info)
        text.Wrap(max_w)
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(text, 0, wx.ALL, 8)
        sizer.Add(wx.StaticBitmap(self, bitmap=wx.Bitmap(wimg)), 0, wx.LEFT | wx.RIGHT, 8)
        sizer.Add(wx.Button(self, wx.ID_CLOSE, 'Закрыть'), 0, wx.ALL | wx.ALIGN_RIGHT, 8)
        self.SetSizerAndFit(sizer)
        self.Bind(wx.EVT_BUTTON, lambda e: self.EndModal(wx.ID_CLOSE), id=wx.ID_CLOSE)
        self.CenterOnScreen()


class PhotoDrop(wx.FileDropTarget):
    def __init__(self, frame):
        super().__init__()
        self.frame = frame

    def OnDropFiles(self, x, y, filenames):
        self.frame.add_photos(filenames)
        return True


# --- главное окно -------------------------------------------------------------------

class MainFrame(wx.Frame):
    def __init__(self):
        super().__init__(None, title=APP_TITLE, size=wx.Size(1400, 900))
        self.config = wx.Config('GeneratorPI')
        self.photos: list[Photo] = []
        self.cards: dict[int, PhotoCard] = {}
        self.categories: dict[int, str] = {}
        self.values: dict[int, dict[str, str]] = {}
        self.current_category = V.METAL
        self.grid_rows: list[int] = []
        self._grid_mute = False

        self.ocr_queue: queue.Queue = queue.Queue()
        threading.Thread(target=self._ocr_worker, daemon=True).start()

        self._build_ui()
        self.Maximize()
        self._select_category(V.METAL)
        self.refresh_all()
        wx.CallAfter(self._init_sashes)

    def _init_sashes(self):
        w = self.split_main.GetClientSize().width
        self.split_main.SetSashPosition(max(self.FromDIP(420), int(w * 0.4)))
        h = self.split_right.GetClientSize().height
        self.split_right.SetSashPosition(int(h * 0.48))

    # -- интерфейс ------------------------------------------------------------
    def _build_ui(self):
        root = wx.Panel(self)
        main = wx.BoxSizer(wx.VERTICAL)

        top = wx.FlexGridSizer(2, 3, 6, 8)
        top.AddGrowableCol(1, 1)
        top.Add(wx.StaticText(root, label='Шаблон ПИ (.docx):'), 0, wx.ALIGN_CENTER_VERTICAL)
        self.template = wx.TextCtrl(root, value=self.config.Read('template', ''))
        top.Add(self.template, 1, wx.EXPAND)
        btn_tpl = wx.Button(root, label='Выбрать…')
        top.Add(btn_tpl)
        top.Add(wx.StaticText(root, label='Наименование образца:'), 0, wx.ALIGN_CENTER_VERTICAL)
        self.name = wx.TextCtrl(root)
        self.name.SetHint('Например: Светильник светодиодный потолочный, марка "Homsly", модель 40W-W-R-4000K-GLSY')
        top.Add(self.name, 1, wx.EXPAND)
        top.AddSpacer(1)
        main.Add(top, 0, wx.EXPAND | wx.ALL, 10)

        split = wx.SplitterWindow(root, style=wx.SP_LIVE_UPDATE | wx.SP_3D)
        split.SetMinimumPaneSize(self.FromDIP(300))

        # левая часть — фото
        left = wx.Panel(split)
        ls = wx.BoxSizer(wx.VERTICAL)
        lh = wx.BoxSizer(wx.HORIZONTAL)
        lbl = wx.StaticText(left, label='Фотографии')
        self._bold(lbl, 2)
        lh.Add(lbl, 1, wx.ALIGN_CENTER_VERTICAL)
        btn_add = wx.Button(left, label='Добавить фото…')
        btn_clear = wx.Button(left, label='Очистить')
        lh.Add(btn_add, 0, wx.RIGHT, 4)
        lh.Add(btn_clear, 0)
        ls.Add(lh, 0, wx.EXPAND | wx.BOTTOM, 4)
        hint = wx.StaticText(left, label='Перетащите фото сюда или нажмите «Добавить фото». '
                                          'Номера на фото распознаются автоматически — проверьте диапазоны.')
        hint.SetForegroundColour(_rgb('#555555'))
        _auto_wrap(hint)
        ls.Add(hint, 0, wx.BOTTOM, 6)
        self.photo_panel = scrolled.ScrolledPanel(left, style=wx.BORDER_THEME)
        self.photo_panel.SetBackgroundColour(_rgb('#EEF2F7'))
        self.photo_sizer = wx.BoxSizer(wx.VERTICAL)
        self.photo_panel.SetSizer(self.photo_sizer)
        self.photo_panel.SetupScrolling(scroll_x=False)
        ls.Add(self.photo_panel, 1, wx.EXPAND)
        left.SetSizer(ls)

        # правая часть — квадратики и значения
        right = wx.SplitterWindow(split, style=wx.SP_LIVE_UPDATE | wx.SP_3D)
        right.SetMinimumPaneSize(self.FromDIP(160))
        sq = wx.Panel(right)
        ss = wx.BoxSizer(wx.VERTICAL)
        self.comp_label = wx.StaticText(sq, label='Компоненты')
        self._bold(self.comp_label, 2)
        ss.Add(self.comp_label, 0, wx.BOTTOM, 4)
        chips = wx.WrapSizer(wx.HORIZONTAL, wx.REMOVE_LEADING_SPACES)
        self.chips = {}
        for cat in (V.METAL, V.WIRE, V.BOARD, V.SMD, V.POLYMER):
            chip = CategoryChip(sq, cat, self._select_category)
            self.chips[cat] = chip
            chips.Add(chip, 0, wx.RIGHT | wx.BOTTOM, 6)
        ss.Add(chips, 0, wx.EXPAND)
        h2 = wx.StaticText(sq, label='Выберите тип и щёлкайте (или проводите мышью с зажатой кнопкой) по квадратикам. '
                                     'Правая кнопка — снять выделение. Не выделенные = чистый полимер. '
                                     'Красная рамка — нужно ввести значения в таблице ниже.')
        h2.SetForegroundColour(_rgb('#555555'))
        _auto_wrap(h2)
        ss.Add(h2, 0, wx.EXPAND | wx.BOTTOM, 4)
        self.squares = SquaresPanel(sq, self)
        ss.Add(self.squares, 1, wx.EXPAND)
        sq.SetSizer(ss)

        gp = wx.Panel(right)
        gs = wx.BoxSizer(wx.VERTICAL)
        gh = wx.BoxSizer(wx.HORIZONTAL)
        gl = wx.StaticText(gp, label='Значения для Таблицы №4, мг/кг')
        self._bold(gl, 2)
        gh.Add(gl, 1, wx.ALIGN_CENTER_VERTICAL)
        self.only_required = wx.CheckBox(gp, label='Показывать только компоненты, где нужен ввод')
        self.only_required.SetValue(True)
        gh.Add(self.only_required, 0, wx.ALIGN_CENTER_VERTICAL)
        gs.Add(gh, 0, wx.EXPAND | wx.BOTTOM, 4)
        h3 = wx.StaticText(gp, label='Жёлтые ячейки обязательны. Формат: «120 ± 12» (можно «120 12» или «120+-12»), '
                                     'для значений ниже предела — «Менее 10». Колонка «%» и BL/X/OL считаются сами.')
        h3.SetForegroundColour(_rgb('#555555'))
        _auto_wrap(h3)
        gs.Add(h3, 0, wx.EXPAND | wx.BOTTOM, 4)
        self.grid = gridlib.Grid(gp)
        self.grid.CreateGrid(0, len(GRID_COLS))
        self.grid.SetRowLabelSize(0)
        for i, title in enumerate(GRID_COLS):
            self.grid.SetColLabelValue(i, title)
            self.grid.SetColSize(i, self.FromDIP(50 if i == 0 else 130 if i == 1 else 115))
        self.grid.EnableDragRowSize(False)
        gs.Add(self.grid, 1, wx.EXPAND)
        gp.SetSizer(gs)

        right.SplitHorizontally(sq, gp)
        right.SetSashGravity(0.45)
        split.SplitVertically(left, right)
        split.SetSashGravity(0.4)
        self.split_main, self.split_right = split, right
        main.Add(split, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 10)

        bottom = wx.BoxSizer(wx.HORIZONTAL)
        self.status = wx.StaticText(root, label='')
        bottom.Add(self.status, 1, wx.ALIGN_CENTER_VERTICAL)
        self.btn_go = wx.Button(root, label='Сгенерировать ПИ', size=self.FromDIP(wx.Size(240, 44)))
        self._bold(self.btn_go, 2)
        bottom.Add(self.btn_go, 0)
        main.Add(bottom, 0, wx.EXPAND | wx.ALL, 10)
        root.SetSizer(main)

        for target in (self.photo_panel, left):
            target.SetDropTarget(PhotoDrop(self))

        btn_tpl.Bind(wx.EVT_BUTTON, self._choose_template)
        btn_add.Bind(wx.EVT_BUTTON, self._choose_photos)
        btn_clear.Bind(wx.EVT_BUTTON, self._clear_photos)
        self.only_required.Bind(wx.EVT_CHECKBOX, lambda e: self.refresh_grid())
        self.grid.Bind(gridlib.EVT_GRID_CELL_CHANGED, self._on_cell_changed)
        self.btn_go.Bind(wx.EVT_BUTTON, self._generate)

    def _bold(self, ctrl, plus=0):
        f = ctrl.GetFont()
        f.SetWeight(wx.FONTWEIGHT_BOLD)
        f.SetPointSize(f.GetPointSize() + plus)
        ctrl.SetFont(f)

    # -- состояние -------------------------------------------------------------
    def component_count(self) -> int:
        ends = [p.range[1] for p in self.photos if p.range]
        return max(ends) if ends else 0

    def needs_input(self, n: int) -> bool:
        """Есть незаполненное обязательное или неверно введённое значение."""
        cat = self.categories.get(n, V.POLYMER)
        vals = self.values.get(n, {})
        if any(not (vals.get(e) or '').strip() for e in V.required_for(cat)):
            return True
        for e in V.elements_for(cat):
            raw = (vals.get(e) or '').strip()
            if raw:
                try:
                    V.parse(raw)
                except ValueError:
                    return True
        return False

    def ordered_photos(self) -> list[Photo]:
        return sorted(self.photos, key=lambda p: (p.range is not None, p.range[0] if p.range else 0,
                                                  self.photos.index(p)))

    def _select_category(self, cat):
        self.current_category = cat
        for c, chip in self.chips.items():
            chip.set_selected(c == cat)

    # -- фото -------------------------------------------------------------------
    def _choose_template(self, _evt):
        start = os.path.dirname(self.template.GetValue()) if self.template.GetValue() else ''
        with wx.FileDialog(self, 'Шаблон ПИ', defaultDir=start, wildcard='Документ Word (*.docx)|*.docx',
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                self.template.SetValue(dlg.GetPath())
                self.config.Write('template', dlg.GetPath())

    def _choose_photos(self, _evt):
        wildcard = 'Изображения|' + ';'.join('*' + e for e in IMAGE_EXTS) + '|Все файлы|*.*'
        with wx.FileDialog(self, 'Фотографии', defaultDir=self.config.Read('photo_dir', ''), wildcard=wildcard,
                           style=wx.FD_OPEN | wx.FD_MULTIPLE | wx.FD_FILE_MUST_EXIST) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                paths = dlg.GetPaths()
                if paths:
                    self.config.Write('photo_dir', os.path.dirname(paths[0]))
                self.add_photos(paths)

    def _clear_photos(self, _evt):
        if self.photos and wx.MessageBox('Убрать все фото?', APP_TITLE, wx.YES_NO | wx.ICON_QUESTION) == wx.YES:
            for p in list(self.photos):
                self.remove_photo(p, refresh=False)
            self.categories.clear()
            self.values.clear()
            self.refresh_all()

    def add_photos(self, paths):
        paths = sorted((p for p in paths if p.lower().endswith(IMAGE_EXTS)), key=_natural_key)
        known = {os.path.normcase(p.path) for p in self.photos}
        added = 0
        for path in paths:
            if os.path.normcase(path) in known:
                continue
            photo = Photo(path)
            try:
                photo.bitmap = _thumbnail(path, self.FromDIP(THUMB))
            except Exception as e:  # noqa: BLE001 — показываем ошибку в карточке
                photo.error = f'не удалось открыть ({e})'
                photo.numbers = []
            self.photos.append(photo)
            card = PhotoCard(self.photo_panel, self, photo)
            self.cards[photo.id] = card
            if not photo.error:
                self.ocr_queue.put(photo)
            added += 1
        if added:
            self.refresh_all()

    def remove_photo(self, photo: Photo, refresh=True):
        if photo in self.photos:
            self.photos.remove(photo)
        card = self.cards.pop(photo.id, None)
        if card:
            card.Destroy()
        if refresh:
            self.refresh_all()

    # -- распознавание --------------------------------------------------------------
    def _ocr_worker(self):
        reader = None
        while True:
            photo = self.ocr_queue.get()
            try:
                if reader is None:
                    reader = NumberReader()
                dets, size = reader.read_full(photo.path)
                photo.detections, photo.analysis_size = dets, size
                numbers = sorted({d.number for d in dets})
                wx.CallAfter(self._on_ocr_done, photo, numbers, None)
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                wx.CallAfter(self._on_ocr_done, photo, [], str(e))

    def _on_ocr_done(self, photo: Photo, numbers, error):
        if photo not in self.photos:
            return
        photo.numbers = numbers
        photo.error = error
        if all(p.numbers is not None for p in self.photos):
            self._auto_ranges()
        if photo.id in self.cards:
            self.cards[photo.id].refresh()
        self.refresh_all()

    def _auto_ranges(self):
        """Диапазоны для фото, которые пользователь не правил руками."""
        inputs = []
        for p in self.photos:
            if p.manual:
                inputs.append(list(range(p.range[0], p.range[1] + 1)) if p.range else [])
            else:
                inputs.append(p.numbers or [])
        ranges = assign_ranges(inputs)
        for p, r in zip(self.photos, ranges):
            if not p.manual:
                p.range = r

    def on_ranges_changed(self):
        self.refresh_all(cards=False)
        self._reorder_cards()

    # -- обновление экрана -----------------------------------------------------------
    def refresh_all(self, cards=True):
        n = self.component_count()
        for k in list(self.categories):
            if k > n:
                del self.categories[k]
        if cards:
            for c in self.cards.values():
                c.refresh()
            self._reorder_cards()
        self.squares.set_count(n)
        self.comp_label.SetLabel(f'Компоненты: {n} шт.' if n else 'Компоненты')
        self.refresh_grid()
        self._update_status()

    def _reorder_cards(self):
        self.photo_sizer.Clear(delete_windows=False)
        for k, p in enumerate(self.ordered_photos(), start=1):
            card = self.cards[p.id]
            card.set_number(k)
            self.photo_sizer.Add(card, 0, wx.EXPAND | wx.ALL, 4)
        self.photo_panel.Layout()
        self.photo_panel.SetupScrolling(scroll_x=False, scrollToTop=False)

    def on_categories_changed(self):
        self.squares.Refresh()
        self.refresh_grid()
        self._update_status()

    def refresh_grid(self):
        n = self.component_count()
        only = self.only_required.GetValue()
        rows = [k for k in range(1, n + 1)
                if not only or V.required_for(self.categories.get(k, V.POLYMER))]
        self._grid_mute = True
        g = self.grid
        g.BeginBatch()
        if g.GetNumberRows() > len(rows):
            g.DeleteRows(0, g.GetNumberRows() - len(rows))
        elif g.GetNumberRows() < len(rows):
            g.AppendRows(len(rows) - g.GetNumberRows())
        self.grid_rows = rows
        for r, k in enumerate(rows):
            cat = self.categories.get(k, V.POLYMER)
            g.SetCellValue(r, 0, str(k))
            g.SetCellValue(r, 1, V.CATEGORY_TITLES[cat])
            for c in (0, 1):
                g.SetReadOnly(r, c, True)
                g.SetCellBackgroundColour(r, c, _rgb(V.CATEGORY_COLORS[cat]) if c == 1 else C_NA)
                g.SetCellAlignment(r, c, wx.ALIGN_CENTER, wx.ALIGN_CENTER)
            for c, el in enumerate(COL_ELEMENTS, start=2):
                self._paint_cell(r, c, k, cat, el)
        g.EndBatch()
        g.ForceRefresh()
        self._grid_mute = False
        self.squares.Refresh()

    def _paint_cell(self, r, c, k, cat, el):
        g = self.grid
        g.SetCellAlignment(r, c, wx.ALIGN_CENTER, wx.ALIGN_CENTER)
        if el not in V.elements_for(cat):
            g.SetCellValue(r, c, '—')
            g.SetReadOnly(r, c, True)
            g.SetCellBackgroundColour(r, c, C_NA)
            return
        g.SetReadOnly(r, c, False)
        raw = (self.values.get(k, {}).get(el) or '').strip()
        required = el in V.required_for(cat)
        if raw:
            g.SetCellValue(r, c, raw)
            try:
                V.parse(raw)
                g.SetCellBackgroundColour(r, c, C_FILLED if required else C_NORMAL)
            except ValueError:
                g.SetCellBackgroundColour(r, c, C_INVALID)
        elif required:
            g.SetCellValue(r, c, '')
            g.SetCellBackgroundColour(r, c, C_REQUIRED)
        else:
            g.SetCellValue(r, c, V.DEFAULT_TEXT)
            g.SetCellBackgroundColour(r, c, C_NORMAL)

    def _on_cell_changed(self, evt):
        if self._grid_mute:
            return
        r, c = evt.GetRow(), evt.GetCol()
        if c < 2 or r >= len(self.grid_rows):
            return
        k = self.grid_rows[r]
        el = COL_ELEMENTS[c - 2]
        raw = self.grid.GetCellValue(r, c).strip()
        cat = self.categories.get(k, V.POLYMER)
        vals = self.values.setdefault(k, {})
        if not raw or (raw.lower() == V.DEFAULT_TEXT.lower() and el not in V.required_for(cat)):
            vals.pop(el, None)
        else:
            vals[el] = raw
        self._grid_mute = True
        self._paint_cell(r, c, k, cat, el)
        self._grid_mute = False
        self.grid.ForceRefresh()
        self.squares.Refresh()
        self._update_status()

    def _update_status(self):
        busy = sum(1 for p in self.photos if p.numbers is None and not p.error)
        n = self.component_count()
        missing = sum(1 for k in range(1, n + 1) if self.needs_input(k))
        parts = []
        if busy:
            parts.append(f'Распознаю фото: осталось {busy}')
        if n:
            parts.append(f'Компонентов: {n}')
            parts.append(f'не заполнено значений: {missing}' if missing else 'все обязательные значения заполнены')
        self.status.SetLabel('   •   '.join(parts))

    # -- генерация --------------------------------------------------------------------------
    def _range_problems(self, ordered) -> list[str]:
        problems = []
        expect = 1
        for k, p in enumerate(ordered, start=1):
            if p.range is None:
                continue
            a, b = p.range
            if a > expect:
                problems.append(f'Нет фото для компонентов {expect}…{a - 1} (перед фото №{k}).')
            elif a < expect:
                problems.append(f'Фото №{k}: номера {a}…{min(b, expect - 1)} уже есть на предыдущих фото.')
            expect = max(expect, b + 1)
        return problems

    def _generate(self, _evt):
        tpl = self.template.GetValue().strip()
        if not tpl or not os.path.exists(tpl):
            wx.MessageBox('Выберите файл шаблона ПИ (.docx).', APP_TITLE, wx.ICON_WARNING)
            return
        if any(p.numbers is None and not p.error for p in self.photos):
            wx.MessageBox('Подождите — ещё распознаются номера на фото.', APP_TITLE, wx.ICON_INFORMATION)
            return
        name = self.name.GetValue().strip()
        ordered = self.ordered_photos()
        n = self.component_count()
        photos = [PhotoSpec(p.path, p.range) for p in ordered]
        comps = [ComponentSpec(k, self.categories.get(k, V.POLYMER), dict(self.values.get(k, {})))
                 for k in range(1, n + 1)]
        errors = validate(name, photos, comps)
        if errors:
            more = f'\n… и ещё {len(errors) - 15}' if len(errors) > 15 else ''
            wx.MessageBox('Нельзя сгенерировать ПИ:\n\n' + '\n'.join(errors[:15]) + more,
                          APP_TITLE, wx.ICON_WARNING)
            return
        problems = self._range_problems(ordered)
        if problems:
            msg = 'Проверьте номера компонентов на фото:\n\n' + '\n'.join(problems) + '\n\nВсё равно продолжить?'
            if wx.MessageBox(msg, APP_TITLE, wx.YES_NO | wx.ICON_QUESTION) != wx.YES:
                return

        safe = ''.join(ch for ch in name if ch not in '\\/:*?"<>|«»').strip()[:80] or 'ПИ'
        with wx.FileDialog(self, 'Сохранить ПИ', defaultDir=self.config.Read('out_dir', os.path.dirname(tpl)),
                           defaultFile=f'ПИ {safe}.docx', wildcard='Документ Word (*.docx)|*.docx',
                           style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as dlg:
            if dlg.ShowModal() != wx.ID_OK:
                return
            out = dlg.GetPath()
        if not out.lower().endswith('.docx'):
            out += '.docx'
        self.config.Write('out_dir', os.path.dirname(out))
        self.config.Write('template', tpl)

        try:
            with wx.BusyInfo('Собираю ПИ…', self):
                report = build(tpl, out, name, photos, comps)
        except TemplateError as e:
            wx.MessageBox(f'Не удалось собрать ПИ:\n\n{e}', APP_TITLE, wx.ICON_ERROR)
            return
        except PermissionError:
            wx.MessageBox('Не удалось сохранить файл — возможно, он открыт в Word. Закройте его и повторите.',
                          APP_TITLE, wx.ICON_ERROR)
            return
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            wx.MessageBox(f'Ошибка при сборке ПИ:\n\n{e}', APP_TITLE, wx.ICON_ERROR)
            return

        msg = f'ПИ сохранён:\n{out}'
        if report.warnings:
            msg += '\n\nОбратите внимание:\n' + '\n'.join('• ' + w for w in report.warnings)
        msg += '\n\nОткрыть документ?'
        if wx.MessageBox(msg, APP_TITLE, wx.YES_NO | wx.ICON_INFORMATION) == wx.YES:
            _open_file(out)


# --- вспомогательное ----------------------------------------------------------------

def _auto_wrap(label: wx.StaticText):
    """Перенос строк подсказки по ширине панели."""
    text = label.GetLabel()
    state = {'w': 0}

    def on_size(evt):
        width = evt.GetSize().width
        if width > 50 and abs(width - state['w']) > 4:
            state['w'] = width
            label.SetLabel(text)
            label.Wrap(width)
            wx.CallAfter(label.GetParent().Layout)
        evt.Skip()

    label.Bind(wx.EVT_SIZE, on_size)

def _natural_key(path: str):
    import re
    base = os.path.basename(path).lower()
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', base)]


def _thumbnail(path: str, size: int) -> wx.Bitmap:
    im = load_image(path)
    im.thumbnail((size, size * 3 // 4), Image.LANCZOS)
    wimg = wx.Image(im.width, im.height)
    wimg.SetData(im.tobytes())
    return wx.Bitmap(wimg)


def _open_file(path: str):
    try:
        if sys.platform.startswith('win'):
            os.startfile(path)  # noqa: S606
        else:
            import subprocess
            subprocess.Popen(['xdg-open', path])
    except OSError:
        pass


def _enable_dpi_awareness():
    if sys.platform.startswith('win'):
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:  # noqa: BLE001
            pass


def main():
    _enable_dpi_awareness()
    app = wx.App(False)
    frame = MainFrame()
    frame.Show()
    app.MainLoop()


if __name__ == '__main__':
    main()
