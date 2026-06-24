"""GUI: конвейер анализа капельного теста с диагностикой."""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

import cv2
import numpy as np
from PIL import Image, ImageGrab, ImageTk

from pipeline import PipelineResult, run_pipeline
from pipeline.diag_report import (
    format_brief_diagnostics,
    format_detailed_diagnostics,
    format_property_loss_report,
    format_result_summary,
)
from pipeline.diag_segments import DiagSegment

DEVELOPER = "Alim Unagasov"
APP_NAME = "Oil Blotter Analyzer"
MAX_IMAGE_SIDE = 1800
APP_VERSION = "1.2"

ABOUT_TEXT = f"""{APP_NAME}
Версия {APP_VERSION}

Программа для цифрового анализа капельного (blotter spot) теста моторного масла.

Загрузите фото пятна на фильтровальной бумаге — программа автоматически выделит зоны хроматограммы (C/A/D/T), построит спектрограмму и рассчитает коэффициент дисперсии DS, состояние присадок, топлива и ядра.

Запуск: двойной щелчок по run.bat или Запуск.bat.
При первом запуске программа сама установит Python и нужные библиотеки (если их нет) и покажет, что скачивается.

Разработчик: {DEVELOPER}
"""


class WebcamDialog:
    def __init__(self, parent: tk.Tk, on_capture) -> None:
        self.on_capture = on_capture
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            messagebox.showerror("Камера", "Не удалось открыть веб-камеру.")
            return

        self.window = tk.Toplevel(parent)
        self.window.title("Веб-камера")
        self.window.geometry("800x620")
        self.window.protocol("WM_DELETE_WINDOW", self.close)

        self.label = ttk.Label(self.window)
        self.label.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        buttons = ttk.Frame(self.window, padding=8)
        buttons.pack(fill=tk.X)
        ttk.Button(buttons, text="Сделать снимок", command=self.capture).pack(side=tk.LEFT)
        ttk.Button(buttons, text="Закрыть", command=self.close).pack(side=tk.LEFT, padx=8)

        self._running = True
        self._update_frame()

    def _update_frame(self) -> None:
        if not self._running:
            return

        ok, frame = self.cap.read()
        if ok:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            image.thumbnail((760, 520), Image.Resampling.LANCZOS)
            self._photo = ImageTk.PhotoImage(image)
            self.label.config(image=self._photo)

        self.window.after(30, self._update_frame)

    def capture(self) -> None:
        ok, frame = self.cap.read()
        if not ok:
            messagebox.showerror("Камера", "Не удалось получить кадр.")
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.on_capture(Image.fromarray(rgb))
        self.close()

    def close(self) -> None:
        self._running = False
        if self.cap.isOpened():
            self.cap.release()
        if hasattr(self, "window"):
            self.window.destroy()


class OilDropAnalyzerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(APP_NAME)
        self.root.geometry("1280x820")
        self.root.minsize(1100, 700)

        self.source_image: Image.Image | None = None
        self.result: PipelineResult | None = None
        self._photos: dict[str, ImageTk.PhotoImage] = {}

        self.contrast_method = tk.StringVar(value="clahe")
        self.clahe_clip = tk.DoubleVar(value=2.5)
        self.padding = tk.DoubleVar(value=0.08)
        self.selected_candidate = tk.IntVar(value=0)
        self._original_display_meta: dict | None = None
        self._click_point: tuple[float, float] | None = None
        self._analysis_mode: str = "auto"
        self._manual_regions: list[tuple[float, float, float]] = []
        self._drag_mode: str | None = None
        self._drag_region_index: int | None = None
        self._move_offset_image: tuple[float, float] | None = None
        self._drag_start_canvas: tuple[float, float] | None = None
        self._drag_center_image: tuple[float, float] | None = None
        self._click_pending = False
        self._drag_preview_id: int | None = None
        self._canvas_image_id: int | None = None
        self._analysis_running = False
        self._pending_analysis: dict | None = None
        self._diag_view = "brief"
        self._diag_click_reasons: dict[str, str] = {}

        self._build_menu()
        self._build_ui()
        self._bind_shortcuts()

    def _build_menu(self) -> None:
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Справка", menu=help_menu)
        help_menu.add_command(label="О программе", command=self.show_about)

    def show_about(self) -> None:
        messagebox.showinfo("О программе", ABOUT_TEXT)

    def _build_ui(self) -> None:
        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(fill=tk.X)

        for text, cmd in (
            ("Вставить (Ctrl+V)", self.paste_from_clipboard),
            ("Открыть файл", self.open_file),
            ("Веб-камера", self.open_webcam),
            ("Анализ (авто)", self.run_auto_analysis),
            ("Сохранить спектр", self.save_spectral),
            ("Сохранить оверлей", self.save_overlay),
            ("О программе", self.show_about),
        ):
            ttk.Button(toolbar, text=text, command=cmd).pack(side=tk.LEFT, padx=4)

        settings = ttk.LabelFrame(self.root, text="Конвейер", padding=8)
        settings.pack(fill=tk.X, padx=8, pady=(0, 6))

        ttk.Label(settings, text="Контраст:").pack(side=tk.LEFT)
        ttk.Radiobutton(
            settings, text="CLAHE", value="clahe", variable=self.contrast_method, command=self._rerun
        ).pack(side=tk.LEFT, padx=6)
        ttk.Radiobutton(
            settings, text="Equalize Hist", value="equalize", variable=self.contrast_method, command=self._rerun
        ).pack(side=tk.LEFT, padx=6)

        self._add_slider(settings, "CLAHE clip", self.clahe_clip, 1.0, 6.0)
        self._add_slider(settings, "Padding кропа", self.padding, 0.02, 0.2)

        ttk.Label(settings, text="Капля:").pack(side=tk.LEFT, padx=(16, 4))
        self.drop_selector = ttk.Combobox(settings, state="readonly", width=14)
        self.drop_selector.pack(side=tk.LEFT)
        self.drop_selector.bind("<<ComboboxSelected>>", self._on_drop_selected)
        ttk.Label(settings, text="(авто — «Анализ»; клик — капля в точке; круг — вручную)").pack(
            side=tk.LEFT, padx=6
        )
        ttk.Button(settings, text="Сбросить капли", command=self._reset_manual_regions).pack(side=tk.LEFT, padx=8)

        body = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        diag_frame = ttk.LabelFrame(body, text="Диагностика", padding=8)
        body.add(diag_frame, weight=1)

        diag_header = ttk.Frame(diag_frame)
        diag_header.pack(fill=tk.X, pady=(0, 6))
        self.diag_detail_btn = ttk.Button(
            diag_header,
            text="Подробно",
            command=self._toggle_diag_detail,
            state=tk.DISABLED,
        )
        self.diag_detail_btn.pack(side=tk.RIGHT)
        self.diag_summary_btn = ttk.Button(
            diag_header,
            text="Описать результат",
            command=self._show_result_summary,
            state=tk.DISABLED,
        )
        self.diag_summary_btn.pack(side=tk.RIGHT, padx=(0, 6))
        self.diag_properties_btn = ttk.Button(
            diag_header,
            text="Потеря свойств",
            command=self._show_property_loss,
            state=tk.DISABLED,
        )
        self.diag_properties_btn.pack(side=tk.RIGHT, padx=(0, 6))

        self.diag_text = scrolledtext.ScrolledText(diag_frame, wrap=tk.WORD, height=20, width=42)
        self.diag_text.pack(fill=tk.BOTH, expand=True)
        self._configure_diag_tags()
        self.diag_text.insert(tk.END, "Загрузите снимок капельного теста.\n")
        self.diag_text.config(state=tk.DISABLED)
        self.diag_text.bind("<Button-1>", self._on_diag_text_click)
        self.diag_text.bind("<Key>", lambda _e: "break")

        preview = ttk.Frame(body)
        body.add(preview, weight=3)

        top = ttk.Frame(preview)
        top.pack(fill=tk.BOTH, expand=True)
        bottom = ttk.Frame(preview)
        bottom.pack(fill=tk.BOTH, expand=True)

        self.panels = {}
        self._original_canvas: tk.Canvas | None = None
        for key, title, parent in (
            ("original", "Исходник (клик / круг / перетаскивание)", top),
            ("cropped", "Кроп капли", top),
            ("spectral", "Спектрограмма (LUT)", bottom),
            ("overlay", "Метрики на изображении", bottom),
        ):
            frame = ttk.LabelFrame(parent, text=title, padding=4)
            frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=3, pady=3)
            if key == "original":
                canvas = tk.Canvas(frame, highlightthickness=0, bg="#1e1e1e")
                canvas.pack(fill=tk.BOTH, expand=True)
                canvas.bind("<ButtonPress-1>", self._on_canvas_press)
                canvas.bind("<B1-Motion>", self._on_canvas_drag)
                canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
                canvas.bind("<Configure>", self._on_canvas_configure)
                self._original_canvas = canvas
                self.panels[key] = canvas
            else:
                label = ttk.Label(frame, text="Нет изображения", anchor=tk.CENTER)
                label.pack(fill=tk.BOTH, expand=True)
                self.panels[key] = label

        self.status = ttk.Label(
            self.root,
            text=f"Разработчик: {DEVELOPER}  |  Вставьте снимок (Ctrl+V), откройте файл или используйте веб-камеру",
            padding=6,
        )
        self.status.pack(fill=tk.X)

    def _add_slider(self, parent: ttk.Frame, label: str, variable: tk.DoubleVar, from_: float, to: float) -> None:
        frame = ttk.Frame(parent)
        frame.pack(side=tk.LEFT, padx=12)
        ttk.Label(frame, text=label).pack(side=tk.LEFT)
        ttk.Scale(
            frame,
            from_=from_,
            to=to,
            variable=variable,
            orient=tk.HORIZONTAL,
            length=120,
            command=lambda _e: self._rerun(),
        ).pack(side=tk.LEFT, padx=6)

    def _bind_shortcuts(self) -> None:
        self.root.bind("<Control-v>", lambda _e: self.paste_from_clipboard())
        self.root.bind("<Control-V>", lambda _e: self.paste_from_clipboard())
        self.root.bind("<Control-o>", lambda _e: self.open_file())
        self.root.bind("<Control-s>", lambda _e: self.save_spectral())

    def _configure_diag_tags(self) -> None:
        import tkinter.font as tkfont

        base = tkfont.nametofont("TkDefaultFont")
        family = base.cget("family")
        size = base.cget("size")
        colors = {"good": "#15803d", "warn": "#a16207", "bad": "#b91c1c"}
        for name, color in colors.items():
            self.diag_text.tag_configure(name, foreground=color)
            bold = (family, size, "bold")
            self.diag_text.tag_configure(f"val_{name}", foreground=color, font=bold)
            self.diag_text.tag_configure(f"cur_{name}", foreground=color, font=bold)
        self.diag_text.tag_configure("hint", foreground="#64748b")
        self.diag_text.tag_configure("dim", foreground="#9ca3af")

    def _show_diag_alert(self, reason: str) -> None:
        messagebox.showinfo("Почему показатель красный", reason)

    def _on_diag_text_click(self, event: tk.Event) -> str | None:
        if not self._diag_click_reasons:
            return None
        was_disabled = str(self.diag_text.cget("state")) == str(tk.DISABLED)
        if was_disabled:
            self.diag_text.config(state=tk.NORMAL)
        try:
            index = self.diag_text.index(f"@{event.x},{event.y}")
            for tag in self.diag_text.tag_names(index):
                if tag in self._diag_click_reasons:
                    self._show_diag_alert(self._diag_click_reasons[tag])
                    return "break"
        finally:
            if was_disabled:
                self.diag_text.config(state=tk.DISABLED)
        return None

    def _bind_diag_alert_tags(self) -> None:
        for click_tag, reason in self._diag_click_reasons.items():
            self.diag_text.tag_bind(
                click_tag,
                "<Button-1>",
                lambda _e, r=reason: self._show_diag_alert(r),
            )
            self.diag_text.tag_bind(
                click_tag,
                "<Enter>",
                lambda _e: self.diag_text.config(cursor="hand2"),
            )
            self.diag_text.tag_bind(
                click_tag,
                "<Leave>",
                lambda _e: self.diag_text.config(cursor=""),
            )

    def _set_diag_text(self, segments: list[DiagSegment] | str) -> None:
        self.diag_text.config(state=tk.NORMAL)
        self.diag_text.delete("1.0", tk.END)
        self._diag_click_reasons = {}
        if isinstance(segments, str):
            self.diag_text.insert(tk.END, segments)
        else:
            click_idx = 0
            for part in segments:
                tags: list[str] = []
                if part.tag:
                    tags.append(part.tag)
                if part.alert:
                    click_tag = f"diag_click_{click_idx}"
                    click_idx += 1
                    self._diag_click_reasons[click_tag] = part.alert
                    tags.append(click_tag)
                if tags:
                    self.diag_text.insert(tk.END, part.text, tuple(tags))
                else:
                    self.diag_text.insert(tk.END, part.text)
            self._bind_diag_alert_tags()
        self.diag_text.config(state=tk.DISABLED)

    def _rerun(self) -> None:
        if self.source_image is None or self._analysis_running:
            return
        if self._analysis_mode == "manual" and not self._manual_regions:
            return
        if self._analysis_mode == "click" and self._click_point is None:
            if self.result is not None:
                self._analysis_mode = "auto"
            else:
                return
        self.run_analysis()

    def _set_loaded_hint(self) -> None:
        self._diag_view = "brief"
        self._set_diag_text(
            "Снимок загружен — идёт автоопределение капель.\n\n"
            "Три способа выбора области:\n"
            "1. Авто — кнопка «Анализ (авто)» или при открытии снимка.\n"
            "2. Клик — короткий щелчок по центру капли на исходнике.\n"
            "3. Вручную — зажмите на центре и потяните до края; круг рисуется в реальном времени. "
            "Крестик над кругом удаляет зону.\n"
        )
        self.diag_detail_btn.config(state=tk.DISABLED)
        self.diag_summary_btn.config(state=tk.DISABLED)
        self.diag_properties_btn.config(state=tk.DISABLED)

    def _clear_result_panels(self) -> None:
        for key in ("cropped", "spectral", "overlay"):
            panel = self.panels[key]
            panel.config(image="", text="Нет изображения")

    def _show_source_preview(self) -> None:
        if self.source_image is None:
            return
        self._show_original_canvas(self.source_image)
        self.root.after(120, self._refresh_source_preview)

    def _refresh_source_preview(self) -> None:
        if self.source_image is not None:
            self._show_original_canvas(self.source_image)

    def _toggle_diag_detail(self) -> None:
        if self.result is None:
            return
        if self._diag_view == "detailed":
            self._diag_view = "brief"
        else:
            self._diag_view = "detailed"
        self._refresh_diagnostics()

    def _show_result_summary(self) -> None:
        if self.result is None:
            return
        self._diag_view = "summary"
        self._refresh_diagnostics()

    def _show_property_loss(self) -> None:
        if self.result is None:
            return
        self._diag_view = "properties"
        self._refresh_diagnostics()

    def _format_brief_diagnostics(self) -> list[DiagSegment]:
        return format_brief_diagnostics(self.result)

    def _format_detailed_diagnostics(self) -> list[DiagSegment]:
        return format_detailed_diagnostics(self.result)

    def _format_summary_diagnostics(self) -> list[DiagSegment]:
        return format_result_summary(self.result)

    def _format_property_loss_diagnostics(self) -> list[DiagSegment]:
        return format_property_loss_report(self.result)

    def _refresh_diagnostics(self) -> None:
        if self.result is None:
            return
        if self._diag_view == "summary":
            segments = self._format_summary_diagnostics()
        elif self._diag_view == "properties":
            segments = self._format_property_loss_diagnostics()
        elif self._diag_view == "detailed":
            segments = self._format_detailed_diagnostics()
        else:
            segments = self._format_brief_diagnostics()
        self._set_diag_text(segments)
        self.diag_detail_btn.config(
            text="Кратко" if self._diag_view == "detailed" else "Подробно",
            state=tk.NORMAL,
        )
        self.diag_summary_btn.config(state=tk.NORMAL)
        self.diag_properties_btn.config(state=tk.NORMAL)

    def paste_from_clipboard(self) -> None:
        try:
            image = ImageGrab.grabclipboard()
        except Exception as exc:
            messagebox.showerror("Ошибка", f"Буфер обмена: {exc}")
            return

        if image is None or not isinstance(image, Image.Image):
            messagebox.showinfo("Буфер", "Скопируйте изображение (Ctrl+C / Print Screen).")
            return

        self.load_image(self._prepare_image(image))

    def open_file(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("Изображения", "*.png;*.jpg;*.jpeg;*.bmp;*.webp;*.tif"), ("Все", "*.*")]
        )
        if not path:
            return
        try:
            self.load_image(self._prepare_image(Image.open(path)))
            self.status.config(text=f"Открыт: {path}")
        except Exception as exc:
            messagebox.showerror("Ошибка", str(exc))

    def open_webcam(self) -> None:
        WebcamDialog(self.root, self.load_image)

    def load_image(self, image: Image.Image) -> None:
        self.source_image = self._prepare_image(image)
        self.selected_candidate.set(0)
        self._click_point = None
        self._manual_regions = []
        self._analysis_mode = "auto"
        self.result = None
        self._update_drop_selector()
        self._clear_result_panels()
        self._show_source_preview()
        self._set_loaded_hint()
        self.run_analysis(mode="auto")

    def _reset_manual_regions(self) -> None:
        if self.source_image is None:
            return
        self._manual_regions = []
        self._click_point = None
        self._analysis_mode = "auto"
        self.selected_candidate.set(0)
        self.result = None
        self._update_drop_selector()
        self._clear_result_panels()
        self._show_source_preview()
        self._set_loaded_hint()
        self.status.config(text="Ручные области сброшены — автоопределение…")
        self.run_analysis(mode="auto")

    def _canvas_pointer(self, event: tk.Event) -> tuple[float, float]:
        canvas = self._original_canvas
        if canvas is None:
            return float(event.x), float(event.y)
        return float(canvas.canvasx(event.x)), float(canvas.canvasy(event.y))

    def _canvas_to_image(self, canvas_x: float, canvas_y: float, *, clamp: bool = False) -> tuple[float, float] | None:
        if self._original_display_meta is None:
            return None
        meta = self._original_display_meta
        local_x = canvas_x - meta["offset_x"]
        local_y = canvas_y - meta["offset_y"]
        if clamp:
            local_x = min(max(local_x, 0.0), meta["draw_w"])
            local_y = min(max(local_y, 0.0), meta["draw_h"])
        elif local_x < 0 or local_y < 0 or local_x > meta["draw_w"] or local_y > meta["draw_h"]:
            return None
        return local_x / meta["scale"], local_y / meta["scale"]

    def _image_radius_to_canvas(self, radius: float) -> float:
        if self._original_display_meta is None:
            return radius
        return radius * self._original_display_meta["scale"]

    def _hit_test_remove_button(self, canvas_x: float, canvas_y: float) -> int | None:
        canvas = self._original_canvas
        if canvas is None or not self._manual_regions:
            return None
        items = canvas.find_overlapping(canvas_x - 2, canvas_y - 2, canvas_x + 2, canvas_y + 2)
        for item in items:
            for tag in canvas.gettags(item):
                if tag.startswith("region_remove_"):
                    try:
                        return int(tag.split("_")[-1])
                    except ValueError:
                        continue
        return None

    def _image_to_canvas(self, image_x: float, image_y: float) -> tuple[float, float] | None:
        if self._original_display_meta is None:
            return None
        meta = self._original_display_meta
        return (
            meta["offset_x"] + image_x * meta["scale"],
            meta["offset_y"] + image_y * meta["scale"],
        )

    def _hit_test_region(self, canvas_x: float, canvas_y: float) -> int | None:
        point = self._canvas_to_image(canvas_x, canvas_y)
        if point is None:
            return None
        px, py = point
        for index in reversed(range(len(self._manual_regions))):
            cx, cy, radius = self._manual_regions[index]
            if (px - cx) ** 2 + (py - cy) ** 2 <= radius**2:
                return index
        return None

    def _original_display_image(self) -> Image.Image | None:
        if self.source_image is None:
            return None
        if self._manual_regions:
            return self.source_image
        if self.result is not None:
            return self.result.original
        return self.source_image

    def _remove_manual_region(self, index: int) -> None:
        if index < 0 or index >= len(self._manual_regions):
            return
        del self._manual_regions[index]
        if not self._manual_regions:
            self._analysis_mode = "auto"
            self.selected_candidate.set(0)
            image = self._original_display_image()
            if image is not None:
                self._show_original_canvas(image)
            if self.source_image is not None:
                self.run_analysis(mode="auto")
            return

        next_index = min(index, len(self._manual_regions) - 1)
        self.selected_candidate.set(next_index)
        self._analysis_mode = "manual"
        image = self._original_display_image()
        if image is not None:
            self._show_original_canvas(image)
        self.run_analysis(mode="manual")

    def _draw_region_overlays(self) -> None:
        canvas = self._original_canvas
        if canvas is None:
            return
        canvas.delete("region_overlay")
        if not self._manual_regions:
            return

        selected = self.selected_candidate.get()
        for index, (cx, cy, radius) in enumerate(self._manual_regions):
            center = self._image_to_canvas(cx, cy)
            if center is None:
                continue
            sx, sy = center
            sr = self._image_radius_to_canvas(radius)
            color = "#40dc40" if index == selected else "#ffc800"
            canvas.create_oval(
                sx - sr,
                sy - sr,
                sx + sr,
                sy + sr,
                outline=color,
                width=2,
                tags="region_overlay",
            )
            canvas.create_text(
                sx,
                sy - sr - 10,
                text=str(index + 1),
                fill=color,
                tags="region_overlay",
            )
            btn_x = sx
            btn_y = sy - sr - 22
            btn_size = 9
            remove_tag = f"region_remove_{index}"
            canvas.create_rectangle(
                btn_x - btn_size,
                btn_y - btn_size,
                btn_x + btn_size,
                btn_y + btn_size,
                fill="#b91c1c",
                outline="#ffffff",
                width=1,
                tags=("region_overlay", remove_tag),
            )
            canvas.create_text(
                btn_x,
                btn_y,
                text="×",
                fill="#ffffff",
                font=("Segoe UI", 10, "bold"),
                tags=("region_overlay", remove_tag),
            )

            def on_remove(_event: tk.Event, region_index: int = index) -> str:
                self._remove_manual_region(region_index)
                return "break"

            canvas.tag_bind(remove_tag, "<ButtonPress-1>", on_remove)
            canvas.tag_bind(remove_tag, "<Enter>", lambda _e: canvas.config(cursor="hand2"))
            canvas.tag_bind(remove_tag, "<Leave>", lambda _e: canvas.config(cursor="crosshair"))
            canvas.tag_raise(remove_tag)

    def _clear_create_preview(self) -> None:
        canvas = self._original_canvas
        if canvas is not None:
            canvas.delete("drag_preview")
        self._drag_preview_id = None

    def _draw_create_preview(self, center_image: tuple[float, float], radius_image: float) -> None:
        canvas = self._original_canvas
        if canvas is None or radius_image <= 0:
            return
        center = self._image_to_canvas(center_image[0], center_image[1])
        if center is None:
            return
        sx, sy = center
        sr = self._image_radius_to_canvas(radius_image)
        self._clear_create_preview()
        self._drag_preview_id = canvas.create_oval(
            sx - sr,
            sy - sr,
            sx + sr,
            sy + sr,
            outline="#40dc40",
            width=2,
            dash=(4, 3),
            tags="drag_preview",
        )
        canvas.create_oval(
            sx - 3,
            sy - 3,
            sx + 3,
            sy + 3,
            fill="#40dc40",
            outline="",
            tags="drag_preview",
        )

    def _clear_drag_state(self) -> None:
        self._drag_mode = None
        self._drag_region_index = None
        self._move_offset_image = None
        self._drag_start_canvas = None
        self._drag_center_image = None
        self._click_pending = False
        self._clear_create_preview()

    def _run_click_analysis(self, image_point: tuple[float, float]) -> None:
        self._manual_regions = []
        self._click_point = image_point
        self._analysis_mode = "click"
        self.run_analysis(mode="click", click_point=self._click_point)

    def _on_canvas_configure(self, _event: tk.Event) -> None:
        if self._drag_start_canvas is not None:
            return
        image = self._original_display_image()
        if image is not None:
            self._show_original_canvas(image)

    def _on_canvas_press(self, event: tk.Event) -> None:
        if self.source_image is None:
            return

        cx, cy = self._canvas_pointer(event)
        remove_index = self._hit_test_remove_button(cx, cy)
        if remove_index is not None:
            return

        self._clear_create_preview()

        hit = None
        if self._manual_regions:
            hit = self._hit_test_region(cx, cy)
        if hit is not None:
            center_x, center_y, _radius = self._manual_regions[hit]
            point = self._canvas_to_image(cx, cy, clamp=True)
            if point is None:
                return
            self._drag_mode = "move"
            self._drag_region_index = hit
            self._move_offset_image = (point[0] - center_x, point[1] - center_y)
            self._drag_start_canvas = (cx, cy)
            self._drag_center_image = None
            self._click_pending = False
            self.selected_candidate.set(hit)
            self._analysis_mode = "manual"
            if self._original_canvas is not None:
                self._original_canvas.config(cursor="fleur")
            return

        center = self._canvas_to_image(cx, cy, clamp=True)
        if center is None:
            return

        self._drag_mode = None
        self._drag_region_index = None
        self._move_offset_image = None
        self._drag_start_canvas = (cx, cy)
        self._drag_center_image = center
        self._click_pending = True

    def _on_canvas_drag(self, event: tk.Event) -> None:
        if self._drag_start_canvas is None or self._original_canvas is None:
            return

        cx, cy = self._canvas_pointer(event)
        sx, sy = self._drag_start_canvas
        moved_canvas = ((cx - sx) ** 2 + (cy - sy) ** 2) ** 0.5

        if self._drag_mode == "move" and self._drag_region_index is not None:
            point = self._canvas_to_image(cx, cy, clamp=True)
            if point is None or self._move_offset_image is None:
                return
            ox, oy = self._move_offset_image
            _center_x, _center_y, radius = self._manual_regions[self._drag_region_index]
            self._manual_regions[self._drag_region_index] = (
                point[0] - ox,
                point[1] - oy,
                radius,
            )
            self._draw_region_overlays()
            return

        if self._click_pending and moved_canvas >= 6.0 and self._drag_center_image is not None:
            self._click_pending = False
            self._drag_mode = "create"

        if self._drag_mode != "create" or self._drag_center_image is None:
            return

        current = self._canvas_to_image(cx, cy, clamp=True)
        if current is None:
            return
        center_x, center_y = self._drag_center_image
        radius = float(
            ((current[0] - center_x) ** 2 + (current[1] - center_y) ** 2) ** 0.5
        )
        self._draw_create_preview(self._drag_center_image, radius)

    def _on_canvas_release(self, event: tk.Event) -> None:
        if self._drag_start_canvas is None or self.source_image is None:
            return

        sx, sy = self._drag_start_canvas
        ex, ey = self._canvas_pointer(event)
        moved_canvas = ((ex - sx) ** 2 + (ey - sy) ** 2) ** 0.5

        if self._drag_mode == "move" and self._drag_region_index is not None:
            index = self._drag_region_index
            self._clear_drag_state()
            if self._original_canvas is not None:
                self._original_canvas.config(cursor="crosshair")
            self.selected_candidate.set(index)
            self._click_point = None
            self._analysis_mode = "manual"
            if moved_canvas >= 3.0:
                self.run_analysis(mode="manual")
            else:
                self._draw_region_overlays()
                self.run_analysis(mode="manual")
            return

        if self._click_pending and self._drag_center_image is not None:
            click_point = self._drag_center_image
            self._clear_drag_state()
            self._run_click_analysis(click_point)
            return

        finished_mode = self._drag_mode
        center = self._drag_center_image
        self._clear_drag_state()

        if center is None or finished_mode != "create":
            return

        current = self._canvas_to_image(ex, ey, clamp=True)
        if current is None:
            current = center

        radius = float(
            ((current[0] - center[0]) ** 2 + (current[1] - center[1]) ** 2) ** 0.5
        )
        radius = max(10.0, radius)

        self._manual_regions.append((center[0], center[1], radius))
        self.selected_candidate.set(len(self._manual_regions) - 1)
        self._click_point = None
        self._analysis_mode = "manual"
        self._draw_region_overlays()
        self.run_analysis(mode="manual")

    def _prepare_image(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        if max(image.size) > MAX_IMAGE_SIDE:
            image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.Resampling.LANCZOS)
        return image

    def _on_drop_selected(self, _event=None) -> None:
        if self.source_image is None or not self.drop_selector.get():
            return
        label = self.drop_selector.get()
        try:
            index = int(label.split()[-1]) - 1
        except ValueError:
            return
        self.selected_candidate.set(index)
        self._click_point = None
        self.run_analysis()

    def _update_drop_selector(self) -> None:
        if self.result is None or not self.result.candidates:
            self.drop_selector["values"] = ()
            self.drop_selector.set("")
            return

        values = [f"Капля {c.index + 1}" for c in self.result.candidates]
        self.drop_selector["values"] = values
        selected = self.result.selected_candidate
        if selected < len(values):
            self.drop_selector.set(values[selected])

    def run_auto_analysis(self) -> None:
        if self.source_image is None:
            return
        self._manual_regions = []
        self._analysis_mode = "auto"
        self._click_point = None
        self.run_analysis(mode="auto")

    def run_analysis(
        self,
        *,
        mode: str | None = None,
        click_point: tuple[float, float] | None = None,
    ) -> None:
        if self.source_image is None:
            return

        if mode is not None:
            self._analysis_mode = mode
        if click_point is not None:
            self._click_point = click_point
            self._analysis_mode = "click"

        if self._analysis_mode == "manual" and not self._manual_regions:
            return
        if self._analysis_mode == "click" and self._click_point is None:
            return

        if self._analysis_running:
            self._pending_analysis = {
                "mode": self._analysis_mode,
                "click_point": self._click_point if self._analysis_mode == "click" else None,
            }
            return

        pipeline_click: tuple[float, float] | None = None
        manual_regions: list[tuple[float, float, float]] | None = None

        if self._analysis_mode == "manual":
            manual_regions = list(self._manual_regions)
        elif self._analysis_mode == "click":
            pipeline_click = self._click_point
        # auto: оба параметра None — полное автоопределение

        self._analysis_running = True
        mode_labels = {"auto": "автоопределение", "click": "определение по клику", "manual": "ручные области"}
        self.status.config(text=f"Анализ ({mode_labels.get(self._analysis_mode, self._analysis_mode)})…")
        self.root.update_idletasks()

        args = (
            self.source_image,
            self.contrast_method.get(),
            self.clahe_clip.get(),
            self.padding.get(),
            self.selected_candidate.get(),
            pipeline_click,
            manual_regions,
        )

        thread = threading.Thread(target=self._analysis_worker, args=args, daemon=True)
        thread.start()

    def _analysis_worker(
        self,
        image: Image.Image,
        contrast_method: str,
        clahe_clip: float,
        padding: float,
        candidate_index: int,
        click_point: tuple[float, float] | None,
        manual_regions: list[tuple[float, float, float]] | None,
    ) -> None:
        try:
            result = run_pipeline(
                image,
                contrast_method=contrast_method,
                clahe_clip=clahe_clip,
                padding_ratio=padding,
                candidate_index=candidate_index,
                click_point=click_point,
                manual_regions=manual_regions,
            )
            self.root.after(0, lambda r=result: self._apply_result(r, None))
        except Exception as exc:
            self.root.after(0, lambda e=exc: self._apply_result(None, e))

    def _apply_result(self, result: PipelineResult | None, error: Exception | None) -> None:
        self._analysis_running = False
        pending = self._pending_analysis
        self._pending_analysis = None

        if error is not None:
            image = self._original_display_image()
            if image is not None:
                self._show_original_canvas(image)
            if self._analysis_mode == "manual":
                messagebox.showerror("Ошибка конвейера", str(error))
                self.status.config(text="Ошибка анализа ручной области")
            elif self._analysis_mode == "click":
                messagebox.showinfo(
                    "Клик не удался",
                    f"{error}\n\nПопробуйте «Анализ (авто)» или выделите каплю кругом вручную.",
                )
                self.status.config(text="Клик не удался — попробуйте авто или ручной круг")
            else:
                messagebox.showinfo(
                    "Автоопределение не удалось",
                    f"{error}\n\nЩёлкните по центру капли или выделите круг вручную.",
                )
                self.status.config(text="Авто не удалось — клик или ручной круг")
            if pending:
                self.run_analysis(mode=pending["mode"], click_point=pending["click_point"])
            return

        assert result is not None
        self.result = result
        self.selected_candidate.set(result.selected_candidate)
        self._update_drop_selector()
        self._show_result()
        self._diag_view = "brief"
        self._refresh_diagnostics()
        self.status.config(
            text=f"Анализ завершён — капля {self.result.selected_candidate + 1} из {len(self.result.candidates)}"
        )
        if pending:
            self.run_analysis(mode=pending["mode"], click_point=pending["click_point"])

    def _show_result(self) -> None:
        if self.result is None:
            return

        mapping = {
            "cropped": self.result.cropped,
            "spectral": self.result.spectral,
            "overlay": self.result.overlay,
        }
        for key, image in mapping.items():
            self._show_image(key, image)
        display = self._original_display_image()
        if display is not None:
            self._show_original_canvas(display)

    def _show_original_canvas(self, image: Image.Image) -> None:
        canvas = self._original_canvas
        if canvas is None:
            return
        if self._drag_start_canvas is not None:
            return

        canvas.update_idletasks()
        max_w = max(canvas.winfo_width(), 280)
        max_h = max(canvas.winfo_height(), 220)
        preview = image.copy()
        preview.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)

        scale = preview.width / image.width
        offset_x = (max_w - preview.width) // 2
        offset_y = (max_h - preview.height) // 2
        self._original_display_meta = {
            "scale": scale,
            "offset_x": offset_x,
            "offset_y": offset_y,
            "draw_w": preview.width,
            "draw_h": preview.height,
            "image_w": image.width,
            "image_h": image.height,
        }

        canvas.delete("all")
        self._canvas_image_id = None
        self._drag_preview_id = None
        photo = ImageTk.PhotoImage(preview)
        self._photos["original"] = photo
        self._canvas_image_id = canvas.create_image(offset_x, offset_y, anchor=tk.NW, image=photo)
        canvas.config(bg="#2a2a2a", cursor="crosshair")
        self._draw_region_overlays()

    def _show_image(self, key: str, image: Image.Image) -> None:
        label = self.panels[key]
        label.update_idletasks()
        max_w = max(label.winfo_width(), 280)
        max_h = max(label.winfo_height(), 220)
        preview = image.copy()
        preview.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)

        photo = ImageTk.PhotoImage(preview)
        self._photos[key] = photo
        label.config(image=photo, text="")

    def _save_image(self, image: Image.Image | None, title: str) -> None:
        if image is None:
            messagebox.showinfo("Нет данных", "Сначала выполните анализ.")
            return
        path = filedialog.asksaveasfilename(
            title=title, defaultextension=".png", filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")]
        )
        if path:
            image.save(path)
            self.status.config(text=f"Сохранено: {path}")

    def save_spectral(self) -> None:
        self._save_image(self.result.spectral if self.result else None, "Сохранить спектрограмму")

    def save_overlay(self) -> None:
        self._save_image(self.result.overlay if self.result else None, "Сохранить оверлей")


def main() -> None:
    print(f"{APP_NAME} — разработчик: {DEVELOPER}")
    root = tk.Tk()
    style = ttk.Style()
    if "vista" in style.theme_names():
        style.theme_use("vista")
    OilDropAnalyzerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
