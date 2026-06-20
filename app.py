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
        self._analysis_running = False
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
            ("Анализ", self.run_analysis),
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
        ttk.Label(settings, text="(кликните по капле на исходнике)").pack(side=tk.LEFT, padx=6)

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
        for key, title, parent in (
            ("original", "Исходник (клик — выбрать каплю)", top),
            ("cropped", "Кроп капли", top),
            ("spectral", "Спектрограмма (LUT)", bottom),
            ("overlay", "Метрики на изображении", bottom),
        ):
            frame = ttk.LabelFrame(parent, text=title, padding=4)
            frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=3, pady=3)
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
        if self.source_image is not None:
            self.run_analysis()

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
        self.run_analysis()

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

    def _on_original_click(self, event: tk.Event) -> None:
        if self.source_image is None or self._original_display_meta is None:
            return

        meta = self._original_display_meta
        local_x = event.x - meta["offset_x"]
        local_y = event.y - meta["offset_y"]
        if local_x < 0 or local_y < 0 or local_x > meta["draw_w"] or local_y > meta["draw_h"]:
            return

        self._click_point = (local_x / meta["scale"], local_y / meta["scale"])
        self.run_analysis(click_point=self._click_point)

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

    def run_analysis(self, click_point: tuple[float, float] | None = None) -> None:
        if self.source_image is None or self._analysis_running:
            return

        self._analysis_running = True
        self.status.config(text="Анализ…")
        self.root.update_idletasks()

        if click_point is not None:
            self._click_point = click_point

        args = (
            self.source_image,
            self.contrast_method.get(),
            self.clahe_clip.get(),
            self.padding.get(),
            self.selected_candidate.get(),
            self._click_point,
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
    ) -> None:
        try:
            result = run_pipeline(
                image,
                contrast_method=contrast_method,
                clahe_clip=clahe_clip,
                padding_ratio=padding,
                candidate_index=candidate_index,
                click_point=click_point,
            )
            self.root.after(0, lambda r=result: self._apply_result(r, None))
        except Exception as exc:
            self.root.after(0, lambda e=exc: self._apply_result(None, e))

    def _apply_result(self, result: PipelineResult | None, error: Exception | None) -> None:
        self._analysis_running = False
        if error is not None:
            messagebox.showerror("Ошибка конвейера", str(error))
            self.status.config(text="Ошибка анализа")
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

    def _show_result(self) -> None:
        if self.result is None:
            return

        mapping = {
            "original": self.result.original,
            "cropped": self.result.cropped,
            "spectral": self.result.spectral,
            "overlay": self.result.overlay,
        }
        for key, image in mapping.items():
            clickable = key == "original"
            self._show_image(key, image, clickable=clickable)

    def _show_image(self, key: str, image: Image.Image, *, clickable: bool = False) -> None:
        label = self.panels[key]
        label.update_idletasks()
        max_w = max(label.winfo_width(), 280)
        max_h = max(label.winfo_height(), 220)
        preview = image.copy()
        preview.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)

        scale = preview.width / image.width
        offset_x = (max_w - preview.width) // 2
        offset_y = (max_h - preview.height) // 2

        if clickable:
            self._original_display_meta = {
                "scale": scale,
                "offset_x": offset_x,
                "offset_y": offset_y,
                "draw_w": preview.width,
                "draw_h": preview.height,
            }
            label.bind("<Button-1>", self._on_original_click)
            label.config(cursor="hand2")
        else:
            label.unbind("<Button-1>")
            label.config(cursor="")

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
