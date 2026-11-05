"""tkinterの画面（view）。表示と入力の受け渡しだけを行い、状態遷移はviewmodel/controllerに任せる。

自動テストで確認できるのは組み立てが例外にならないことだけで、操作や見た目は手動確認が必要。
"""

from __future__ import annotations

import base64
import tkinter as tk
from tkinter import filedialog, ttk

from image_drawer.gui.viewmodel import (
    ViewerController,
    ViewerEmpty,
    ViewerLoaded,
    ViewerLoadError,
    ViewerModel,
)
from image_drawer.inference.png import encode_png_rgba

STATUS_LABEL = {
    "unreached": "未到達", "in_progress": "進行中", "done": "完了",
    "reviewed": "レビュー", "interrupted": "中断",
}  # fmt: skip
VIEW_SIDE = 320
PLAY_INTERVAL_MS = 120


# {
#   責務: [ ReplayView: Replay画面(G1)。読み取り専用で、Episodeも筆・色の選択も変更できない ]
# }
class ReplayView(ttk.Frame):
    def __init__(self, master: tk.Misc, controller: ViewerController | None = None) -> None:
        super().__init__(master)
        self.controller = controller or ViewerController()
        self._photo: tk.PhotoImage | None = None
        self._playing = False
        self._build()

    def _build(self) -> None:
        top = ttk.Frame(self)
        top.pack(fill="x")
        ttk.Button(top, text="開く", command=self.ask_open).pack(side="left")
        self.badge = ttk.Label(top, text="")
        self.badge.pack(side="left", padx=8)
        self.draft = ttk.Label(top, text="", foreground="#b00020")
        self.draft.pack(side="left")
        body = ttk.Frame(self)
        body.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(body, width=VIEW_SIDE, height=VIEW_SIDE, background="#dddddd")
        self.canvas.pack(side="left")
        side = ttk.Frame(body)
        side.pack(side="left", fill="both", expand=True, padx=8)
        self.stage_labels = [ttk.Label(side, text="") for _ in range(6)]
        for label in self.stage_labels:
            label.pack(anchor="w")
        self.layers = tk.Listbox(side, height=8)
        self.layers.pack(fill="x")
        self.brush = ttk.Label(side, text="")
        self.brush.pack(anchor="w")
        controls = ttk.Frame(self)
        controls.pack(fill="x")
        ttk.Button(controls, text="◀", width=3, command=lambda: self._step(-1)).pack(side="left")
        ttk.Button(controls, text="▶", width=3, command=lambda: self._step(1)).pack(side="left")
        ttk.Button(controls, text="再生/停止", command=self.toggle_play).pack(side="left")
        self.slider = tk.Scale(controls, from_=-1, to=0, orient="horizontal", command=self._on_seek)
        self.slider.pack(side="left", fill="x", expand=True)
        self.history = tk.Listbox(self, height=8)
        self.history.pack(fill="both", expand=True)
        self.history.bind("<<ListboxSelect>>", self._on_history_select)
        self._syncing = False

    def ask_open(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Episode", "*.json")])
        if path:
            self.open_path(path)

    def open_path(self, path: str, *, policy_trained: bool | None = None) -> None:
        self.controller.load(path, policy_trained=policy_trained)
        self.refresh(reload=True)

    def _model(self) -> ViewerModel | None:
        state = self.controller.state
        return state.model if isinstance(state, ViewerLoaded) else None

    def _step(self, delta: int) -> None:
        model = self._model()
        if model:
            model.cursor.seek_index(model.cursor.index + delta)
            self.refresh()

    def _on_seek(self, value: str) -> None:
        model = self._model()
        if model and not self._syncing:
            model.cursor.seek_index(int(value))
            self.refresh()

    def _on_history_select(self, _event: object) -> None:
        model, selected = self._model(), self.history.curselection()
        if model and selected and not self._syncing:
            model.cursor.seek(model.episode.events[selected[0]].seq)
            self.refresh()

    def toggle_play(self) -> None:
        self._playing = not self._playing
        if self._playing:
            self._tick()

    def _tick(self) -> None:
        model = self._model()
        if not self._playing or model is None or model.cursor.index >= model.cursor.last_index:
            self._playing = False
            return
        self._step(1)
        self.after(PLAY_INTERVAL_MS, self._tick)

    def refresh(self, reload: bool = False) -> None:
        state = self.controller.state
        if isinstance(state, ViewerLoadError):
            self.badge.config(text="")
            self.draft.config(text=f"読み込めません: {state.reason}")
            return
        model = self._model()
        if isinstance(state, ViewerEmpty) or model is None:
            return
        self._syncing = True
        try:
            self.badge.config(text=model.policy_badge)
            self.draft.config(text=model.draft_badge or "")
            if reload:
                self.slider.config(to=model.cursor.last_index)
                self.history.delete(0, "end")
                for row in model.action_history():
                    self.history.insert("end", f"{row.seq} {row.actor} {row.type} {row.summary}")
            self.slider.set(model.cursor.index)
            self._draw_canvas(model)
            for label, item in zip(self.stage_labels, model.stage_bar()):
                reason = f" ({item.review_reason})" if item.review_reason else ""
                label.config(text=f"{item.stage_id}: {STATUS_LABEL[item.status]}{reason}")
            self.layers.delete(0, "end")
            for row in model.layer_panel():
                mark = "表示" if row.visible else "非表示"
                self.layers.insert("end", f"{row.name} [{row.role}] {mark}")
            panel = model.brush_panel()
            self.brush.config(
                text=f"色{panel.color_id} 筆{panel.brush_id} サイズ{panel.size_level_id} "
                f"不透明度{panel.opacity_level_id} 筆圧{panel.pressure_level_id}"
            )
        finally:
            self._syncing = False

    def _draw_canvas(self, model: ViewerModel) -> None:
        image = model.canvas()
        zoom = max(1, VIEW_SIDE // max(image.width, image.height))
        png = encode_png_rgba(image.width, image.height, image.rgba)
        self._photo = tk.PhotoImage(data=base64.b64encode(png)).zoom(zoom)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo)
        stroke = model.cursor.last_stroke()
        if stroke:
            points = [
                coordinate
                for sample in stroke.payload["samples"]
                for coordinate in (sample["x_q8"] / 256 * zoom, sample["y_q8"] / 256 * zoom)
            ]
            if len(points) >= 4:
                self.canvas.create_line(*points, fill="#e53935", width=2)


# {
#   責務: [ build_main_window: ウィンドウとReplay画面を組み立てる ]
# }
def build_main_window(root: tk.Tk) -> ReplayView:
    root.title("Image Drawer")
    view = ReplayView(root)
    view.pack(fill="both", expand=True)
    return view


def run(open_path: str | None = None) -> int:
    try:
        root = tk.Tk()
    except tk.TclError as error:
        print(f"画面を開けません: {error}")
        return 2
    view = build_main_window(root)
    if open_path:
        view.open_path(open_path)
    root.mainloop()
    return 0
