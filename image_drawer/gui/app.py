"""Tkinter workflow workbench wired to the headless controller."""

from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

from image_drawer.gui.controller import ArtifactRow, RunRecord, WorkbenchController


class WorkbenchApp:
    def __init__(self, root: tk.Tk, controller: WorkbenchController) -> None:
        self.root = root
        self.controller = controller
        self.root.title("Image Drawer Workbench")
        self.root.geometry("1220x820")
        self._parameter_vars: dict[str, tk.StringVar] = {}
        self._preview_image: tk.PhotoImage | None = None
        self._thumbnail_images: list[tk.PhotoImage] = []
        self._artifact_by_tree_id: dict[str, ArtifactRow] = {}

        self.prompt_var = tk.StringVar(value="generic")
        self.status_var = tk.StringVar(value="Ready")
        self.add_type_var = tk.StringVar()
        self._build()
        self.refresh_all()

    def _build(self) -> None:
        toolbar = ttk.Frame(self.root, padding=6)
        toolbar.pack(fill=tk.X)
        ttk.Button(toolbar, text="Load", command=self.load_workflow).pack(
            side=tk.LEFT
        )
        ttk.Button(toolbar, text="Save", command=self.save_workflow).pack(
            side=tk.LEFT, padx=(4, 0)
        )
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=8
        )
        ttk.Label(toolbar, text="Prompt").pack(side=tk.LEFT)
        ttk.Entry(
            toolbar,
            textvariable=self.prompt_var,
            width=34,
        ).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="Run", command=self.run_workflow).pack(
            side=tk.LEFT, padx=(0, 8)
        )
        ttk.Button(
            toolbar,
            text="Validate",
            command=self.validate_workflow,
        ).pack(side=tk.LEFT)
        ttk.Label(
            toolbar,
            textvariable=self.status_var,
        ).pack(side=tk.RIGHT)

        main = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

        workflow_panel = ttk.Frame(main, padding=6)
        dsl_panel = ttk.Frame(main, padding=6)
        main.add(workflow_panel, weight=2)
        main.add(dsl_panel, weight=3)

        self._build_workflow_panel(workflow_panel)
        self._build_dsl_panel(dsl_panel)

        results = ttk.Notebook(self.root)
        results.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))
        self._build_artifacts_tab(results)
        self._build_candidates_tab(results)
        self._build_scores_tab(results)
        self._build_history_tab(results)

    def _build_workflow_panel(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Workflow steps").pack(anchor=tk.W)
        step_area = ttk.Frame(parent)
        step_area.pack(fill=tk.BOTH, expand=True, pady=(4, 6))

        self.step_list = tk.Listbox(step_area, exportselection=False)
        self.step_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.step_list.bind("<<ListboxSelect>>", self._on_step_select)
        scrollbar = ttk.Scrollbar(
            step_area,
            orient=tk.VERTICAL,
            command=self.step_list.yview,
        )
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.step_list.configure(yscrollcommand=scrollbar.set)

        controls = ttk.Frame(parent)
        controls.pack(fill=tk.X)
        step_types = sorted(self.controller.registry.schemas())
        if step_types:
            self.add_type_var.set(step_types[0])
        self.add_type = ttk.Combobox(
            controls,
            state="readonly",
            values=step_types,
            textvariable=self.add_type_var,
            width=18,
        )
        self.add_type.pack(side=tk.LEFT)
        ttk.Button(controls, text="Add", command=self.add_step).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(controls, text="Delete", command=self.delete_step).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(
            controls, text="↑", width=3, command=lambda: self.move_step(-1)
        ).pack(side=tk.LEFT, padx=2)
        ttk.Button(
            controls, text="↓", width=3, command=lambda: self.move_step(1)
        ).pack(side=tk.LEFT, padx=2)

        ttk.Separator(parent).pack(fill=tk.X, pady=8)
        ttk.Label(parent, text="Parameters").pack(anchor=tk.W)
        self.parameter_frame = ttk.Frame(parent)
        self.parameter_frame.pack(fill=tk.X, pady=(4, 0))

    def _build_dsl_panel(self, parent: ttk.Frame) -> None:
        header = ttk.Frame(parent)
        header.pack(fill=tk.X)
        ttk.Label(header, text="Canonical DSL").pack(side=tk.LEFT)
        ttk.Button(
            header,
            text="Apply DSL",
            command=self.apply_dsl,
        ).pack(side=tk.RIGHT)
        self.dsl_text = tk.Text(parent, wrap="none", undo=True, height=18)
        self.dsl_text.pack(fill=tk.BOTH, expand=True, pady=(4, 6))
        self.validation_label = ttk.Label(
            parent,
            text="",
            foreground="#b00020",
            wraplength=650,
        )
        self.validation_label.pack(fill=tk.X)

    def _build_artifacts_tab(self, notebook: ttk.Notebook) -> None:
        tab = ttk.Frame(notebook, padding=6)
        notebook.add(tab, text="Artifacts")
        split = ttk.Panedwindow(tab, orient=tk.HORIZONTAL)
        split.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(split)
        right = ttk.Frame(split)
        split.add(left, weight=3)
        split.add(right, weight=2)

        columns = ("type", "step", "uri")
        self.artifact_tree = ttk.Treeview(
            left,
            columns=columns,
            show="headings",
            height=9,
        )
        for name, width in (("type", 120), ("step", 140), ("uri", 300)):
            self.artifact_tree.heading(name, text=name)
            self.artifact_tree.column(name, width=width, stretch=True)
        self.artifact_tree.pack(fill=tk.BOTH, expand=True)
        self.artifact_tree.bind(
            "<<TreeviewSelect>>",
            self._on_artifact_select,
        )

        self.preview_label = ttk.Label(
            right,
            text="Select an image Artifact to preview",
            anchor=tk.CENTER,
        )
        self.preview_label.pack(fill=tk.BOTH, expand=True)
        self.artifact_metadata = tk.Text(
            right,
            height=10,
            wrap="word",
            state="disabled",
        )
        self.artifact_metadata.pack(fill=tk.BOTH, expand=True, pady=(6, 0))

    def _build_candidates_tab(self, notebook: ttk.Notebook) -> None:
        tab = ttk.Frame(notebook, padding=6)
        notebook.add(tab, text="Candidates")
        columns = ("status", "overall", "components")
        self.candidate_tree = ttk.Treeview(
            tab,
            columns=columns,
            show="tree headings",
            height=8,
        )
        self.candidate_tree.heading("#0", text="Artifact")
        self.candidate_tree.column("#0", width=260)
        for name, width in (
            ("status", 100),
            ("overall", 100),
            ("components", 480),
        ):
            self.candidate_tree.heading(name, text=name)
            self.candidate_tree.column(name, width=width, stretch=True)
        self.candidate_tree.pack(fill=tk.BOTH, expand=True)
        ttk.Label(tab, text="Candidate thumbnails").pack(
            anchor=tk.W, pady=(8, 2)
        )
        self.thumbnail_frame = ttk.Frame(tab)
        self.thumbnail_frame.pack(fill=tk.X)

    def _build_scores_tab(self, notebook: ttk.Notebook) -> None:
        tab = ttk.Frame(notebook, padding=6)
        notebook.add(tab, text="Scores")
        columns = ("artifact", "evaluator", "version", "overall", "components")
        self.score_tree = ttk.Treeview(
            tab,
            columns=columns,
            show="headings",
            height=10,
        )
        widths = (230, 100, 80, 90, 520)
        for name, width in zip(columns, widths):
            self.score_tree.heading(name, text=name)
            self.score_tree.column(name, width=width, stretch=True)
        self.score_tree.pack(fill=tk.BOTH, expand=True)

    def _build_history_tab(self, notebook: ttk.Notebook) -> None:
        tab = ttk.Frame(notebook, padding=6)
        notebook.add(tab, text="Run history")
        self.history_tree = ttk.Treeview(
            tab,
            columns=("prompt", "steps", "artifacts", "errors"),
            show="tree headings",
        )
        self.history_tree.heading("#0", text="Run")
        self.history_tree.column("#0", width=150)
        for name, width in (
            ("prompt", 340),
            ("steps", 80),
            ("artifacts", 80),
            ("errors", 80),
        ):
            self.history_tree.heading(name, text=name)
            self.history_tree.column(name, width=width, stretch=True)
        self.history_tree.pack(fill=tk.BOTH, expand=True)
        self.history_tree.bind(
            "<<TreeviewSelect>>",
            self._on_history_select,
        )

    def _selected_step_id(self) -> str | None:
        selection = self.step_list.curselection()
        if not selection:
            return None
        return self.controller.step_ids()[selection[0]]

    def _on_step_select(self, _event: object = None) -> None:
        step_id = self._selected_step_id()
        self._render_parameters(step_id)

    def _render_parameters(self, step_id: str | None) -> None:
        for child in self.parameter_frame.winfo_children():
            child.destroy()
        self._parameter_vars.clear()
        if step_id is None:
            ttk.Label(
                self.parameter_frame,
                text="Select a Step",
            ).grid(row=0, column=0, sticky=tk.W)
            return

        step = self.controller.get_step(step_id)
        schema = self.controller.schema_for(step_id)
        values = self.controller.parameter_values(step_id)
        ttk.Label(
            self.parameter_frame,
            text=f"{step.id} · {step.type}",
        ).grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 4))
        row = 1
        for name, spec in schema.parameters.items():
            ttk.Label(self.parameter_frame, text=name).grid(
                row=row,
                column=0,
                sticky=tk.W,
                padx=(0, 6),
                pady=2,
            )
            value = values.get(name, "")
            if isinstance(value, list):
                text = json.dumps(value, ensure_ascii=False)
            elif isinstance(value, bool):
                text = "true" if value else "false"
            else:
                text = str(value)
            variable = tk.StringVar(value=text)
            self._parameter_vars[name] = variable
            if spec.choices:
                widget = ttk.Combobox(
                    self.parameter_frame,
                    textvariable=variable,
                    values=list(spec.choices),
                    state="readonly",
                )
                widget.bind(
                    "<<ComboboxSelected>>",
                    lambda _event, n=name: self._commit_parameter(n),
                )
            else:
                widget = ttk.Entry(
                    self.parameter_frame,
                    textvariable=variable,
                )
                widget.bind(
                    "<Return>",
                    lambda _event, n=name: self._commit_parameter(n),
                )
                widget.bind(
                    "<FocusOut>",
                    lambda _event, n=name: self._commit_parameter(n),
                )
            widget.grid(row=row, column=1, sticky=tk.EW, pady=2)
            row += 1
        self.parameter_frame.columnconfigure(1, weight=1)

    def _commit_parameter(self, name: str) -> None:
        step_id = self._selected_step_id()
        if step_id is None:
            return
        try:
            self.controller.set_parameter_text(
                step_id,
                name,
                self._parameter_vars[name].get(),
            )
        except Exception as exc:
            self._show_error(str(exc))
            self._render_parameters(step_id)
            return
        self.refresh_dsl()
        self._show_status(f"Updated {step_id}.{name}")

    def add_step(self) -> None:
        step_type = self.add_type_var.get()
        if not step_type:
            return
        try:
            step_id = self.controller.add_step(
                step_type,
                after_step_id=self._selected_step_id(),
            )
        except Exception as exc:
            self._show_error(str(exc))
            return
        self.refresh_steps(select_id=step_id)
        self.refresh_dsl()
        self._sync_validation()

    def delete_step(self) -> None:
        step_id = self._selected_step_id()
        if step_id is None:
            return
        try:
            self.controller.delete_step(step_id)
        except Exception as exc:
            self._show_error(str(exc))
            return
        self.refresh_steps()
        self.refresh_dsl()
        self._sync_validation()

    def move_step(self, delta: int) -> None:
        step_id = self._selected_step_id()
        if step_id is None:
            return
        try:
            self.controller.move_step(step_id, delta)
        except Exception as exc:
            self._show_error(str(exc))
            return
        self.refresh_steps(select_id=step_id)
        self.refresh_dsl()

    def apply_dsl(self) -> None:
        source = self.dsl_text.get("1.0", tk.END)
        try:
            self.controller.apply_dsl(source)
        except Exception as exc:
            self._show_error(str(exc))
            return
        self.refresh_steps()
        self.refresh_dsl()
        self._show_status("DSL applied")

    def validate_workflow(self) -> None:
        self._sync_validation(show_success=True)

    def _sync_validation(self, *, show_success: bool = False) -> None:
        error = self.controller.validate()
        if error:
            self.validation_label.configure(text=error)
            self.status_var.set("Invalid workflow")
        else:
            self.validation_label.configure(text="")
            if show_success:
                self._show_status("Workflow is valid")

    def load_workflow(self) -> None:
        path = filedialog.askopenfilename(
            title="Load workflow DSL",
            filetypes=[("Workflow DSL", "*.dsl"), ("Text", "*.txt"), ("All", "*")],
        )
        if not path:
            return
        try:
            self.controller.load_workflow(path)
        except Exception as exc:
            self._show_error(str(exc))
            return
        self.refresh_steps()
        self.refresh_dsl()
        self._show_status(f"Loaded {Path(path).name}")

    def save_workflow(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save workflow DSL",
            defaultextension=".dsl",
            filetypes=[("Workflow DSL", "*.dsl"), ("All", "*")],
        )
        if not path:
            return
        try:
            self.controller.save_workflow(path)
        except Exception as exc:
            self._show_error(str(exc))
            return
        self._show_status(f"Saved {Path(path).name}")

    def run_workflow(self) -> None:
        try:
            record = self.controller.run(self.prompt_var.get())
        except Exception as exc:
            self._show_error(str(exc))
            return
        self.refresh_results(record)
        self._show_status(f"Completed {record.run_id}")

    def refresh_all(self) -> None:
        self.refresh_steps()
        self.refresh_dsl()
        self.refresh_history()
        self._sync_validation()

    def refresh_steps(self, select_id: str | None = None) -> None:
        current = select_id or self._selected_step_id()
        self.step_list.delete(0, tk.END)
        ids = self.controller.step_ids()
        for step_id in ids:
            step = self.controller.get_step(step_id)
            backend = step.backend or self.controller.registry.backend_name(
                step.type
            )
            self.step_list.insert(
                tk.END,
                f"{step.id}  [{step.type}]  · {backend}",
            )
        if current in ids:
            index = ids.index(current)
            self.step_list.selection_set(index)
            self.step_list.see(index)
        elif ids:
            self.step_list.selection_set(0)
        self._on_step_select()

    def refresh_dsl(self) -> None:
        try:
            source = self.controller.canonical_dsl()
        except Exception as exc:
            self._show_error(str(exc))
            return
        self.dsl_text.delete("1.0", tk.END)
        self.dsl_text.insert("1.0", source)

    def refresh_results(self, record: RunRecord) -> None:
        self.refresh_artifacts(record)
        self.refresh_candidates(record)
        self.refresh_scores(record)
        self.refresh_history(select_run_id=record.run_id)

    def refresh_artifacts(self, record: RunRecord) -> None:
        for item in self.artifact_tree.get_children():
            self.artifact_tree.delete(item)
        self._artifact_by_tree_id.clear()
        for artifact in record.artifacts:
            item = self.artifact_tree.insert(
                "",
                tk.END,
                values=(
                    artifact.artifact_type,
                    artifact.step_id or "",
                    artifact.uri or "",
                ),
            )
            self._artifact_by_tree_id[item] = artifact
        self._clear_preview()

    def refresh_candidates(self, record: RunRecord) -> None:
        for item in self.candidate_tree.get_children():
            self.candidate_tree.delete(item)
        for candidate in record.candidates:
            self.candidate_tree.insert(
                "",
                tk.END,
                text=candidate.artifact_id,
                values=(
                    candidate.status,
                    "" if candidate.score_overall is None else f"{candidate.score_overall:.6g}",
                    json.dumps(candidate.components, sort_keys=True),
                ),
                tags=(candidate.status,),
            )
        self.candidate_tree.tag_configure("selected", background="#dff5df")
        self.candidate_tree.tag_configure("rejected", background="#f4e3e3")
        self._render_thumbnails(record)

    def refresh_scores(self, record: RunRecord) -> None:
        for item in self.score_tree.get_children():
            self.score_tree.delete(item)
        for score in record.scores:
            self.score_tree.insert(
                "",
                tk.END,
                values=(
                    score.artifact_id or "",
                    score.evaluator,
                    score.evaluator_version or "",
                    "" if score.overall is None else f"{score.overall:.6g}",
                    json.dumps(score.components, sort_keys=True),
                ),
            )

    def refresh_history(self, select_run_id: str | None = None) -> None:
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)
        for record in self.controller.run_history:
            item = self.history_tree.insert(
                "",
                tk.END,
                text=record.run_id,
                values=(
                    record.prompt,
                    len(record.result.trajectory.executions),
                    len(record.result.trajectory.artifacts),
                    len(record.result.trajectory.errors),
                ),
            )
            if record.run_id == select_run_id:
                self.history_tree.selection_set(item)
                self.history_tree.see(item)

    def _on_history_select(self, _event: object = None) -> None:
        selection = self.history_tree.selection()
        if not selection:
            return
        run_id = self.history_tree.item(selection[0], "text")
        record = next(
            (
                item
                for item in self.controller.run_history
                if item.run_id == run_id
            ),
            None,
        )
        if record is not None:
            self.refresh_artifacts(record)
            self.refresh_candidates(record)
            self.refresh_scores(record)

    def _on_artifact_select(self, _event: object = None) -> None:
        selection = self.artifact_tree.selection()
        if not selection:
            return
        artifact = self._artifact_by_tree_id.get(selection[0])
        if artifact is None:
            return
        self._set_metadata(
            json.dumps(
                {
                    "id": artifact.id,
                    "type": artifact.artifact_type,
                    "step": artifact.step_id,
                    "uri": artifact.uri,
                    "parents": artifact.parents,
                    "metadata": artifact.metadata,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        self._show_artifact_image(artifact)

    def _set_metadata(self, text: str) -> None:
        self.artifact_metadata.configure(state="normal")
        self.artifact_metadata.delete("1.0", tk.END)
        self.artifact_metadata.insert("1.0", text)
        self.artifact_metadata.configure(state="disabled")

    def _clear_preview(self) -> None:
        self._preview_image = None
        self.preview_label.configure(
            image="",
            text="Select an image Artifact to preview",
        )
        self._set_metadata("")

    def _load_photo(self, path: Path, max_size: int = 300) -> tk.PhotoImage | None:
        try:
            image = tk.PhotoImage(file=str(path))
        except tk.TclError:
            return None
        factor = max(
            1,
            (max(image.width(), image.height()) + max_size - 1) // max_size,
        )
        if factor > 1:
            image = image.subsample(factor, factor)
        return image

    def _show_artifact_image(self, artifact: ArtifactRow) -> None:
        path = self.controller.artifact_path(artifact)
        if path is None or not path.is_file():
            self._preview_image = None
            self.preview_label.configure(image="", text="No local image preview")
            return
        photo = self._load_photo(path)
        if photo is None:
            self._preview_image = None
            self.preview_label.configure(image="", text=str(path))
            return
        self._preview_image = photo
        self.preview_label.configure(image=photo, text="")

    def _render_thumbnails(self, record: RunRecord) -> None:
        for child in self.thumbnail_frame.winfo_children():
            child.destroy()
        self._thumbnail_images.clear()

        artifact_by_id = {artifact.id: artifact for artifact in record.artifacts}
        shown = 0
        for candidate in record.candidates:
            artifact = artifact_by_id.get(candidate.artifact_id)
            if artifact is not None:
                path = self.controller.artifact_path(artifact)
            elif candidate.uri and self.controller.bank_dir is not None:
                candidate_path = (
                    self.controller.bank_dir / candidate.uri
                ).resolve()
                path = (
                    candidate_path
                    if candidate_path.is_relative_to(
                        self.controller.bank_dir
                    )
                    else None
                )
            else:
                path = None
            if path is None or not path.is_file():
                continue
            photo = self._load_photo(path, max_size=110)
            if photo is None:
                continue
            self._thumbnail_images.append(photo)
            frame = ttk.Frame(self.thumbnail_frame, padding=3)
            frame.pack(side=tk.LEFT)
            ttk.Label(frame, image=photo).pack()
            ttk.Label(
                frame,
                text=f"{candidate.status}\n{candidate.artifact_id}",
                justify=tk.CENTER,
            ).pack()
            shown += 1
        if shown == 0:
            ttk.Label(
                self.thumbnail_frame,
                text="No candidate image files available for preview",
            ).pack(anchor=tk.W)

    def _show_error(self, message: str) -> None:
        self.validation_label.configure(text=message)
        self.status_var.set("Error")
        messagebox.showerror("Image Drawer", message)

    def _show_status(self, message: str) -> None:
        self.validation_label.configure(text="")
        self.status_var.set(message)


def run_app(controller: WorkbenchController) -> int:
    root = tk.Tk()
    WorkbenchApp(root, controller)
    root.mainloop()
    return 0
