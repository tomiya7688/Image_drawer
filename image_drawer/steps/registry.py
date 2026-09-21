"""Step implementation registry."""

from __future__ import annotations

from image_drawer.steps.base import Step


class StepRegistry:
    """Resolve logical Step types to independently registered backends."""

    def __init__(self) -> None:
        self._steps: dict[tuple[str, str], Step] = {}
        self._defaults: dict[str, str] = {}

    def register(
        self,
        step: Step,
        *,
        backend: str | None = None,
        default: bool = False,
    ) -> None:
        step_type = step.schema.step_type
        backend_name = backend or step.backend
        key = (step_type, backend_name)
        if key in self._steps:
            raise ValueError(
                f"Step backend already registered: {step_type}/{backend_name}"
            )
        self._steps[key] = step
        if default or step_type not in self._defaults:
            self._defaults[step_type] = backend_name

    def resolve(self, step_type: str, backend: str | None = None) -> Step:
        backend_name = backend or self._defaults.get(step_type)
        if backend_name is None:
            raise KeyError(f"unknown Step type: {step_type}")
        try:
            return self._steps[(step_type, backend_name)]
        except KeyError as exc:
            raise KeyError(
                f"unknown Step backend: {step_type}/{backend_name}"
            ) from exc

    def backend_name(self, step_type: str, backend: str | None = None) -> str:
        implementation = self.resolve(step_type, backend)
        return backend or self._defaults[step_type] or implementation.backend

    def schemas(self) -> dict[str, object]:
        return {
            step_type: self.resolve(step_type).schema
            for step_type in self._defaults
        }
