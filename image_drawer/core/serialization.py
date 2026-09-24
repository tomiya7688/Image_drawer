"""安定したImage Drawer record向けの小さなJSON serialization layer。"""

from __future__ import annotations

import json
import types
from dataclasses import fields, is_dataclass
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

T = TypeVar("T", bound="SerializableModel")


class SerializableModel:
    """JSON互換dictとのround-tripを行うdataclass record向けMixin。"""

    def to_dict(self) -> dict[str, Any]:
        if not is_dataclass(self):
            raise TypeError("SerializableModel subclasses must be dataclasses")
        return {field.name: _encode(getattr(self, field.name)) for field in fields(self)}

    @classmethod
    def from_dict(cls: type[T], data: dict[str, Any]) -> T:
        if not isinstance(data, dict):
            raise TypeError(f"{cls.__name__}.from_dict expects a dict")

        type_hints = get_type_hints(cls)
        known_fields = {field.name: field for field in fields(cls)}
        kwargs: dict[str, Any] = {}

        for name, field in known_fields.items():
            if name not in data:
                continue
            kwargs[name] = _decode(data[name], type_hints.get(name, field.type))

        return cls(**kwargs)

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
        )

    @classmethod
    def from_json(cls: type[T], payload: str) -> T:
        data = json.loads(payload)
        if not isinstance(data, dict):
            raise TypeError(f"{cls.__name__}.from_json expects a JSON object")
        return cls.from_dict(data)


def _encode(value: Any) -> Any:
    if isinstance(value, SerializableModel):
        return value.to_dict()
    if is_dataclass(value):
        raise TypeError(
            f"dataclass {type(value).__name__} must inherit SerializableModel"
        )
    if isinstance(value, tuple):
        return [_encode(item) for item in value]
    if isinstance(value, list):
        return [_encode(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _encode(item) for key, item in value.items()}
    return value


def _decode(value: Any, annotation: Any) -> Any:
    if value is None:
        return None
    if annotation is Any:
        return value

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin in (Union, types.UnionType):
        non_none = [arg for arg in args if arg is not type(None)]
        if len(non_none) == 1:
            return _decode(value, non_none[0])
        for candidate in non_none:
            try:
                return _decode(value, candidate)
            except (TypeError, ValueError):
                continue
        return value

    if origin is list:
        item_type = args[0] if args else Any
        return [_decode(item, item_type) for item in value]

    if origin is dict:
        key_type, value_type = args if len(args) == 2 else (Any, Any)
        return {
            _decode(key, key_type): _decode(item, value_type)
            for key, item in value.items()
        }

    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_decode(item, args[0]) for item in value)
        if args:
            if len(value) != len(args):
                raise ValueError(
                    f"expected tuple with {len(args)} items, got {len(value)}"
                )
            return tuple(_decode(item, item_type) for item, item_type in zip(value, args))
        return tuple(value)

    if isinstance(annotation, type) and issubclass(annotation, SerializableModel):
        if not isinstance(value, dict):
            raise TypeError(f"expected object for {annotation.__name__}")
        return annotation.from_dict(value)

    return value
