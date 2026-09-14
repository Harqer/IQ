from __future__ import annotations

from dataclasses import dataclass

from .config import IQArchitectureConfig


@dataclass(frozen=True)
class SharedCoreTarget:
    physical_block: int
    teacher_layers: tuple[int, ...]


@dataclass(frozen=True)
class DenseToRecurrentLayout:
    """Architecture-level mapping used by later transfer/distillation code.

    This file contains no checkpoint loading and performs no weight transfer. It only
    defines which effective teacher depths correspond to each physical IQ block.
    """

    prelude: tuple[tuple[int, int], ...]
    core: tuple[SharedCoreTarget, ...]
    coda: tuple[tuple[int, int], ...]

    @classmethod
    def from_config(cls, config: IQArchitectureConfig) -> "DenseToRecurrentLayout":
        prelude = tuple(
            (target, config.teacher_layer_for_prelude(target))
            for target in range(config.prelude_layers)
        )

        core = tuple(
            SharedCoreTarget(
                physical_block=block,
                teacher_layers=tuple(
                    config.teacher_layer_for_core(pass_index, block)
                    for pass_index in range(config.recurrent_passes)
                ),
            )
            for block in range(config.recurrent_layers)
        )

        coda = tuple(
            (target, config.teacher_layer_for_coda(target))
            for target in range(config.coda_layers)
        )
        return cls(prelude=prelude, core=core, coda=coda)

    def teacher_layers_for_core_block(self, physical_block: int) -> tuple[int, ...]:
        for target in self.core:
            if target.physical_block == physical_block:
                return target.teacher_layers
        raise IndexError(physical_block)
