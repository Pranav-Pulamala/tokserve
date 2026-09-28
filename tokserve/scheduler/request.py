"""Request data and explicit scheduler lifecycle states."""

from enum import StrEnum

import torch

from tokserve.generation.types import GenerationConfig, GenerationResult


class RequestState(StrEnum):
    """Lifecycle states for one inference request."""

    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"
    CANCELLED = "cancelled"


class ScheduledRequest:
    """One validated generation request managed by the scheduler."""

    def __init__(
        self,
        request_id: str,
        input_ids: torch.Tensor,
        generation_config: GenerationConfig,
    ) -> None:
        if not request_id:
            raise ValueError("request_id must be nonempty")

        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("input_ids must have shape (1, T)")

        if input_ids.shape[1] < 1:
            raise ValueError("input_ids must contain at least one token")

        if input_ids.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise TypeError("input_ids must contain integers")

        self.request_id = request_id
        self.generation_config = generation_config
        self._input_ids = input_ids.clone()
        self._generated_token_ids = input_ids.new_empty((1, 0))
        self._state = RequestState.WAITING
        self._result: GenerationResult | None = None

    @property
    def input_ids(self) -> torch.Tensor:
        """Return a copy of the prompt token IDs."""

        return self._input_ids.clone()

    @property
    def generated_token_ids(self) -> torch.Tensor:
        """Return a copy of tokens generated so far."""

        return self._generated_token_ids.clone()

    @property
    def state(self) -> RequestState:
        """Return the current lifecycle state."""

        return self._state

    @property
    def result(self) -> GenerationResult | None:
        """Return the completed generation result, if available."""

        return self._result

    def mark_running(self) -> None:
        """Transition this request from waiting to running."""

        self._require_state(RequestState.WAITING)
        self._state = RequestState.RUNNING

    def append_generated_token(self, token_ids: torch.Tensor) -> None:
        """Record one generated token while this request is running."""

        self._require_state(RequestState.RUNNING)

        if token_ids.shape != (1, 1):
            raise ValueError("token_ids must have shape (1, 1)")

        if token_ids.dtype != self._input_ids.dtype:
            raise TypeError("generated tokens must match the prompt dtype")

        if token_ids.device != self._input_ids.device:
            raise ValueError("generated tokens must match the prompt device")

        self._generated_token_ids = torch.cat(
            (self._generated_token_ids, token_ids),
            dim=1,
        )

    def mark_finished(self, result: GenerationResult | None = None) -> None:
        """Transition this request from running to finished."""

        self._require_state(RequestState.RUNNING)

        if result is not None:
            self._generated_token_ids = result.generated_token_ids.clone()
            self._result = result

        self._state = RequestState.FINISHED

    def cancel(self) -> None:
        """Cancel a waiting or running request."""

        if self._state not in (RequestState.WAITING, RequestState.RUNNING):
            raise RuntimeError(f"cannot cancel request in state {self._state.value}")

        self._state = RequestState.CANCELLED

    def _require_state(self, expected: RequestState) -> None:
        """Require one exact lifecycle state."""

        if self._state is not expected:
            raise RuntimeError(
                f"request must be {expected.value}, not {self._state.value}"
            )
