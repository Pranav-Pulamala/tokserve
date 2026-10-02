"""Request data and explicit scheduler lifecycle states."""

from enum import StrEnum

import torch

from tokserve.generation.types import (
    GenerationConfig,
    GenerationResult,
    StopReason,
)


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
        self._prefill_complete = False
        self._next_input_token: torch.Tensor | None = None
        self._completion_reason: StopReason | None = None
        self._generator: torch.Generator | None = None

        if generation_config.do_sample and generation_config.seed is not None:
            self._generator = torch.Generator(device=input_ids.device)
            self._generator.manual_seed(generation_config.seed)

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

    @property
    def prefill_complete(self) -> bool:
        """Return whether the prompt has populated its paged cache."""

        return self._prefill_complete

    @property
    def decode_ready(self) -> bool:
        """Return whether the request can perform one decode step."""

        return (
            self._state is RequestState.RUNNING
            and self._prefill_complete
            and self._next_input_token is not None
            and self._completion_reason is None
        )

    @property
    def next_input_token(self) -> torch.Tensor:
        """Return the token that should be appended during the next decode."""

        if not self.decode_ready or self._next_input_token is None:
            raise RuntimeError("request is not decode-ready")

        return self._next_input_token.clone()

    @property
    def sequence_length(self) -> int:
        """Return prompt length plus generated output length."""

        return self._input_ids.shape[1] + self._generated_token_ids.shape[1]

    @property
    def num_generated_tokens(self) -> int:
        """Return the number of generated tokens."""

        return self._generated_token_ids.shape[1]

    @property
    def completion_reason(self) -> StopReason | None:
        """Return why iterative generation should finish, if applicable."""

        return self._completion_reason

    @property
    def sampling_generator(self) -> torch.Generator | None:
        """Return this request's independent seeded sampling generator."""

        return self._generator

    def mark_prefill_complete(self) -> None:
        """Record successful prompt prefill."""

        self._require_state(RequestState.RUNNING)

        if self._prefill_complete:
            raise RuntimeError("prefill is already complete")

        self._prefill_complete = True

        if self.generation_config.max_new_tokens == 0:
            self._completion_reason = "max_new_tokens"

    def build_result(self) -> GenerationResult:
        """Build the final result after an iterative completion condition."""

        self._require_state(RequestState.RUNNING)

        if self._completion_reason is None:
            raise RuntimeError("request has not reached a completion condition")

        return GenerationResult(
            token_ids=torch.cat(
                (self._input_ids, self._generated_token_ids),
                dim=1,
            ),
            generated_token_ids=self._generated_token_ids.clone(),
            stop_reason=self._completion_reason,
        )

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

        if not self._prefill_complete:
            raise RuntimeError("prefill must complete before generating tokens")

        if self._completion_reason is not None:
            raise RuntimeError("request has already completed generation")

        self._generated_token_ids = torch.cat(
            (self._generated_token_ids, token_ids),
            dim=1,
        )
        self._next_input_token = token_ids.clone()

        eos_token_id = self.generation_config.eos_token_id

        if eos_token_id is not None and token_ids.item() == eos_token_id:
            self._completion_reason = "eos"
        elif self.num_generated_tokens >= self.generation_config.max_new_tokens:
            self._completion_reason = "max_new_tokens"

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
