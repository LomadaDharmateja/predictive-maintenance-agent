"""The `ToolError` -> HTTP status mapping.

Milestone 7 item 1: a failed call must come back as a status code, never as a
200 carrying an error string. That is the same rule `docs/MILESTONE_4.md`
section 4 applies inside the agent, moved out to the wire -- v1's
`search_manual` returned its error text as a normal result and the agent kept
answering; a 200 with `{"status": "error"}` is that defect wearing an HTTP
costume, because every client that checks the status code would sail past it.

The mapping is explicit rather than a default: each `ErrorCode` is a decision
about whose fault the failure is, and a wrong guess here is the difference
between a client retrying forever and a client giving up on a transient blip.
"""

from __future__ import annotations

from http import HTTPStatus

from src.agent.contracts import ErrorCode

#: Every `ErrorCode` has a row. `test_api.py` asserts exhaustiveness, so adding
#: a code without deciding its status fails the build rather than silently
#: falling through to 500.
STATUS_FOR: dict[ErrorCode, int] = {
    # The caller asked for something that does not exist.
    ErrorCode.NOT_FOUND: HTTPStatus.NOT_FOUND,
    # The caller sent something the contract rejects. 422 rather than 400
    # because FastAPI already uses 422 for schema validation, and a client
    # should not have to tell two kinds of "your input was wrong" apart.
    ErrorCode.INVALID_INPUT: HTTPStatus.UNPROCESSABLE_ENTITY,
    # The request was valid and the answer is genuinely absent. Not an error on
    # anyone's part, but not an answer either, so it must not be a 200.
    ErrorCode.NO_DATA: HTTPStatus.NOT_FOUND,
    # Ours, and temporary: the model artefact is not loadable right now.
    ErrorCode.MODEL_UNAVAILABLE: HTTPStatus.SERVICE_UNAVAILABLE,
    # Ours, and not the caller's problem to fix.
    ErrorCode.DATABASE_ERROR: HTTPStatus.INTERNAL_SERVER_ERROR,
    # Upstream took too long. 504 tells a client this is worth retrying.
    ErrorCode.TIMEOUT: HTTPStatus.GATEWAY_TIMEOUT,
    ErrorCode.INTERNAL: HTTPStatus.INTERNAL_SERVER_ERROR,
}

#: Codes a client may sensibly retry. Sent as a header so a caller does not
#: have to hardcode this table on its side.
RETRYABLE = frozenset(
    {ErrorCode.TIMEOUT, ErrorCode.MODEL_UNAVAILABLE, ErrorCode.INTERNAL}
)


def status_for(code: ErrorCode) -> int:
    return int(STATUS_FOR.get(code, HTTPStatus.INTERNAL_SERVER_ERROR))


def is_retryable(code: ErrorCode) -> bool:
    return code in RETRYABLE
