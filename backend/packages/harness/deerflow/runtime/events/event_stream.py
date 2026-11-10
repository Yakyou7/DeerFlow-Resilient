"""EventStream: Reactive immutable event stream inspired by OpenHands.

Decouples cognition, execution, observation, and guards through a strongly-typed,
immutable event stream with monotonic ordering and causal DAG lineage tracking.
"""

from __future__ import annotations

import dataclasses
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Callable, Literal
import uuid
import threading

EventSource = Literal["agent", "user", "environment", "guard", "system"]
ObservationStatus = Literal["success", "error", "cancelled"]
SystemStatus = Literal["init", "running", "paused", "completed", "failed"]


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True, kw_only=True)
class EventStreamEvent:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=_utc_now_iso)
    source: str = "agent"
    cause_id: str | None = None
    seq: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, kw_only=True)
class Action(EventStreamEvent):
    action_type: str
    params: dict[str, Any] = field(default_factory=dict)
    timeout: float | None = None

    @classmethod
    def create(
        cls,
        action_type: str,
        params: dict[str, Any] | None = None,
        *,
        source: str = "agent",
        cause_id: str | None = None,
        timeout: float | None = None,
    ) -> Action:
        return cls(
            action_type=action_type,
            params=dict(params or {}),
            source=source,
            cause_id=cause_id,
            timeout=timeout,
        )


@dataclass(frozen=True, kw_only=True)
class Observation(EventStreamEvent):
    observation_type: str
    content: str = ""
    exit_code: int | None = None
    status: ObservationStatus = "success"

    @classmethod
    def create(
        cls,
        observation_type: str,
        content: str = "",
        *,
        exit_code: int | None = None,
        status: ObservationStatus = "success",
        source: str = "environment",
        cause_id: str | None = None,
    ) -> Observation:
        return cls(
            observation_type=observation_type,
            content=content,
            exit_code=exit_code,
            status=status,
            source=source,
            cause_id=cause_id,
        )


@dataclass(frozen=True, kw_only=True)
class StatusEvent(EventStreamEvent):
    status: SystemStatus = "running"
    details: str = ""

    @classmethod
    def create(
        cls,
        status: SystemStatus = "running",
        details: str = "",
        *,
        source: str = "system",
        cause_id: str | None = None,
    ) -> StatusEvent:
        return cls(
            status=status,
            details=details,
            source=source,
            cause_id=cause_id,
        )


class EventStream:
    """Thread-safe reactive event stream with causal lineage tracking and subscription bus."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._events: list[EventStreamEvent] = []
        self._by_id: dict[str, EventStreamEvent] = {}
        self._seq_counter: int = 0
        self._subscribers: dict[
            str,
            tuple[
                Callable[[EventStreamEvent], None],
                Callable[[EventStreamEvent], bool] | None,
            ],
        ] = {}

    def publish(self, event: EventStreamEvent) -> EventStreamEvent:
        """Publish an event to the stream, assigning a monotonic sequence number."""
        with self._lock:
            self._seq_counter += 1
            stamped = dataclasses.replace(event, seq=self._seq_counter)
            self._events.append(stamped)
            self._by_id[stamped.id] = stamped
            subscribers_snapshot = list(self._subscribers.values())

        for callback, filter_fn in subscribers_snapshot:
            try:
                if filter_fn is None or filter_fn(stamped):
                    callback(stamped)
            except Exception:
                # Observers must not crash the stream
                pass

        return stamped

    def publish_action(
        self,
        action_type: str,
        params: dict[str, Any] | None = None,
        *,
        source: str = "agent",
        cause_id: str | None = None,
        timeout: float | None = None,
    ) -> Action:
        act = Action.create(
            action_type,
            params=params,
            source=source,
            cause_id=cause_id,
            timeout=timeout,
        )
        return self.publish(act)  # type: ignore[return-value]

    def publish_observation(
        self,
        observation_type: str,
        content: str = "",
        *,
        exit_code: int | None = None,
        status: ObservationStatus = "success",
        source: str = "environment",
        cause_id: str | None = None,
    ) -> Observation:
        obs = Observation.create(
            observation_type,
            content=content,
            exit_code=exit_code,
            status=status,
            source=source,
            cause_id=cause_id,
        )
        return self.publish(obs)  # type: ignore[return-value]

    def publish_status(
        self,
        status: SystemStatus = "running",
        details: str = "",
        *,
        source: str = "system",
        cause_id: str | None = None,
    ) -> StatusEvent:
        ev = StatusEvent.create(status, details=details, source=source, cause_id=cause_id)
        return self.publish(ev)  # type: ignore[return-value]

    def subscribe(
        self,
        callback: Callable[[EventStreamEvent], None],
        filter_fn: Callable[[EventStreamEvent], bool] | None = None,
    ) -> str:
        """Register a subscriber callback. Returns subscription handle id."""
        sub_id = str(uuid.uuid4())
        with self._lock:
            self._subscribers[sub_id] = (callback, filter_fn)
        return sub_id

    def unsubscribe(self, sub_id: str) -> bool:
        """Unregister a subscriber callback."""
        with self._lock:
            return self._subscribers.pop(sub_id, None) is not None

    def get_events(
        self,
        since_seq: int = 0,
        event_type: type | None = None,
    ) -> list[EventStreamEvent]:
        """Query events starting after since_seq with optional type filter."""
        with self._lock:
            res: list[EventStreamEvent] = []
            for ev in self._events:
                if ev.seq > since_seq:
                    if event_type is None or isinstance(ev, event_type):
                        res.append(ev)
            return res

    def get_event_by_id(self, event_id: str) -> EventStreamEvent | None:
        with self._lock:
            return self._by_id.get(event_id)

    def get_causal_chain(self, event_id: str) -> list[EventStreamEvent]:
        """Traverse backwards via cause_id links and return chronological root-to-leaf lineage."""
        with self._lock:
            chain: list[EventStreamEvent] = []
            curr_id: str | None = event_id
            visited: set[str] = set()

            while curr_id and curr_id not in visited:
                visited.add(curr_id)
                ev = self._by_id.get(curr_id)
                if not ev:
                    break
                chain.append(ev)
                curr_id = ev.cause_id

            chain.reverse()
            return chain

    def replay(self, since_seq: int = 0) -> list[EventStreamEvent]:
        """Replay all events in sequential order."""
        return self.get_events(since_seq=since_seq)

    def count(self) -> int:
        with self._lock:
            return len(self._events)

    def clear(self) -> None:
        with self._lock:
            self._events.clear()
            self._by_id.clear()
            self._seq_counter = 0
            self._subscribers.clear()
