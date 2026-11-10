"""Unit tests for OpenHands-style reactive immutable EventStream."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
import pytest

from deerflow.runtime.events import (
    Action,
    EventStream,
    EventStreamEvent,
    Observation,
    StatusEvent,
)


class TestEventImmutability:
    def test_action_is_frozen(self):
        act = Action.create("run_cmd", {"command": "ls -la"})
        with pytest.raises(FrozenInstanceError):
            act.action_type = "read_file"  # type: ignore[misc]

    def test_observation_is_frozen(self):
        obs = Observation.create("cmd_output", "file1.txt\nfile2.txt", exit_code=0)
        with pytest.raises(FrozenInstanceError):
            obs.content = "mutated"  # type: ignore[misc]

    def test_serialization(self):
        act = Action.create("file_read", {"path": "main.py"})
        d = act.to_dict()
        assert d["action_type"] == "file_read"
        assert d["params"] == {"path": "main.py"}
        assert d["source"] == "agent"


class TestEventStreamPubSub:
    def test_monotonic_sequences(self):
        stream = EventStream()
        a1 = stream.publish_action("read_file", {"path": "a.txt"})
        o1 = stream.publish_observation("file_content", "hello", cause_id=a1.id)
        s1 = stream.publish_status("running", details="processing")

        assert a1.seq == 1
        assert o1.seq == 2
        assert s1.seq == 3
        assert stream.count() == 3

    def test_subscription_and_filtering(self):
        stream = EventStream()
        received_actions: list[EventStreamEvent] = []
        received_errors: list[EventStreamEvent] = []

        # Subscriber for Actions only
        sub1 = stream.subscribe(
            lambda ev: received_actions.append(ev),
            filter_fn=lambda ev: isinstance(ev, Action),
        )

        # Subscriber for Error Observations only
        sub2 = stream.subscribe(
            lambda ev: received_errors.append(ev),
            filter_fn=lambda ev: isinstance(ev, Observation) and ev.status == "error",
        )

        a1 = stream.publish_action("compile", {"flags": "-O2"})
        o1 = stream.publish_observation("cmd_output", "SyntaxError", status="error", cause_id=a1.id)
        o2 = stream.publish_observation("cmd_output", "done", status="success")

        assert len(received_actions) == 1
        assert received_actions[0].id == a1.id

        assert len(received_errors) == 1
        assert received_errors[0].id == o1.id

        # Test unsubscribe
        unsub_ok = stream.unsubscribe(sub1)
        assert unsub_ok
        stream.publish_action("another_action", {})
        assert len(received_actions) == 1  # No new actions received

    def test_subscriber_exception_isolated(self):
        stream = EventStream()
        received: list[EventStreamEvent] = []

        def faulty_subscriber(ev: EventStreamEvent):
            raise RuntimeError("Subscriber crashed!")

        stream.subscribe(faulty_subscriber)
        stream.subscribe(lambda ev: received.append(ev))

        # Publishing should succeed and the second subscriber should still receive
        a = stream.publish_action("ping")
        assert len(received) == 1
        assert received[0].id == a.id


class TestCausalLineage:
    def test_causal_chain_traversal(self):
        stream = EventStream()

        # Build causal chain: User Request -> Action 1 -> Obs 1 -> Action 2 -> Obs 2
        a1 = stream.publish_action("find_files", {"pattern": "*.py"})
        o1 = stream.publish_observation("file_list", "foo.py\nbar.py", cause_id=a1.id)
        a2 = stream.publish_action("read_file", {"path": "foo.py"}, cause_id=o1.id)
        o2 = stream.publish_observation("file_content", "print('hello')", cause_id=a2.id)

        # Unrelated concurrent event
        stream.publish_action("background_ping", {})

        chain = stream.get_causal_chain(o2.id)
        assert len(chain) == 4
        assert [ev.id for ev in chain] == [a1.id, o1.id, a2.id, o2.id]
        assert chain[0] == a1
        assert chain[3] == o2

    def test_replay_and_query(self):
        stream = EventStream()
        a1 = stream.publish_action("step1")
        o1 = stream.publish_observation("obs1")
        a2 = stream.publish_action("step2")

        events_since_1 = stream.get_events(since_seq=1)
        assert len(events_since_1) == 2
        assert events_since_1[0].id == o1.id
        assert events_since_1[1].id == a2.id

        observations_only = stream.get_events(event_type=Observation)
        assert len(observations_only) == 1
        assert observations_only[0].id == o1.id

        replayed = stream.replay()
        assert len(replayed) == 3


class TestConcurrentPublishing:
    def test_multithreaded_publishing_consistency(self):
        stream = EventStream()
        num_threads = 8
        events_per_thread = 25

        def worker(thread_idx: int):
            for i in range(events_per_thread):
                stream.publish_action(f"action_{thread_idx}_{i}")

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker, idx) for idx in range(num_threads)]
            for f in futures:
                f.result()

        total = num_threads * events_per_thread
        assert stream.count() == total
        all_events = stream.replay()
        seqs = [ev.seq for ev in all_events]
        assert seqs == list(range(1, total + 1))
