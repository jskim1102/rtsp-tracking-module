from __future__ import annotations

from app.tracking.confirm import RawTrack, TrackConfirmer


def _raw(track_id: int = 7) -> RawTrack:
    return RawTrack(
        track_id=track_id,
        bbox=(10, 20, 110, 220),
        polygon=[(10, 20), (110, 20), (110, 220)],
        score=0.94,
    )


def test_confirmation_requires_consecutive_observations():
    confirmer = TrackConfirmer(confirm_frames=3, ttl_frames=20, on_expire=lambda _: None)

    assert confirmer.observe(1, [_raw()]) == []
    assert confirmer.observe(2, []) == []
    assert confirmer.observe(3, [_raw()]) == []
    assert confirmer.observe(4, [_raw()]) == []
    confirmed = confirmer.observe(5, [_raw()])

    assert len(confirmed) == 1
    assert confirmed[0].track_id == 7
    assert confirmed[0].polygon == [(10, 20), (110, 20), (110, 220)]
    assert confirmed[0].is_new is True


def test_new_track_retries_until_acknowledged():
    confirmer = TrackConfirmer(confirm_frames=1, ttl_frames=20, on_expire=lambda _: None)

    assert confirmer.observe(1, [_raw()])[0].is_new is True
    assert confirmer.observe(2, [_raw()])[0].is_new is True
    confirmer.acknowledge_new_tracks([7, 99])

    assert confirmer.observe(3, [_raw()])[0].is_new is False


def test_expiry_calls_back_and_allows_the_same_id_to_be_new_again():
    expired: list[int] = []
    confirmer = TrackConfirmer(
        confirm_frames=1,
        ttl_frames=2,
        on_expire=expired.append,
    )

    confirmer.observe(1, [_raw()])
    confirmer.acknowledge_new_tracks([7])
    assert confirmer.observe(2, []) == []
    assert confirmer.observe(3, []) == []
    assert confirmer.observe(4, []) == []
    reappeared = confirmer.observe(5, [_raw()])

    assert expired == [7]
    assert reappeared[0].is_new is True


def test_reappearance_before_ttl_preserves_confirmation_without_expiry():
    expired: list[int] = []
    confirmer = TrackConfirmer(
        confirm_frames=2,
        ttl_frames=2,
        on_expire=expired.append,
    )

    assert confirmer.observe(1, [_raw()]) == []
    assert confirmer.observe(2, [_raw()])[0].is_new is True
    confirmer.acknowledge_new_tracks([7])
    assert confirmer.observe(3, []) == []
    retained = confirmer.observe(4, [_raw()])

    assert expired == []
    assert retained[0].is_new is False


def test_reset_clears_confirmation_and_announcement_state():
    confirmer = TrackConfirmer(confirm_frames=2, ttl_frames=20, on_expire=lambda _: None)
    confirmer.observe(1, [_raw()])
    confirmer.observe(2, [_raw()])
    confirmer.acknowledge_new_tracks([7])

    confirmer.reset()

    assert confirmer.observe(1, [_raw()]) == []
    assert confirmer.observe(2, [_raw()])[0].is_new is True
