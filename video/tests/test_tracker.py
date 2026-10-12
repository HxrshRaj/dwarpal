from video.tracker import IouTracker, iou


def test_iou_identical_and_disjoint():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_id_persists_across_frames_and_new_object_gets_new_id():
    t = IouTracker()
    a = t.update(0, [[0, 0, 10, 10]])
    b = t.update(3, [[2, 0, 12, 10], [100, 100, 120, 120]])
    assert b[0] == a[0]
    assert b[1] != a[0]


def test_track_dropped_after_max_gap():
    t = IouTracker(max_gap=1)
    a = t.update(0, [[0, 0, 10, 10]])
    b = t.update(5, [[0, 0, 10, 10]])
    assert a[0] != b[0]


def test_two_objects_do_not_swap_ids():
    t = IouTracker()
    a = t.update(0, [[0, 0, 10, 10], [50, 0, 60, 10]])
    b = t.update(1, [[51, 0, 61, 10], [1, 0, 11, 10]])
    assert b[1] == a[0] and b[0] == a[1]
