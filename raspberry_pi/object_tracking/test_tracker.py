from utils import BoundingBox
from object_tracker import Tracker

# Lateral left
def test_lateral_left():
    prev = BoundingBox(0,50,12,62)
    tracker = Tracker(prev)
    # Translate box left
    for _ in range(10):
        bbox = BoundingBox(prev.ymin, prev.xmin-3, prev.ymax, prev.xmax-3)
        prev = bbox

        tracker.addFrame(bbox)

    tracks = tracker.getTracklets()
    assert len(tracks) == 1, 'Expected only 1 contiguous track, got: {}'.format(tracks)
    assert tracks[0].xdir == 'l', 'Expected xdir {} to be l'.format(tracks[0].xdir)
    assert tracks[0].ydir == 's', 'Expected ydir {} to be s'.format(tracks[0].ydir)
    assert tracks[0].zdir == 's', 'Expected zdir {} to be s'.format(tracks[0].zdir)

# Lateral right
def test_lateral_right():
    prev = BoundingBox(0,50,12,62)
    tracker = Tracker(prev)
    # Translate box rigth
    for _ in range(10):
        bbox = BoundingBox(prev.ymin, prev.xmin+3, prev.ymax, prev.xmax+3)
        prev = bbox

        tracker.addFrame(bbox)

    tracks = tracker.getTracklets()
    assert len(tracks) == 1, 'Expected only 1 contiguous track, got: {}'.format(tracks)
    assert tracks[0].xdir == 'r', 'Expected xdir {} to be r'.format(tracks[0].xdir)
    assert tracks[0].ydir == 's', 'Expected ydir {} to be s'.format(tracks[0].ydir)
    assert tracks[0].zdir == 's', 'Expected zdir {} to be s'.format(tracks[0].zdir)

# Vertical Up
def test_vertical_up():
    prev = BoundingBox(100,50,150,62)
    tracker = Tracker(prev)
    # Translate box up
    for _ in range(10):
        bbox = BoundingBox(prev.ymin-3, prev.xmin, prev.ymax-3, prev.xmax)
        prev = bbox

        tracker.addFrame(bbox)

    tracks = tracker.getTracklets()
    assert len(tracks) == 1, 'Expected only 1 contiguous track, got: {}'.format(tracks)
    assert tracks[0].xdir == 's', 'Expected xdir {} to be s'.format(tracks[0].xdir)
    assert tracks[0].ydir == 'u', 'Expected ydir {} to be u'.format(tracks[0].ydir)
    assert tracks[0].zdir == 's', 'Expected zdir {} to be s'.format(tracks[0].zdir)

# Vertical Down
def test_vertical_down():
    prev = BoundingBox(100,50,150,62)
    tracker = Tracker(prev)
    # Translate box down
    for _ in range(10):
        bbox = BoundingBox(prev.ymin+3, prev.xmin, prev.ymax+3, prev.xmax)
        prev = bbox

        tracker.addFrame(bbox)

    tracks = tracker.getTracklets()
    assert len(tracks) == 1, 'Expected only 1 contiguous track, got: {}'.format(tracks)
    assert tracks[0].xdir == 's', 'Expected xdir {} to be s'.format(tracks[0].xdir)
    assert tracks[0].ydir == 'd', 'Expected ydir {} to be d'.format(tracks[0].ydir)
    assert tracks[0].zdir == 's', 'Expected zdir {} to be s'.format(tracks[0].zdir)

# Incoming
def test_incoming():
    prev = BoundingBox(50,50,92,62)
    tracker = Tracker(prev)
    # Bpx gets bigger every iter
    for _ in range(10):
        bbox = BoundingBox(prev.ymin-1, prev.xmin-1, prev.ymax+1, prev.xmax+1)
        prev = bbox

        tracker.addFrame(bbox)

    tracks = tracker.getTracklets()
    assert len(tracks) == 1, 'Expected only 1 contiguous track, got: {}'.format(tracks)
    assert tracks[0].xdir == 's', 'Expected xdir {} to be l'.format(tracks[0].xdir)
    assert tracks[0].ydir == 's', 'Expected ydir {} to be s'.format(tracks[0].ydir)
    assert tracks[0].zdir == 'i', 'Expected zdir {} to be i'.format(tracks[0].zdir)

# Outgoing
def test_outgoing():
    prev = BoundingBox(20,20,82,62)
    tracker = Tracker(prev)
    # Box smaller every iter
    for _ in range(10):
        bbox = BoundingBox(prev.ymin+1, prev.xmin+1, prev.ymax-1, prev.xmax-1)
        prev = bbox

        tracker.addFrame(bbox)

    tracks = tracker.getTracklets()
    assert len(tracks) == 1, 'Expected only 1 contiguous track, found {}'.format(tracks)
    assert tracks[0].xdir == 's', 'Expected xdir {} to be l'.format(tracks[0].xdir)
    assert tracks[0].ydir == 's', 'Expected ydir {} to be s'.format(tracks[0].ydir)
    assert tracks[0].zdir == 'o', 'Expected zdir {} to be r'.format(tracks[0].zdir)

# Angling Left Up
def test_left_up():
    prev = BoundingBox(50,50,102,62)
    tracker = Tracker(prev)
    # Translate box up and left
    for _ in range(10):
        bbox = BoundingBox(prev.ymin-3, prev.xmin-3, prev.ymax-3, prev.xmax-3)
        prev = bbox

        tracker.addFrame(bbox)

    tracks = tracker.getTracklets()
    assert len(tracks) == 1, 'Expected only 1 contiguous track, got: {}'.format(tracks)
    assert tracks[0].xdir == 'l', 'Expected xdir {} to be l'.format(tracks[0].xdir)
    assert tracks[0].ydir == 'u', 'Expected ydir {} to be u'.format(tracks[0].ydir)
    assert tracks[0].zdir == 's', 'Expected zdir {} to be s'.format(tracks[0].zdir)

# Angling right Down
def test_right_down():
    prev = BoundingBox(35,50,102,62)
    tracker = Tracker(prev)
    # Translate box right and down
    for _ in range(10):
        bbox = BoundingBox(prev.ymin+3, prev.xmin+3, prev.ymax+3, prev.xmax+3)
        prev = bbox

        tracker.addFrame(bbox)

    tracks = tracker.getTracklets()
    assert len(tracks) == 1, 'Expected only 1 contiguous track, got: {}'.format(tracks)
    assert tracks[0].xdir == 'r', 'Expected xdir {} to be r'.format(tracks[0].xdir)
    assert tracks[0].ydir == 'd', 'Expected ydir {} to be d'.format(tracks[0].ydir)
    assert tracks[0].zdir == 's', 'Expected zdir {} to be s'.format(tracks[0].zdir)

# Angling Left Down
def test_left_down():
    prev = BoundingBox(50,50,102,62)
    tracker = Tracker(prev)
    # Translate left and down
    for _ in range(10):
        bbox = BoundingBox(prev.ymin+3, prev.xmin-3, prev.ymax+3, prev.xmax-3)
        prev = bbox

        tracker.addFrame(bbox)

    tracks = tracker.getTracklets()
    assert len(tracks) == 1, 'Expected only 1 contiguous track, got: {}'.format(tracks)
    assert tracks[0].xdir == 'l', 'Expected xdir {} to be l'.format(tracks[0].xdir)
    assert tracks[0].ydir == 'd', 'Expected ydir {} to be d'.format(tracks[0].ydir)
    assert tracks[0].zdir == 's', 'Expected zdir {} to be s'.format(tracks[0].zdir)

# Angling right Up
def test_right_up():
    prev = BoundingBox(35,50,62,62)
    tracker = Tracker(prev)
    # Translate box right and up
    for _ in range(10):
        bbox = BoundingBox(prev.ymin-3, prev.xmin+3, prev.ymax-3, prev.xmax+3)
        prev = bbox

        tracker.addFrame(bbox)

    tracks = tracker.getTracklets()
    assert len(tracks) == 1, 'Expected only 1 contiguous track, got: {}'.format(tracks)
    assert tracks[0].xdir == 'r', 'Expected xdir {} to be r'.format(tracks[0].xdir)
    assert tracks[0].ydir == 'u', 'Expected ydir {} to be u'.format(tracks[0].ydir)
    assert tracks[0].zdir == 's', 'Expected zdir {} to be s'.format(tracks[0].zdir)

# Angling right in down
def test_right_down_in():
    prev = BoundingBox(55,60,129,151)
    tracker = Tracker(prev)
    for x in range(16):
        bbox = BoundingBox(prev.ymin+3, prev.xmin+3, prev.ymax+3, prev.xmax+3) # right down
        bbox = BoundingBox(bbox.ymin-1, bbox.xmin-1, bbox.ymax+1, bbox.xmax+1) # incoming
        prev = bbox

        tracker.addFrame(bbox)

    tracks = tracker.getTracklets()
    assert len(tracks) == 1, 'Expected only 1 contiguous track, got: {}'.format(tracks)
    assert tracks[0].xdir == 'r', 'Expected xdir {} to be r'.format(tracks[0].xdir)
    assert tracks[0].ydir == 'd', 'Expected ydir {} to be d'.format(tracks[0].ydir)
    assert tracks[0].zdir == 'i', 'Expected zdir {} to be i'.format(tracks[0].zdir)

# Angling right out up
def test_right_up_out():
    prev = BoundingBox(60,50,106,99)
    tracker = Tracker(prev)
    for x in range(16):
        bbox = BoundingBox(prev.ymin-3, prev.xmin+3, prev.ymax-3, prev.xmax+3) # right up
        bbox = BoundingBox(bbox.ymin+1, bbox.xmin+1, bbox.ymax-1, bbox.xmax-1) # out

        prev = bbox

        tracker.addFrame(bbox)

    tracks = tracker.getTracklets()
    assert len(tracks) == 1, 'Expected only 1 contiguous track, got: {}'.format(tracks)
    assert tracks[0].xdir == 'r', 'Expected xdir {} to be r'.format(tracks[0].xdir)
    assert tracks[0].ydir == 'u', 'Expected ydir {} to be u'.format(tracks[0].ydir)
    assert tracks[0].zdir == 'o', 'Expected zdir {} to be o'.format(tracks[0].zdir)

# Angling left in down
def test_left_down_in():
    prev = BoundingBox(50,50,150,150)
    tracker = Tracker(prev)
    for x in range(16):
        bbox = BoundingBox(prev.ymin+3, prev.xmin-3, prev.ymax+3, prev.xmax-3) # Left down
        bbox = BoundingBox(bbox.ymin-1, bbox.xmin-1, bbox.ymax+1, bbox.xmax+1) # in

        prev = bbox
        tracker.addFrame(bbox)

    tracks = tracker.getTracklets()
    assert len(tracks) == 1, 'Expected only 1 contiguous track, got: {}'.format(tracks)
    assert tracks[0].xdir == 'l', 'Expected xdir {} to be l'.format(tracks[0].xdir)
    assert tracks[0].ydir == 'd', 'Expected ydir {} to be d'.format(tracks[0].ydir)
    assert tracks[0].zdir == 'i', 'Expected zdir {} to be i'.format(tracks[0].zdir)

# Angling left out up
def test_left_up_out():
    prev = BoundingBox(50,50,100,100)
    tracker = Tracker(prev)
    for x in range(16):
        bbox = BoundingBox(prev.ymin-3, prev.xmin-3, prev.ymax-3, prev.xmax-3) # Left up
        bbox = BoundingBox(bbox.ymin+1, bbox.xmin+1, bbox.ymax-1, bbox.xmax-1) # out
        prev = bbox

        tracker.addFrame(bbox)

    tracks = tracker.getTracklets()
    assert len(tracks) == 1, 'Expected only 1 contiguous track, got: {}'.format(tracks)
    assert tracks[0].xdir == 'l', 'Expected xdir {} to be l'.format(tracks[0].xdir)
    assert tracks[0].ydir == 'u', 'Expected ydir {} to be u'.format(tracks[0].ydir)
    assert tracks[0].zdir == 'o', 'Expected zdir {} to be o'.format(tracks[0].zdir)

# Static
def test_static():
    prev = BoundingBox(0,50,12,62)
    tracker = Tracker(prev)
    # No Translate
    for _ in range(10):
        bbox = BoundingBox(prev.ymin, prev.xmin, prev.ymax, prev.xmax)
        prev = bbox

        tracker.addFrame(bbox)

    tracks = tracker.getTracklets()
    assert len(tracks) == 1, 'Expected only 1 contiguous track, got: {}'.format(tracks)
    assert tracks[0].xdir == 's', 'Expected xdir {} to be s'.format(tracks[0].xdir)
    assert tracks[0].ydir == 's', 'Expected ydir {} to be s'.format(tracks[0].ydir)
    assert tracks[0].zdir == 's', 'Expected zdir {} to be s'.format(tracks[0].zdir)

if __name__ == '__main__':
    funcs = [test_lateral_left, test_lateral_right, test_vertical_up, test_vertical_down,
             test_incoming, test_outgoing, test_static, test_left_down, test_right_down,
             test_right_up, test_left_down, test_left_down_in, test_left_up_out,
             test_right_down_in, test_right_up_out]

    print('Running total {} tests..'.format(len(funcs)))
    failed_funcs = []
    for f in funcs:
        try:
            f()
        except AssertionError as e:
            print('Test: {} failed: {}'.format(f.__name__, e))
            failed_funcs.append(f.__name__)

    print('------- Summary --------')
    print('Total Tests: {}, Passed: {}, Failed: {}'.format(len(funcs), len(funcs) - len(failed_funcs), len(failed_funcs)))
    if len(failed_funcs) > 0:
        print('Failed: {}'.format(failed_funcs))
