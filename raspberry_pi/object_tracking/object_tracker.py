from utils import BoundingBox, DetectionResult
from collections import namedtuple

Tracklet = namedtuple('Tracklet', 'xdir ydir zdir')
# xdir is lateral movement, one of (l)eft, (r)ight or (s)tatic
# ydir is veritical movement, one of (u)p, (d)own or (s)tatic
# zdir is z-axis moveent, one of (i)ncoming, (o)utgoing or (s)tatic

def mergeTracklets(tracklets):
    # Merge any very similar tracks into single track
    # If windw_size tracklets differ only in 1 dimension, retain only the latest one and remove older tracklet
    if len(tracklets) <= 1:
        return tracklets

    def differ_by_2dim(tr1, tr2):
        if (tr1.xdir == tr2.xdir and tr1.ydir == tr2.ydir 
           or tr1.ydir == tr2.ydir and tr1.zdir == tr2.zdir
           or tr1.xdir == tr2.xdir and tr1.zdir == tr2.zdir):
            return True
        return False

    i = -1
    while abs(i) < len(tracklets):
        last = tracklets[i]
        penum = tracklets[i-1]
        if differ_by_2dim(last, penum):
            # Retain only last
            del tracklets[i-1]
        else:
            i -= 1
    
    if len(tracklets) > 1:
        # Remove any completely static tracks
        tracklets = [x for x in tracklets if not x == Tracklet('s', 's', 's')]
    return tracklets

def getDirections(tracklets):
    DIR_MAP = {'s': 'Still', 'l': 'Left', 'r': 'Right', 'u': 'Up', 'd': 'Down', 'i': 'Inwards', 'o': 'Outwards'}
    directions = []
    for xdir,ydir,zdir in tracklets:
        directions.append('Horizontally {}, Vertically {}, and {}'.format(
                DIR_MAP[xdir], DIR_MAP[ydir], DIR_MAP[zdir]))

    return directions

class Tracker:
    def __init__(self, initBB, max_frames=100, window_size=2):
        self.frames = [initBB]
        self.tracklets = []
        self.max_frames = 100
        self.window_size = 2 # Direction for every 2 frames

    def __repr__(self):
        return 'Tracking Window Size: {}, Current Num Frames: {}, Total Tracklets: {}'.format(
                        self.window_size, len(self.frames), len(self.tracklets))

    def addFrame(self, newBB):
        # Retain only max_frames frames, after which remove older frames
        if len(self.frames) == self.max_frames:
            self.frames = self.frames[1:]
        self.frames.append(newBB)
        self._updateTrack()

    def _updateTrack(self):
        # Get last frame
        # Check direction - if in keeping, continue same tracklet
        # If change in direction, add new tracklet
        cur = -1
        lst = max(-len(self.frames), -self.window_size - 1)

        curframe = self.frames[cur]
        lastframe = self.frames[lst]

        last_x, last_y = self._getBboxCenter(lastframe)
        cur_x, cur_y = self._getBboxCenter(curframe)
        old_area = self._getBboxArea(lastframe)
        new_area = self._getBboxArea(curframe)

        xdir,ydir,zdir = self._getDirections(last_x, last_y, cur_x, cur_y, old_area, new_area)

        newtrack = Tracklet(xdir, ydir, zdir)
        if len(self.tracklets) > 0:
            oldtrack = self.tracklets[-1]
        else:
            self.tracklets.append(newtrack)
            return
        if oldtrack == newtrack:
            pass
        else:
            self.tracklets.append(newtrack)

    def _getBboxCenter(self, bbox):
        return bbox.xmin + (bbox.xmax - bbox.xmin)/2.0, bbox.ymin + (bbox.ymax - bbox.ymin)/2.0

    def _getBboxArea(self, bbox):
        # l * b
        return (bbox.xmax - bbox.xmin) * (bbox.ymax - bbox.ymin)

    def _getDirections(self, x1, y1, x2, y2, a1, a2):
        # Set a small buffer for movement, so that tracklets are not too fine
        buffer = 12.0 # 5 pixel buffer. For 300,300 image, approx 2%
        area_buffer = 1.10 # 5%
        xdir = ydir = zdir = 's'

        if x2 > x1 and (x2 - x1) > buffer:
            xdir = 'r' # right
        elif x1 > x2 and (x1 - x2) > buffer:
            xdir = 'l' # left

        if y2 > y1 and (y2 - y1) > buffer:
            ydir = 'd' # down
        elif y1 > y2 and (y1 - y2) > buffer:
            ydir = 'u' # up

        if a1 <= 0 or a2 <= 0:
            print('Invalid areas!: {}, {}'.format(a1, a2))
            zdir = None
        elif a1 > a2 and a1/a2 > area_buffer:
            zdir = 'o' # out going
        elif a2 > a1 and a2/a1 > area_buffer:
            zdir = 'i' # incoming

        return xdir, ydir, zdir

    def getTracklets(self):
        self.tracklets = mergeTracklets(self.tracklets)
        return self.tracklets
