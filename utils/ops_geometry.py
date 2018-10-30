from matplotlib.path import Path
from shapely.geometry import Polygon
from shapely.geometry import Point
import numpy as np


def judge_p_in_polygon(p, pts, visualize=True):
    '''
    p: (x, y)
    pts: [(x1, y1), (x2 y2) ...]
    '''
    assert isinstance(pts, (list, tuple, np.ndarray)), ''
    
    # on the line True
    # surface = Polygon(pts)
    # inters = surface.intersection(Point(p))
    # if inter.empty():
    #     return False
    # else:
    #     return True

    # on the line False
    plg = Path(np.array(pts))
    return plg.contains_point(p)




if __name__ == '__main__':

    pts = [(1, 2), (5, 5), (10, 0)]
    pt = (1, 2)
    print(judge_p_in_polygon(pt, pts))