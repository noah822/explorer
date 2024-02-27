import copy
import cv2
import numpy as np

from dataclasses import dataclass
from pathlib import Path
from scipy import ndimage
from typing import Any, List, Union, Optional

from .constants import TOP_DOWN_MAP_COLOR_MAP

DEFAULT_SPRITE_PATH = (Path(__file__).parent.absolute() / 'icon' / '100x100.png').as_posix()
DEFAULT_SPRITE_SIZE = [30, 30]

__all__ = ['TopDownMap']

# TODO: scatter new pixel

@dataclass
class SpriteState:
    content: np.ndarray
    covered_patch: Optional[np.ndarray]
    x: int
    y: int
    rot: float

    

class TopDownMap:
    '''
    Provide efficient top-down-map update 
    Conventions:
    throughout this class, rotation is in unit of degree
    '''
    RAW_MAP_DTYPE = np.uint8
    def __init__(self,
                 size: List[int],
                 sprite: Union[str, np.ndarray]=DEFAULT_SPRITE_PATH,
                 sprite_size: List[int]=None,
                 track_trajactory: bool=True):
        assert len(size) == 2, 'top down map can only be 2D'

        if type(sprite) == str:
            # save in BGR channel space
            sprite_content = cv2.imread(sprite)
        elif type(sprite) == np.ndarray:
            sprite_content = sprite
        else:
            raise TypeError('sprite of the agent could only be of type str or ndarray' )

        self.map = np.zeros((*size, 3), dtype=TopDownMap.RAW_MAP_DTYPE)
        self.map[:,:] = np.array(TOP_DOWN_MAP_COLOR_MAP['unexplorered'])
        self._map_size = size 

        if sprite_size is None:
            sprite_size = DEFAULT_SPRITE_SIZE

        # wrap into internal sprite struct
        self.sprite = SpriteState(
            cv2.resize(sprite_content[...,:3], sprite_size), None,
            int(size[0]/2), int(size[1]/2), 0 
        )

        self._track_trajactory = track_trajactory

    
    def init_sprite_at(self, pos: List[int], rot: float):
        # by default, sprite heads towards +y initially
        (self.sprite.x, self.sprite.y), self.sprite.rot = pos, rot

        covered_patch = self._draw_patch(self.sprite.content, pos, rot)
        if covered_patch is None:
            raise ValueError(
                f'Invalid initial position of sprite\n' \
                f'It should be in the range of the map, which has size {self.map.shape[:2]}'
            )
        else:
            self.sprite.covered_patch = covered_patch


    def scatter_pixel_by_index(self, content, index):
        '''
        scatter pixels over the maintained map to locations specified by indexes array
        Args:
            content: (h, w)
            new pixel value to scatter, this should follow cv2's BGR channel ordering convention
            index: (h, w, 2)
        '''
        map_h, map_w = self._map_size
        x_index, y_index = index[...,0], index[...,1]

        # guard array indexing
        x_index = np.clip(x_index, 0, map_h-1); y_index = np.clip(y_index, 0, map_w-1)
        self.map[x_index, y_index] = content

    
    def move_sprite(self, dxy, dtheta):
        '''
        reflect the change of sprite state onto the underlying map
        Args:
            dxy: delta value of agent position in index
            dtheta: delta value of agent rotation in degree
        '''
        cached_patch = self.sprite.covered_patch

        if cached_patch is not None:
            # recover previous covered patch
            self._draw_patch(cached_patch, [self.sprite.x, self.sprite.y], 0)

        # tentatively update sprite's position        
        dx, dy = dxy 
        covered_patch = self._draw_patch(
            self.sprite.content, [self.sprite.x + dx, self.sprite.y + dy], self.sprite.rot + dtheta
        )
        if covered_patch is not None:
            # update success
            if self._track_trajactory:
                # swap x, y to accommodate cv2 convention
                cv2.line(
                    self.map, [self.sprite.y, self.sprite.x], [self.sprite.y + dy, self.sprite.x + dx],
                    (0, 0, 255)
                )
            self.sprite.x += dx; self.sprite.y += dy; self.sprite.rot += dtheta
            self.sprite.covered_patch = covered_patch

    
    def _draw_patch(self, content, pos, rot) -> Optional[np.ndarray]:
        '''
        Convention:
        x is aliased to w
        y is aliased to h
        +x points up; +y points right
        
        Args:
            content: array of pixels to patch onto the map
            pos: where to patch the content onto the map. 
            It denotes the location of center of the content patch
            if part of the content exceeds the boundary of the map
            map will not be updated
        Returns:
            if content is successfully patched onto the map
            the region it covers will be returned
        '''
        w, h = content.shape[:2]
        cx, cy = pos
        l, r = cx-int(w/2), cy-int(h/2)
        content = ndimage.rotate(content, angle=rot, reshape=False)

        if self._in_boundary(l, r) and self._in_boundary(l+w-1, r+h-1):
            covered_patch = copy.deepcopy(self.map[l:l+w, r:r+h])
            # if successful, mark sprite location as explorered
            self.map[l:l+w, r:r+h] = content
            self.map[cx, cy] = TOP_DOWN_MAP_COLOR_MAP['explorered']
            return covered_patch
        else:
            return None
    
    def _in_boundary(self, xx, yy):
        w, h = self.map.shape[:2]
        return (xx >= 0) and (xx < w) and (yy >= 0) and (yy < h)
       

    
    @property
    def raw_map(self) -> np.ndarray:
        return self.map
