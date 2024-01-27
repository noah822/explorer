import functools
import itertools
import numpy as np


def _assign_spatial_weight(dist):
    '''
    compute weight associated with distance
    '''
    return 1 - np.abs(dist) 

    

def splat_feature(features, coords, grids):
    '''
    partially adapted from: ____
    splat features to corresponding grids according to their coordinates
    Args:
        Notation: 
        #numPt: the number of points
        #D: dimension of the coordinate system
        #F: the number of feature

        1. features: (..., numPt, F)
        2. coords: (..., numPt, D)
           each coord value is normalized to [-1, 1]
        3. grids: (..., X, Y, Z, ..., F)
           the number axis to splat to should align with D
    Returns:
        None, operation is taken in place
    Complexity:
        O(2^D * numPt * F)
    '''
    D = coords.shape[-1]
    numPt, F = features.shape[-2:]
    grid_size = grids.shape[-(D+1):-1]

    grid_indexes = []
    grid_weights = []

    for i, size in enumerate(grid_size):
       coord_along_i = coords[..., i]  
       coord_along_i = coord_along_i * ((size-1)/2) + ((size-1)/2) # unnormalize from [-1, 1] to real index

       lower_bound = np.floor(coord_along_i) 
       higher_bound = lower_bound + 1
       # clip lower & higher bound index value
       lower_bound[lower_bound < 0] = 0
       higher_bound[higher_bound >= size] = size - 1

       lower_bound_w = _assign_spatial_weight(coord_along_i - lower_bound)
       higher_bound_w = _assign_spatial_weight(higher_bound - coord_along_i)
       grid_indexes.append([lower_bound.astype(np.int32), higher_bound.astype(np.int32)])
       grid_weights.append([lower_bound_w, higher_bound_w])

    

    # get batch number, accepting multi-dimension batch as well 
    batch_shape = grids.shape[:-(D+1)]
    B = functools.reduce(lambda x, y: x*y, batch_shape)
    grids.reshape(B, *grid_size, F)         # (B, X, Y, ..., F)

    for vertex in itertools.product(*[[0, 1]]*D):
        # w: (B, numPt)
        w = functools.reduce(lambda x, y: x*y, [grid_weights[d][i] for d, i in enumerate(vertex)])

        # some hacky approach to select numPt across each batch dimension
        # each grid_indexes[x][y] is of shape (B, numPt), selector will select (B, numPt) points to accumulate feature
        # TODO: come up with neater approach to achieve this, efficient indexing w/o extra memory alloc
        selector = tuple(
            [np.arange(B, dtype=np.int32)[..., np.newaxis].repeat(numPt, axis=-1),
            *[grid_indexes[d][i] for d, i in enumerate(vertex)]]
        )
        np.add.at(grids, selector, w[..., np.newaxis] * features.reshape(B, numPt, F))
    
    grids.reshape(*batch_shape, *grid_size, F)



