"""
uses code from the keras preprocessing and adds functionality to transform bounding boxes
"""

import numpy as np
from keras.preprocessing.image import *

def random_rotation_with_boxes(x, rg, boxes, row_axis=1, col_axis=2, channel_axis=0,
                    fill_mode='nearest', cval=0.):
    """Performs a random rotation of a Numpy image tensor. Also rotates the corresponding bounding boxes

    # Arguments
        x: Input tensor. Must be 3D.
        rg: Rotation range, in degrees.
        boxes: a list of bounding boxes [xmin, ymin, xmax, ymax], values in [0,1]
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.

    # Returns
        Rotated Numpy image tensor.
        And rotated bounding boxes
    """

    # sample parameter for augmentation
    theta = np.pi / 180 * np.random.uniform(-rg, rg)

    # apply to image
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])

    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)

    # apply to vertices
    vertices = boxes_to_vertices(boxes)

    vertices = vertices.reshape((-1, 2))

    # apply offset to have pivot point at [0.5, 0.5]
    vertices -= [0.5, 0.5]

    # apply rotation, we only need the rotation part of the matrix
    vertices = np.dot(vertices, rotation_matrix[:2, :2])
    vertices += [0.5, 0.5]

    boxes = vertices_to_boxes(vertices)

    return x, boxes

def boxes_to_vertices(boxes):
    """
    Takes a list of bounding boxes and creates a list of vertices
    The output shape is [number of boxes, 4, 2]
    4 for the 4 vertices,
    2 for x/y
    :param boxes: Input tensor, must be 2D
    :return: output tensor, 3D
    """

    assert len(boxes.shape) == 2, "boxes must be a 2D tensor"
    assert boxes.shape[1] == 4, "boxes must be [:, 4] tensor"

    min_val = boxes.min()
    max_val = boxes.max()
    assert 0 <= min_val <= 1, "bounding box coordinates must be in [0,1]"
    assert 0 <= max_val <= 1, "bounding box coordinates must be in [0,1]"

    num_boxes = boxes.shape[0]

    # look at the four vertices of each box
    x_min = boxes[:, 0]
    y_min = boxes[:, 1]
    x_max = boxes[:, 2]
    y_max = boxes[:, 3]

    assert np.all(x_min < x_max), "coordinates must be given as [xmin, ymin, xmax, ymax"
    assert np.all(y_min < y_max), "coordinates must be given as [xmin, ymin, xmax, ymax"

    # create new axis to stack the x,y coordinates
    x_min = np.expand_dims(x_min, axis=-1)
    y_min = np.expand_dims(y_min, axis=-1)
    x_max = np.expand_dims(x_max, axis=-1)
    y_max = np.expand_dims(y_max, axis=-1)

    # stack the x,y coordinates to create the vertices
    # the resulting arrays are indexed [idx of box, idx of x or y]
    up_left = np.concatenate([x_min, y_min], axis=-1)
    up_right = np.concatenate([x_min, y_max], axis=-1)
    down_right = np.concatenate([x_max, y_max], axis=-1)
    down_left = np.concatenate([x_max, y_min], axis=-1)

    # now stack the vertices, along axis 1
    up_left = np.expand_dims(up_left, axis=1)
    up_right = np.expand_dims(up_right, axis=1)
    down_right = np.expand_dims(down_right, axis=1)
    down_left = np.expand_dims(down_left, axis=1)

    # create an array of all vertices, of all boxes
    # the shape is [number of boxes, number of vertices, number of coordinates]
    # ->  shape is [number of boxes, 4, 2]
    vertices = np.concatenate([up_left, up_right, down_right, down_left], axis=1)

    return vertices

def vertices_to_boxes(vertices):
    """
    Takes a list of vertices and converts them to bounding boxes
    :param vertices: Input tensor, must be 2D
    :return: output tensor, 2D
    """

    assert len(vertices.shape)==2, "vertices must be a 2D tensor"
    assert vertices.shape[1]==2, "vertices must be [:, 2] tensor"

    vertices = vertices.reshape((-1, 4, 2))

    x = vertices[:, :, 0]
    y = vertices[:, :, 1]

    x_min = x.min(axis=-1)
    x_max = x.max(axis=-1)
    y_min = y.min(axis=-1)
    y_max = y.max(axis=-1)

    x_min = np.expand_dims(x_min, axis=-1)
    x_max = np.expand_dims(x_max, axis=-1)
    y_min = np.expand_dims(y_min, axis=-1)
    y_max = np.expand_dims(y_max, axis=-1)

    boxes = np.concatenate([x_min, y_min, x_max, y_max], axis=-1)

    return boxes
