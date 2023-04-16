import cv2
import numpy as np
import plotly.colors as colors
from skimage import draw

COLOR_LIST = [colors.hex_to_rgb(c) for c in colors.qualitative.Light24]


class Grid_Renderer:
    def __init__(self, grid_size, color_map, img_size, background_color=(0, 0, 0), obj_size=None):
        self.grid_size = int(grid_size)
        self.img_size = img_size
        self.background_color = background_color
        self.color_map = color_map
        self.obj_size = obj_size

    def render_nd_grid(self, grid, flip=False):
        """
        The grid is supposed to be in this format np.shape([h, w, n_objects]), the channel idx
            is supposed to be the object id
        """
        if flip:
            grid = np.swapaxes(grid, 0, 1)
            grid = np.flip(grid, axis=0)
        g_s = self.grid_size
        h, w = grid.shape[0], grid.shape[1]
        img = np.zeros(shape=(int(g_s * h), int(g_s * w), 3))
        img_shape = (img.shape[0], img.shape[1])
        for x in range(h):
            for y in range(w):
                if grid[x, y, :].any():
                    object_id = grid[x, y, :].argmax()
                    if object_id in self.color_map and object_id in self.obj_size:
                        rr, cc = draw.disk((x * g_s, y * g_s), self.obj_size[object_id], shape=img_shape)
                        img[rr, cc, :] = self.color_map[object_id]
        return cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA).astype(np.uint8)

    def render_2d_grid(self, grid, flip=False):
        if flip:
            grid = np.swapaxes(grid, 0, 1)
            grid = np.flip(grid, axis=0)
        g_s = self.grid_size
        h, w = grid.shape[0], grid.shape[1]
        img = np.zeros(shape=(int(g_s * h), int(g_s * w), 3))
        img_shape = (img.shape[0], img.shape[1])
        for x in range(h):
            for y in range(w):
                object_id = grid[x, y]
                if object_id in self.color_map:
                    if object_id in self.color_map and object_id in self.obj_size:
                        rr, cc = draw.disk((x * g_s, y * g_s), self.obj_size[object_id], shape=img_shape)
                        img[rr, cc, :] = self.color_map[object_id]
        return img.astype(np.uint8)


def main():
    # define grid
    grid = np.zeros(shape=(100, 100, 4), dtype=np.int8)
    # define color map
    color_map = {1: [255, 0, 0], 2: [0, 255, 0], 3: [0, 0, 255]}
    obj_size = {1: 2, 2: 2, 3: 2}
    # a moving snake
    body_id = 1
    n_link = 15
    # compute y axes
    x_axes = [20 + i * int(60 / n_link) for i in range(n_link)]
    y_axis = 50

    # init grid
    for i in range(n_link):
        grid[x_axes[i], y_axis, body_id] = 1

    grid_renderer = Grid_Renderer(grid_size=20, color_map=color_map,
                                  img_size=64, obj_size=obj_size)
    import cv2
    grid_img = grid_renderer.render_nd_grid(grid)
    cv2.imshow('grid render', cv2.cvtColor(grid_img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
