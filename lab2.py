from math import sqrt

import pandas as pd
import numpy as np
from random import uniform
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


PAD_EMOTIONS = {
    "Engagement": np.array([1, 1, 1]),
    "Excitement": np.array([1, 1, -1]),
    "Stress":     np.array([-1, 1, -1]),
    "Relaxation": np.array([1, -1, 1]),
    "Interest":   np.array([1, 1, -1]),
    "Focus":      np.array([1, 1, -1]),
}


def get_pad_from_csv(filename: str) -> pd.DataFrame:
    eye_brain_df = pd.read_csv(filename)
    pad = np.zeros((len(eye_brain_df), 3))

    for ix, row in eye_brain_df.iterrows():
        pad_vec = np.zeros(3)
        for k, v in PAD_EMOTIONS.items():
            raw = row[f"PM.{k}.Scaled"]
            if np.isnan(raw):
                continue
            pad_vec += raw * v
        pad[ix] = np.array(pad_vec)
    return pd.DataFrame(pad, columns=['X', 'Y', 'Z'])  # return PAD values


def _gen_point(minimum: float = 0, maximum: float = 1):
    x = uniform(minimum, maximum)
    y = uniform(minimum, maximum)
    z = uniform(minimum, maximum)
    return np.array([x, y, z])


def get_points(minimum: float = 0, maximum: float = 1, variation=0.2, split=100, count=100):
    assert split != 0, "Split can't be zero."
    
    points = np.zeros((count, 3))
    split_size = count // split
    centers = np.zeros((split_size, 3))
    for ix in range(len(centers)):
        centers[ix] = _gen_point(minimum, maximum)
    for ix in range(len(centers)):
        for jx in range(ix * split_size, (ix + 1) * split_size):
            points[jx] = centers[ix] + _gen_point(-variation, variation)
    return points


class Plotter:
    def __init__(self, data):
        self.data = data
        self.fig = plt.figure()
        self.fig.subplots_adjust(bottom=0.25)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.ax = self.fig.add_subplot(111, projection='3d')

        self._time_range = int(sqrt(len(data)))
        self._time_center_a = self._time_range
        self._time_center_b = self._time_range * 2

        range = self.fig.add_axes([0.25, 0.05, 0.65, 0.03])
        time_a = self.fig.add_axes([0.25, 0.15, 0.65, 0.03])
        time_b = self.fig.add_axes([0.25, 0.1, 0.65, 0.03])

        self.range_slider = Slider(range, 'Range', 1, len(data), valinit=self._time_range, valstep=1)
        self.range_slider.on_changed(self.update_range)

        self.time_slider_a = Slider(time_a, 'Time A', 1, len(data), valinit=self._time_center_a, valstep=1)
        self.time_slider_a.on_changed(self.update_time_a)

        self.time_slider_b = Slider(time_b, 'Time B', 1, len(data), valinit=self._time_center_b, valstep=1)
        self.time_slider_b.on_changed(self.update_time_b)

        data_slice_a = self.get_data(self._time_center_a)
        data_slice_b = self.get_data(self._time_center_b)

        self._plot_a = self.ax.scatter(data_slice_a[:, 0], data_slice_a[:, 1], data_slice_a[:, 2], c='b')
        self._plot_b = self.ax.scatter(data_slice_b[:, 0], data_slice_b[:, 1], data_slice_b[:, 2], c='g')

        c1 = np.average(data_slice_a, axis=0)
        c2 = np.average(data_slice_b, axis=0)
        v = c2 - c1
        self._centroid_a = self.ax.scatter(c1[0], c1[1], c1[2], c='r')
        self._centroid_b = self.ax.scatter(c2[0], c2[1], c2[2], c='r')
        self._line = self.ax.quiver(c1[0], c1[1], c1[2], v[0], v[1], v[2], color='r')

        xmin = np.min(data[:, 0])
        xmax = np.max(data[:, 0])
        ymin = np.min(data[:, 1])
        ymax = np.max(data[:, 1])
        zmin = np.min(data[:, 2])
        zmax = np.max(data[:, 2])
        self.update_limits(xmin, xmax, ymin, ymax, zmin, zmax)

        plt.show()

    def on_click(self, event):
        x, y = event.x, event.y

        (x0, y0), (x1, y1) = self.time_slider_a.label.clipbox.get_points()
        if x0 < x < x1 and y0 < y < y1:
            return

        (x0, y0), (x1, y1) = self.time_slider_b.label.clipbox.get_points()
        if x0 < x < x1 and y0 < y < y1:
            return

    def update_limits(self, x1, x2, y1, y2, z1, z2):
        for axes in (self._plot_a._axes, self._plot_b._axes):
            axes.set_xlim3d(x1, x2)
            axes.set_ylim3d(y1, y2)
            axes.set_zlim3d(z1, z2)

    def get_data(self, center):
        start = max(0, center - self._time_range)
        end = min(len(self.data), center)
        return self.data[start:end]

    def update_a(self,  update_arrow=True):
        data_slice = self.get_data(self._time_center_a)
        self._plot_a._offsets3d = (data_slice[:, 0], data_slice[:, 1], data_slice[:, 2])
        if update_arrow:
            self.update_arrow()

    def update_b(self, update_arrow=True):
        data_slice = self.get_data(self._time_center_b)
        self._plot_b._offsets3d = (data_slice[:, 0], data_slice[:, 1], data_slice[:, 2])
        if update_arrow:
            self.update_arrow()

    def update_arrow(self):
        data_slice_a = self.get_data(self._time_center_a)
        data_slice_b = self.get_data(self._time_center_b)
        c1 = np.average(data_slice_a, axis=0)
        c2 = np.average(data_slice_b, axis=0)
        v = c2 - c1

        if self._line is not None:
            self._line.remove()
            self._line = None
            self._line = self.ax.quiver(c1[0], c1[1], c1[2], v[0], v[1], v[2], color='r')

        c1 = c1.reshape(1, 3)
        c2 = c2.reshape(1, 3)

        self._centroid_a._offsets3d = (c1[:, 0], c1[:, 1], c1[:, 2])
        self._centroid_b._offsets3d = (c2[:, 0], c2[:, 1], c2[:, 2])

    def update_time_a(self, v):
        self._time_center_a = int(v)
        self.update_a()

    def update_time_b(self, v):
        self._time_center_b = int(v)
        self.update_b()

    def update_range(self, v):
        self._time_range = int(v)
        self.update_a(False)
        self.update_b()


points = get_pad_from_csv("modified_emotiv.csv")
points = points.to_numpy()
plotter = Plotter(points)
