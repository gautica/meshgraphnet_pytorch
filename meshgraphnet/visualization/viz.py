from matplotlib import tri as mtri
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    canvas = FigureCanvas(fig)
    canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = canvas.get_width_height()
    buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    buf.shape = (h, w, 3)

    return buf


def cfd2images(mesh_pos, faces, field, every_k_step=30):
    steps = field.shape[1]
    frames = []

    min_bound = field.min(axis=0)
    max_bound = field.max(axis=0)
    for i in range(0, steps, every_k_step):
        fig = Figure(figsize=(300, 100), dpi=1)
        ax = fig.gca()
        ax.axis('off')
        ax.margins(0)
        
        triang = mtri.Triangulation(mesh_pos[:, 0], mesh_pos[:, 1], faces)
        ax.tripcolor(triang, field[:, i, 0], vmin=min_bound[i, 0], vmax=max_bound[i, 0])
        ax.triplot(triang, 'ko-', ms=5, lw=3)
        image_from_plot = fig2data(fig)
        frames.append(image_from_plot)
        fig.clf()
    frames = np.transpose(np.stack(frames, axis=0), (0, 3, 1, 2))
    return frames


def cloth2images(faces, field, every_k_step=30):
    steps = field.shape[1]
    frames = []

    min_bound = field.min(axis=(0, 1))
    max_bound = field.max(axis=(0, 1))
    for i in range(0, steps, every_k_step):
        fig = Figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection='3d')
        ax.axis('off')
        ax.margins(0)
        ax.set_xlim(min_bound[0], max_bound[0])
        ax.set_ylim(min_bound[1], max_bound[1])
        ax.set_zlim(min_bound[2], max_bound[2])

        ax.plot_trisurf(field[:, i, 0], field[:, i, 1], faces, field[:, i, 2], shade=True)

        image_from_plot = fig2data(fig)
        frames.append(image_from_plot)
        fig.clf()
    frames = np.transpose(np.stack(frames, axis=0), (0, 3, 1, 2))           # (N, 3, H, W)
    return frames


def generate_images(mesh_pos, faces, field, mode, every_k_step=30):
    if mode == 'cfd':
        return cfd2images(mesh_pos, faces, field, every_k_step=every_k_step)
    elif mode == 'cloth':
        return cloth2images(faces, field, every_k_step=every_k_step)
    else:
        assert False, 'Invalid mode: cfd|cloth'

