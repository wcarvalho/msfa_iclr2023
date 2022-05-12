import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
try:
    import torchvision
except Exception as e:
    pass

def make_grid(boxes, **kwargs):
  ylim = 0
  xlim = 0
  for box in boxes.values():
    ylim = max(ylim, *box['y'])
    xlim = max(xlim, *box['x'])
  figsize=(xlim, ylim)

  fig = plt.figure(figsize=figsize, constrained_layout=True)

  # ======================================================
  # setup boxes
  # ======================================================
  gs = fig.add_gridspec(ylim, xlim, **kwargs)
  axs = {}
  for key, box in boxes.items():
    x = box['x']
    y = box['y']
    y = ylim - np.array(y)[::-1]

    axs[key] = fig.add_subplot(gs[y[0]:y[1], x[0]:x[1]])

  return fig, axs


class VideoMaker(object):
  """docstring for VideoMaker"""
  def __init__(self,
    xlim,
    ylim,
    boxes,
    initialization_fns=None,
    update_fns=None,
    gridspec_kwargs={},
    settings={},
    verbosity=0,
    fps=1,
    dpi=100,
    ):

    self.xlim = xlim
    self.ylim = ylim
    self.fps = fps
    self.dpi = dpi
    self.verbosity = verbosity
    self.boxes = boxes
    self.initialization_fns = initialization_fns
    self.update_fns = update_fns

    self.env_data = []
    self.agent_data=  []
    self.settings = settings

    self.fig = plt.figure(figsize=(xlim, ylim), constrained_layout=True)

    # ======================================================
    # setup boxes
    # ======================================================
    gs = self.fig.add_gridspec(ylim, xlim, **gridspec_kwargs)
    self.axs = {}
    for key, box in boxes.items():
      x = box['x']
      y = box['y']
      y = ylim - np.array(y)[::-1]

      self.axs[key] = self.fig.add_subplot(gs[y[0]:y[1], x[0]:x[1]])



  def preview(self, env_data, agent_data):
    self.make(env_data, agent_data, preview=True)

  def update(self, idx):
    for key in self.boxes.keys():
      if not key in self.update_fns: continue
      # try:
      self.update_fns[key](
        env_data=self.env_data,
        agent_data=self.agent_data,
        idx=idx,
        ax=self.axs[key],
        updater=self.updaters[key],
        settings=self.settings,
      )
      # except Exception as e:
      #   error =  f"\nKey : {key}\n"
      #   error += f"Step: {idx}\n"
      #   error += str(e)
      #   raise RuntimeError(error)

  def make(self,
    env_data,
    length,
    agent_data,
    preview=False,
    video_path='',
    ):
    # ======================================================
    # initalize each subfigure
    # ======================================================
    self.updaters = {}
    for key in self.boxes.keys():
      if not key in self.initialization_fns: continue
      self.updaters[key] = self.initialization_fns[key](env_data=env_data, ax=self.axs[key], agent_data=agent_data)

    if preview: return

    self.env_data = env_data
    self.agent_data = agent_data

    # ======================================================
    # create video
    # ======================================================

    ani = FuncAnimation(self.fig, self.update, np.arange(0, length))

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=self.fps, metadata=dict(artist='Me'), bitrate=1800)
    # writer = animation.FFMpegFileWriter(fps=self.fps, metadata=dict(artist='Me'), bitrate=1800)
    

    if video_path:
      # filepath_exists(video_path)
      ani.save(video_path, writer=writer, dpi=self.dpi)
      if self.verbosity:
        print(video_path)
    plt.close()



# ======================================================
# pre-defined initialization functions
# ======================================================
def plt_ready_im(image):
  channels = image.shape[-1]
  if channels == 1:
    image = image[:,:,0]
  return image

def plt_initiate_image(ax, image):
  channels = image.shape[-1]
  if channels == 1:
    kwargs=dict(
      cmap='gray',
      vmin=0.0,
      vmax=1.0
      )
  else:
    kwargs=dict()
  return ax.imshow(plt_ready_im(image), **kwargs)

def image_initialization(env_data, ax, title='', title_size=16, **kwargs):
  plot_image(
        ax=ax,
        title=title,
        fontsize=title_size,
    )

  image = env_data.observation.image[0,0]
  start_image = np.zeros(image.transpose(1,2,0).shape)
  return plt_initiate_image(ax, start_image)


def bar_initialization(ax, natoms, title='', title_size=16, label_size=12, **kwargs):
  return bar_plot(
      ax=ax,
      values=np.zeros((natoms)),
      title=title,
      title_size=title_size,
      tick_text_size=label_size,
      **kwargs,
  )

# ======================================================
# Pre-defined update functions
# ======================================================
def update_image(env_data, idx, updater, **kwargs):
  """Update image in figure"""
  image = env_data.observation.image[idx,0]
  updater.set_data(plt_ready_im(image.transpose(1,2,0)))


# ======================================================
# plotting functions
# ======================================================
def bar_plot(ax, names=[], values=[], bar_kwargs={}, colors=None, important_names=[], important_color=None, ylabel="", title="", ylim=[], nticks=0, title_size=16, tick_text_size=16, rotation=45, remove_yticks=False, remove_xticks=False, **kwargs):
  y_pos = np.arange(len(values))

  if values is not None:
    bar = ax.bar(y_pos, values, align='center', **bar_kwargs)
  if colors:
    for child, color in zip(ax.get_children(), colors):
      child.set_color(color)

  if important_names:
    for name, child in zip(names, ax.get_children()):
      if name in important_names:
        child.set_color(important_color)

  if names:
    ax.set_xticks(y_pos)
    ax.set_xticklabels(names, rotation=rotation, fontsize=tick_text_size)
  if title:
    ax.set_title(title, fontsize=title_size)
  if ylabel:
    ax.set_ylabel(ylabel)
  if ylim:
      ax.set_ylim(*ylim)

  if nticks and ylim:
    length = ylim[1]-ylim[0]
    step = length/nticks
    ax.set_yticks(np.arange(ylim[0], ylim[1]+step, step))

  if remove_yticks:
    ax.yaxis.set_ticklabels([])
    ax.set_yticks([])
  if remove_xticks:
    ax.xaxis.set_ticklabels([])
    ax.set_xticks([])

  if values is not None:
    return bar

def pie_chart(ax, names=[], values=[], colors=[], important_names=[], important_color=None, wedge_color='white', wedge_width=4, minimum=.05, title="", title_size=16, label_size=16, **kwargs):

  filtered_names=[]
  filtered_values=[]
  filtered_colors=[]
  for name, value, color in zip(names, values, colors):
    if value > minimum:
      filtered_names.append(name)
      filtered_values.append(value)
      filtered_colors.append(color)

  filtered_values = np.array(filtered_values)
  filtered_values = filtered_values/filtered_values.sum()

  wedges, texts = ax.pie(filtered_values,
      labels=filtered_names,
      colors=filtered_colors
      )
  ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
  if important_names:
    for name, child in zip(filtered_names, ax.get_children()):
      if name in important_names:
        try:
          child.set_color(important_color)
        except AttributeError as ae:
          pass


  for t in texts:
    t.set_fontsize(label_size)

  for w in wedges:
    w.set_linewidth(wedge_width)
    w.set_edgecolor(wedge_color)

  if title:
    ax.set_title(title, fontsize=title_size)


def plot_image(
  ax=None,
  image=None,
  title="",
  fontsize=16,
  imshow_kwargs=dict(),
  figsize=(10,10),
  axis_off=True,
  ):
  """Plot Image
  
  Args:
      ax (None, optional): Description
      image (None, optional): Description
      title (str, optional): Description
      fontsize (int, optional): Description
      imshow_kwargs (TYPE, optional): Description
      figsize (tuple, optional): Description
  """
  if ax is None:
    fig, ax = plt.subplots(1, 1, figsize=figsize)

  if title:
    ax.set_title(title, fontsize=fontsize)
  ax.grid(False)
  ax.xaxis.set_ticklabels([])
  ax.yaxis.set_ticklabels([])

  ax.axis('off')
  if image is not None:
    return ax.imshow(image, **imshow_kwargs)


def plot_image_batch(batch, normalize=True, return_grid=False, **kwargs):
  """plot grid of images
  Args:
      batch (TYPE): B X (X N) H X W X C
      nrow (int, optional): Description
      normalize (bool, optional): Description
  Raises:
      NotImplementedError: Description
  """

  if len(batch.shape) < 2 or len(batch.shape) > 5:
    raise NotImplementedError

  if len(batch.shape) == 2:
    if isinstance(batch, np.ndarray):
      batch = np.expand_dims(batch, 2)

    elif isinstance(batch, torch.Tensor):
      batch = batch.unsqueeze(2)

  if len(batch.shape) == 3:
    if isinstance(batch, np.ndarray):
      batch = np.expand_dims(batch, 0)

    elif isinstance(batch, torch.Tensor):
      batch = batch.unsqueeze(0)

  if isinstance(batch, np.ndarray):
    if len(batch.shape) == 5:
      batch = batch.reshape(-1, *batch.shape[2:])
    x = torch.from_numpy(batch).permute(0, 3, 1, 2)

  elif isinstance(batch, torch.Tensor):
    if len(batch.shape) == 5:
      batch = batch.flatten(0,1)
    x = batch

  else:
    raise NotImplementedError


  if normalize:
    x = x/255

  grid_img = torchvision.utils.make_grid(x, **kwargs)

  if return_grid:
    return grid_img
  plt.imshow(grid_img.permute(1, 2, 0))