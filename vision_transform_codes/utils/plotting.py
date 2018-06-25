"""
Some simple utilities for plotting our transform codes
"""
import numpy as np
from matplotlib import pyplot as plt

class TrainingLivePlot(object):
  """
  A container for a matplotlib plot we'll use to visualize training progress

  Parameters
  ----------
  dict_plot_params : dictionary
      Parameters of the dictionary plot. Currently 'total_num', 'img_height',
      'img_width', 'plot_width', 'plot_height', and 'renorm imgs'
  code_plot_params : dictionary, optional
      Parameters of the code plot. Currently just 'size'
  """
  def __init__(self, dict_plot_params, code_plot_params=None):

    plt.ion()

    #################
    # Dictionary plot
    #################
    self.dict_plot_height = dict_plot_params['plot_height']
    self.dict_plot_width = dict_plot_params['plot_width']
    self.img_height = dict_plot_params['img_height']
    self.img_width = dict_plot_params['img_width']
    self.rand_dict_inds = np.random.choice(
        np.arange(dict_plot_params['total_num']),
        self.dict_plot_height*self.dict_plot_width, replace=False)
    self.dict_renorm_flag = dict_plot_params['renorm imgs']
    # prepare a single image to display
    self.dict_h_margin = 2
    self.dict_w_margin = 2
    full_img_height = (self.img_height * self.dict_plot_height +
                       (self.dict_plot_height - 1) * self.dict_h_margin)
    full_img_width = (self.img_width * self.dict_plot_width +
                       (self.dict_plot_width - 1) * self.dict_w_margin)
    composite_img = np.ones((full_img_height, full_img_width))
    for plot_idx in range(len(self.rand_dict_inds)):
      row_idx = plot_idx // self.dict_plot_height
      col_idx = plot_idx % self.dict_plot_width
      pxr1 = row_idx * (self.img_height + self.dict_h_margin)
      pxr2 = pxr1 + self.img_height
      pxc1 = col_idx * (self.img_width + self.dict_w_margin)
      pxc2 = pxc1 + self.img_width
      composite_img[pxr1:pxr2, pxc1:pxc2] = np.zeros(
          (self.img_height, self.img_width))

    self.dict_fig, self.dict_ax = plt.subplots(1, 1, figsize=(10, 10))
    self.dict_fig.suptitle('Random sample of current dictionary', fontsize=15)
    self.temp_imshow_ref = self.dict_ax.imshow(composite_img, cmap='gray')
    self.dict_ax.axis('off')

    self.requires = ['dictionary']

    ###########
    # Code plot
    ###########
    if code_plot_params is not None:
      # set up the code plot
      self.code_size = code_plot_params['size']
      self.code_fig, self.code_ax = plt.subplots(10, 1, figsize=(10, 5))
      self.code_fig.suptitle('Random sample of codes', fontsize=15)
      for c_idx in range(10):
        # I really want to do stem plots but plt.stem is really slow...
        self.code_ax[c_idx].plot(np.zeros(self.code_size))
      self.requires.append('codes')

  def Requires(self):
    return self.requires

  def ClosePlot(self):
    plt.close()

  def UpdatePlot(self, data, which_plot):

    if which_plot == 'dictionary':
      full_img_height = (self.img_height * self.dict_plot_height +
                         (self.dict_plot_height - 1) * self.dict_h_margin)
      full_img_width = (self.img_width * self.dict_plot_width +
                         (self.dict_plot_width - 1) * self.dict_w_margin)
      if self.dict_renorm_flag:
        maximum_value = 1.0
      else:
        maximum_value = np.max(data[:, self.rand_dict_inds])

      composite_img = maximum_value * np.ones((full_img_height, full_img_width))
      for plot_idx in range(len(self.rand_dict_inds)):
        if self.dict_renorm_flag:
          this_filter = data[:, self.rand_dict_inds[plot_idx]]
          this_filter = this_filter - np.min(this_filter)
          this_filter = this_filter / np.max(this_filter)  # now in [0, 1]
        else:
          this_filter = np.copy(data[:, self.rand_dict_inds[plot_idx]])

        row_idx = plot_idx // self.dict_plot_height
        col_idx = plot_idx % self.dict_plot_width
        pxr1 = row_idx * (self.img_height + self.dict_h_margin)
        pxr2 = pxr1 + self.img_height
        pxc1 = col_idx * (self.img_width + self.dict_w_margin)
        pxc2 = pxc1 + self.img_width
        composite_img[pxr1:pxr2, pxc1:pxc2] = np.reshape(
            this_filter, (self.img_height, self.img_width))

      self.dict_ax.clear()
      self.dict_ax.imshow(composite_img, cmap='gray')
      self.dict_ax.axis('off')
      plt.pause(0.01)

    elif which_plot == 'codes':
      # the dataset will be shuffled after each epoch so there's not particular
      # ordering here, we'll just plot the first 10 codes and then they'll at
      # least stay consistent within epochs
      for c_idx in range(10):
        self.code_ax[c_idx].clear()
        self.code_ax[c_idx].plot(data[:, c_idx])
      plt.pause(0.01)
