import os
import yaml
import copy

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
import pandas as pd

from IPython.display import display
from inspect import getfullargspec

from vistools.tensorboard_data import flatten_dict
import expt

def save__init__args(values, underscore=False, overwrite=False, subclass_only=False):
    """
    Use in `__init__()` only; assign all args/kwargs to instance attributes.
    To maintain precedence of args provided to subclasses, call this in the
    subclass before `super().__init__()` if `save__init__args()` also appears
    in base class, or use `overwrite=True`.  With `subclass_only==True`, only
    args/kwargs listed in current subclass apply.
    """
    prefix = "_" if underscore else ""
    self = values['self']
    args = list()
    Classes = type(self).mro()
    if subclass_only:
        Classes = Classes[:1]
    for Cls in Classes:  # class inheritances
        if '__init__' in vars(Cls):
            args += getfullargspec(Cls.__init__).args[1:]
    for arg in args:
        attr = prefix + arg
        if arg in values and (not hasattr(self, attr) or overwrite):
            setattr(self, attr, values[arg])

def expt_plot(ax, all_x, all_y, label, **kwargs):
  runs = []
  for x, y in zip(all_x, all_y):
    df = pd.DataFrame.from_dict(dict(x=x,y=y))
    runs.append(expt.Run(path='', df=df))

  rl = expt.RunList(runs)

  h = expt.Hypothesis(label, rl)

  return h.plot(ax=ax, x='x', y='y', **kwargs)



# from launchers.sfgen_babyai.launch_individual import load_task_info


# def joint_array(*arrays):
#     """
#     create array of data using minimum length
#     Args:
#         *arrays: Description
    
#     Returns:
#         TYPE: Description
#     """
#     lengths = [len(d) for d in arrays]
#     min_length = min(lengths)
#     return np.array([d[:min_length] for d in arrays])


class VisDataObject:
    """docstring for VisDataObject"""
    def __init__(
        self,
        tensorboard_data,
        settings={},
        color='',
        label='',
        marker='',
        linestyle='',
        markersize=12,
        alpha=0,
        line_alpha=.5,
        ):
        save__init__args(locals())


        # self.tensorboard_data = tensorboard_data
        self.settings_df = tensorboard_data.settings_df
        self.data_df = tensorboard_data.data_df
        # self.settings = settings

        # self.color = color
        # self.label = label
        # self.marker = marker
        # self.markersize = markersize
        # self.linestyle = linestyle
        # self.alpha = alpha
        self.colors = self.defaultcolors()
        self.colororder = self.defaultcolororder()


    def plot_data(self, ax, key, individual_lines=False, **kwargs):
        if individual_lines:
            self.plot_individual_lines(ax, key, **kwargs)
        else:
            self.plot_mean_stderr(ax, key, **kwargs)

    def plot_mean_stderr(self, ax, key, datapoint=0, n_samples=10, xlabel_key=None, settings_idx=-1, label_settings=[], **kwargs):
        df, all_data = self.tensorboard_data[key]
        settings = df['experiment_settings'].tolist()

        if settings_idx > -1:
            settings = [settings[settings_idx]]
        else:
            settings_idx = 0

        for idx, setting in enumerate(settings):
            # this is a list of all the lines
            y = all_data[setting]
            if xlabel_key is None:
                raise NotImplementedError
            else:
                # use key as x-axis
                _, xdata = self.tensorboard_data[xlabel_key]
                # all same, so pick 1st
                x = xdata[setting]

            # -----------------------
            # compute plot/fill kwargs
            # -----------------------
            # load default
            plot_kwargs, fill_kwargs = self.default_plot_fill_kwargs()
            # update with filter settings
            self.update_plot_fill_kwargs(
                plot_kwargs, fill_kwargs,
                label_settings=label_settings,
                datapoint=datapoint,
                idx=settings_idx,
                )

            ax = expt_plot(ax=ax, all_x=x, all_y=y, n_samples=n_samples, err_style="fill", **plot_kwargs)

    def plot_individual_lines(self, ax, key, datapoint=0, xlabel_key=None, settings_idx=-1, label_settings=[], **kwargs):
        df, all_data = self.tensorboard_data[key]
        settings = df['experiment_settings'].tolist()

        if settings_idx > -1:
            settings = [settings[settings_idx]]
        else:
            settings_idx = 0

        # -----------------------
        # compute plot/fill kwargs
        # -----------------------
        # load default
        plot_kwargs, fill_kwargs = self.default_plot_fill_kwargs()
        # update with filter settings
        self.update_plot_fill_kwargs(
            plot_kwargs, fill_kwargs,
            label_settings=label_settings,
            datapoint=datapoint,
            idx=settings_idx,
            )

        if xlabel_key is not None:
            # use key as x-axis
            _, xdata = self.tensorboard_data[xlabel_key]

        for idx, setting in enumerate(settings):
            # this is a list of all the lines
            data = all_data[setting]

            joint = joint_array(*data)
            # -----------------------
            # plot mean
            # -----------------------
            mean = joint.mean(0)
            if xlabel_key is None:
                x = np.arange(len(mean))
            else:
                # all same, so pick 1st
                x = xdata[setting][idx]
                x= x[:len(mean)]
                mean = mean[:len(x)]
            ax.plot(x, mean, **plot_kwargs)


            # -----------------------
            # plot lines
            # -----------------------
            individual_kwargs = copy.deepcopy(plot_kwargs)
            individual_kwargs['label'] = None
            individual_kwargs['alpha'] = self.line_alpha
            if self.alpha:
                individual_kwargs['alpha'] *= self.alpha
            for idx, line in enumerate(data):
                if xlabel_key is None:
                    x = np.arange(len(line))
                else:
                    x = xdata[setting][idx]
                    x = x[:len(line)]

                ax.plot(x, line, **individual_kwargs)


    def label_from_settings(self, columns=[], idx=0):
        """Create a lable using the values in the columns provided. idx indicates which index of the column
        
        Args:
            columns (list, optional): Description
            idx (int, optional): Description
        
        Returns:
            TYPE: Description
        """
        if isinstance(columns, dict):
            columns = flatten_dict(columns, sep=" : ")
            columns = list(columns.keys())

        key0 = columns[0]
        val0 = self.tensorboard_data.settings_df[key0].to_numpy()[idx]
        label = f"{key0}={val0}"
        for key in columns[1:]:
            val = self.tensorboard_data.settings_df[key].to_numpy()[idx]
            label += f"\n{key}={val}"

        if len( columns[1:]):
            label += "\n"
        return label


    def update_plot_fill_kwargs(self, plot_kwargs, fill_kwargs, datapoint=0, label_settings=[], idx=0):
        """Update plot, fill kwargs using settings for this data object
        
        Args:
            plot_kwargs (dict): 
            fill_kwargs (dict): 
            label_settings (list, optional): which settings to use to construct label. not yet implemented.
        """
        if self.color:
            color = self.color
        else:
            color = self.colororder[datapoint]

        plot_kwargs['color'] = self.colors.get(color, color)
        fill_kwargs['color'] = self.colors.get(color, color)


        if self.linestyle:
            plot_kwargs['linestyle'] = self.linestyle

        if self.marker:
            plot_kwargs['marker'] = self.marker
            plot_kwargs['markersize'] = self.markersize

        if self.alpha:
          plot_kwargs['alpha'] = self.alpha
          if 'alpha' in fill_kwargs:
            fill_kwargs['alpha'] = fill_kwargs['alpha']*self.alpha
          else:
            fill_kwargs['alpha'] = self.alpha

        if self.label:
            plot_kwargs['label'] = self.label
        else:
            plot_kwargs['label'] = self.label_from_settings(columns=label_settings, idx=idx)


    @staticmethod
    def lineargs(): 
        return [
            dict(
                linestyle='-.',
                # marker='o',
                ),
            dict(
                linestyle='--',
                # marker='X',
                ),
            dict(
                linestyle='-.',
                # marker='^',
                ),
            dict(
                linestyle='--',
                # marker='x',
                ),
        ]

    @staticmethod
    def default_plot_fill_kwargs():
        plot_kwargs=dict(
            linewidth=4,
            )
        fill_kwargs=dict(
            alpha=.2,
            )

        return plot_kwargs, fill_kwargs

    @staticmethod
    def defaultcolors():
        return dict(
            grey = '#99999e',
            dark_grey = "#363535",
            # =========================================
            # red
            # =========================================
            red='#d9432f',
            light_red='#FF0000',
            dark_red='#8B0000',
            # =========================================
            # Purple
            # =========================================
            light_purple = "#9d81e3",
            purple = "#7C66B4",
            dark_purple = "#4c2d9c",
            # =========================================
            # green
            # =========================================
            light_green = "#2da858",
            green = "#489764",
            dark_green = "#117333",
            # =========================================
            # blue
            # =========================================
            light_blue = "#7dc8e3",
            grey_blue = "#57c4fa",
            blue = "#5C94E1",
            dark_blue = "#1152ad",
            ##
            # Orange
            #
            orange = "#f5a742",
        )

    @staticmethod
    def defaultcolororder():
        return [
            'blue',
            'red',
            'black',
            'orange',
            'purple',
            'green',
            'grey',
            'dark_grey',
            'dark_green',
            'dark_purple',
            'dark_blue',
            'dark_red',
            'dark_orange',
            'light_green',
            'light_blue',
            'light_purple',
            'light_red',
            'light_orange',

        ]


class Vistool(object):
    def __init__(self,
        tensorboard_data,
        plot_settings=[],
        metadata_stats=['num_seeds'],
        metadata_settings_dict={},
        metadata_settings_list=[],
        filter_key=None,
        filter_column='max',
        key_with_legend=None,
        plot_data_kwargs={},
        common_settings={},
        ):

        self.tensorboard_data = tensorboard_data
        self.plot_settings = plot_settings
        self.key_with_legend = key_with_legend
        self.plot_data_kwargs = dict(plot_data_kwargs)
        self.filter_key = filter_key
        self.filter_column = filter_column
        self.common_settings = dict(common_settings)

        # default settings for displaying metadata
        self.metadata_stats = metadata_stats
        # combine list and dict together
        self.metadata_settings = list(metadata_settings_list)
        self.metadata_settings.extend(
            list(flatten_dict(metadata_settings_dict, sep=':').keys())
            )

    def plot_filters(self,
        # ----------------
        # Arguments for getting matches for data
        # ----------------
        data_filters=None,
        data_filter_space=None,
        filter_key=None,
        filter_column='max',
        common_settings={},
        topk=1,
        filter_kwargs={},
        # ----------------
        # Arguments for displaying dataframe
        # ----------------
        display_settings=[],
        display_stats=[],
        # ----------------
        # Arguments for Plot Keys
        # ----------------
        plot_settings=None,
        maxcols=4,
        key_with_legend=None,
        individual_lines=False,
        subplot_kwargs={},
        plot_data_kwargs={},
        fig_kwargs={},
        legend_kwargs={},
        # ----------------
        # misc.
        # ----------------
        verbosity=1,
        ):
        # ======================================================
        # ======================================================
        # load filters
        # ======================================================
        if data_filters is not None and data_filter_space is not None:
            raise RuntimeError("Can only provide filter list or filter search space")

        if data_filters is None:
            data_filter_space = flatten_dict(data_filter_space, sep=":")
            settings = ParameterGrid(data_filter_space)
            data_filters = [dict(settings=s) for s in settings]
        else:
            data_filters=[f if 'settings' in f else dict(settings=f) for f in data_filters]
        # ======================================================
        # get 1 object with data per available data filter
        # ======================================================
        vis_objects = get_vis_objects(
            tensorboard_data=self.tensorboard_data,
            data_filters=data_filters,
            common_settings=common_settings if common_settings else self.common_settings,
            filter_key=filter_key if filter_key else self.filter_key,
            filter_column=filter_column if filter_column else self.filter_column,
            topk=topk,
            filter_kwargs=filter_kwargs,
            verbosity=verbosity,
            )

        # ======================================================
        # display pandas dataframe with relevant data
        # ======================================================
        if verbosity:
            display_metadata(
                vis_objects=vis_objects,
                settings=display_settings if display_settings else self.metadata_settings,
                stats=display_stats if display_stats else self.metadata_stats,
                data_key=filter_key if filter_key else self.filter_key,
                )

        # ======================================================
        # setup kwargs
        # ======================================================
        if not plot_data_kwargs:
            plot_data_kwargs = self.plot_data_kwargs
        plot_data_kwargs = copy.deepcopy(plot_data_kwargs)
        if not 'label_settings' in plot_data_kwargs:
            plot_data_kwargs['label_settings'] = data_filters[0]['settings']


        fig_kwargs = copy.deepcopy(fig_kwargs)
        subplot_kwargs = copy.deepcopy(subplot_kwargs)
        legend_kwargs = copy.deepcopy(legend_kwargs)
        # ======================================================
        # create plot for each top-k value
        # ======================================================
        plot_settings = plot_settings or self.plot_settings
        for k in range(topk):
            _plot_settings = copy.deepcopy(plot_settings)
            # -----------------------
            # add K to titles for identification
            # -----------------------
            if topk > 1:
                for info in _plot_settings:
                    info['title'] = f"{info['title']} (TopK={k})"

                # indicate which settings to use
                plot_data_kwargs['settings_idx'] = k

            plot_keys(
                vis_objects=vis_objects,
                plot_settings=_plot_settings,
                maxcols=maxcols,
                subplot_kwargs=subplot_kwargs,
                plot_data_kwargs=plot_data_kwargs,
                fig_kwargs=fig_kwargs,
                legend_kwargs=legend_kwargs,
                individual_lines=individual_lines,
                key_with_legend=key_with_legend if key_with_legend else self.key_with_legend,
                )


class PanelTool(Vistool):
    """docstring for PanelTool"""
    def __init__(self,
        dim_titles=[],
        dims_to_plot=dict(),
        title_with_legend=None,
        *args, **kwargs,
        ):
        super(PanelTool, self).__init__(*args, **kwargs)
        self.dim_titles = dim_titles
        self.title_with_legend = title_with_legend
        if dims_to_plot:
            if isinstance(dims_to_plot, list):
                self.dimfilters_to_plot = load_formatted_data_filters(data_filters=dims_to_plot)
            elif isinstance(dims_to_plot, dict):
                self.dimfilters_to_plot = load_formatted_data_filters(data_filter_space=dims_to_plot)
            else:
                raise RuntimeError


    def plot_filters_across_dimensions(self,
        # ----------------
        # Arguments for getting matches for data
        # ----------------
        data_filters=None,
        data_filter_space=None,
        filter_key=None,
        filter_column='max',
        common_settings={},
        topk=1,
        filter_kwargs={},
        # ----------------
        # Arguments for displaying dataframe
        # ----------------
        display_settings=[],
        display_stats=[],
        # ----------------
        # Arguments for Plot Keys
        # ----------------
        maxcols=4,
        title_with_legend=None,
        subplot_kwargs={},
        plot_data_kwargs={},
        fig_kwargs={},
        legend_kwargs={},
        title_addition="",
        # ----------------
        # misc.
        # ----------------
        verbosity=1,
        ):
        # ======================================================
        # ======================================================
        if data_filters is not None and data_filter_space is not None:
            raise RuntimeError("Can only provide filter list or filter search space")

        data_filters = load_formatted_data_filters(data_filters, data_filter_space)


        # ======================================================
        # subplots
        # ======================================================
        total_plots = len(self.dimfilters_to_plot)*len(self.plot_settings)


        axis_sharey=len(self.plot_settings)==1 # plotting same key repeatedly
        default_subplot_kwargs=dict(
            sharey=axis_sharey,
            sharex=False,
            gridspec_kw=dict(
                wspace=0,
                # hspace=.
            )
        )
        default_subplot_kwargs.update(
            subplot_kwargs
            )

        fig, axs, ncols = make_subplots(
            num_plots=total_plots,
            maxcols=maxcols,
            **default_subplot_kwargs,
            )

        for idx, dimfilter in enumerate(self.dimfilters_to_plot):

            assert topk == 1, "only support 1 topk"
            # ======================================================
            # get 1 object with data per available data filter
            # ======================================================
            vis_objects = get_vis_objects(
                tensorboard_data=self.tensorboard_data,
                data_filters=data_filters,
                common_settings=dict(
                    **dimfilter['settings'],
                    **common_settings,
                    ),
                filter_key=filter_key if filter_key else self.filter_key,
                filter_column=filter_column if filter_column else self.filter_column,
                topk=topk,
                filter_kwargs=filter_kwargs,
                verbosity=verbosity,
                )

            if not vis_objects:
                print(f"No objects found for {dimfilter['settings']}")
                continue


            # ======================================================
            # display pandas dataframe with relevant data
            # ======================================================
            if verbosity and idx == 0:
                display_metadata(
                    vis_objects=vis_objects,
                    settings=display_settings if display_settings else self.metadata_settings,
                    stats=display_stats if display_stats else self.metadata_stats,
                    data_key=filter_key if filter_key else self.filter_key,
                    )

            # ======================================================
            # setup kwargs
            # ======================================================
            fig_kwargs = copy.deepcopy(fig_kwargs)
            subplot_kwargs = copy.deepcopy(subplot_kwargs)
            legend_kwargs = copy.deepcopy(legend_kwargs)
            plot_settings = copy.deepcopy(self.plot_settings)

            if not plot_data_kwargs:
                plot_data_kwargs = self.plot_data_kwargs
            plot_data_kwargs = copy.deepcopy(plot_data_kwargs)
            if not 'label_settings' in plot_data_kwargs:
                plot_data_kwargs['label_settings'] = data_filters[0]['settings']

            # -----------------------
            # if sharing y-axis, remove ylabel from inner plots
            # -----------------------
            if axis_sharey:
                if idx in np.arange(0, 100, ncols):
                    # keep ylabel
                    pass
                else:
                    #remove ylabel
                    plot_settings[0].pop('ylabel')

                # if idx % 
                # plot_settings[] ncols


            # ======================================================
            # load title + legend information
            # ======================================================
            # make the first key present use the title given by dim titles
            plot_legend=False
            title_with_legend = title_with_legend if title_with_legend else self.title_with_legend
            if self.dim_titles:
                title = self.dim_titles[idx]
                plot_settings[0]['title'] = title
                plot_legend = title_with_legend == title

                if title_addition: 
                    plot_settings[0]['title'] += " " + title_addition


            plot_keys(
                axs=axs[idx:idx+len(self.plot_settings)],
                vis_objects=vis_objects,
                plot_settings=plot_settings,
                maxcols=maxcols,
                subplot_kwargs=subplot_kwargs,
                plot_data_kwargs=plot_data_kwargs,
                fig_kwargs=fig_kwargs,
                legend_kwargs=legend_kwargs,
                key_with_legend=None,
                plot_legend=plot_legend,
                plt_show=False
                )
        plt.show()


def load_formatted_data_filters(data_filters=None, data_filter_space=None):
    if data_filters is not None and data_filter_space is not None:
        raise RuntimeError("Can only provide filter list or filter search space")

    if data_filters is None:
        data_filter_space = flatten_dict(data_filter_space, sep=":")
        settings = ParameterGrid(data_filter_space)
        data_filters = [dict(settings=s) for s in settings]
    else:
        data_filters=[f if 'settings' in f else dict(settings=f) for f in data_filters]

    return data_filters


def get_vis_objects(tensorboard_data, data_filters, common_settings, filter_key, filter_column='max', topk=1, filter_kwargs={}, verbosity=0):
    # copy data so can reuse
    data_filters = copy.deepcopy(data_filters)
    common_settings = copy.deepcopy(common_settings)
    common_settings = flatten_dict(common_settings, sep=':')
    

    vis_objects = []
    for data_filter in data_filters:
        data_filter['settings'] = flatten_dict(data_filter['settings'], sep=':')
        data_filter['settings'].update(common_settings)
        # import ipdb; ipdb.set_trace()
        match = tensorboard_data.filter_topk(
            key=filter_key,
            column=filter_column,
            filters=[data_filter['settings']],
            topk=topk,
            verbose=verbosity,
            **filter_kwargs,
        )[0]

        if match is not None:
            vis_object = VisDataObject(
                tensorboard_data=match,
                **data_filter,
                )
            vis_objects.append(vis_object)

    return vis_objects

def display_metadata(vis_objects, settings=[], stats=[], data_key=None):
    """Display metadata about config (settings) or some stats, e.g. nonzero seeds, number of seeds, etc. (stats).
    
    Args:
        settings (list, optional): config settings
        stats (list, optional): run statistics
    """
    # this enable you to use an empty dictionary to populate settings
    if isinstance(settings, dict):
        settings = flatten_dict(settings, sep=':')
        settings = list(settings.keys())

    settings_df = pd.concat([o.settings_df for o in vis_objects])
    try:
      settings_df = settings_df[settings]
    except Exception as e:
      pass

    if data_key is None:
        data_key = vis_objects[0].tensorboard_data.keys_like('return')[0]
    data_df = pd.concat([o.data_df[data_key]
                            for o in vis_objects])[stats]
    display(pd.concat([data_df, settings_df], axis=1))


# ======================================================
# Plotting functions
# ======================================================
def make_subplots(num_plots, maxcols=4, unit=8, **kwargs):
    #number of subfigures
    ncols = min(num_plots, maxcols)
    nrows = num_plots//maxcols
    nrows = max(nrows, 1)
    if ncols % num_plots != 0:
        nrows += 1


    # import ipdb; ipdb.set_trace()
    if not 'figsize' in kwargs:
        height=nrows*unit
        width=ncols*unit
        kwargs['figsize'] = (width, height)


    fig, axs = plt.subplots(nrows, ncols, **kwargs)

    if num_plots > 1:
      axs = axs.ravel()
    else:
      axs = [axs]

    return fig, axs, ncols

def plot_keys(
    vis_objects,
    keys=[],
    plot_settings=[],
    maxcols=4,
    subplot_kwargs={},
    plot_data_kwargs={},
    legend_kwargs={},
    fig_kwargs={},
    axs=None,
    key_with_legend=None,
    plot_legend=True,
    plt_show=True,
    individual_lines=False,
    ):
    if len(keys) > 0 and len(plot_settings) > 0:
        raise RuntimeError("Either only provide keys or plot infos list of dictionaries, where each dict also has a key. Don't provide both")
    # convert, so can use same code for both
    if len(keys):
        plot_settings = [dict(key=k) for k in keys]

    if axs is None:
        fig, axs, _ = make_subplots(
            num_plots=len(plot_settings),
            maxcols=maxcols,
            **subplot_kwargs,
            )

    # ======================================================
    # plot data
    # ======================================================
    if key_with_legend is None:
        key_with_legend = plot_settings[0]['key']

    individual_lines = plot_data_kwargs.get("individual_lines", individual_lines)

    for ax, plot_setting in zip(axs, plot_settings):
        key = plot_setting['key']
        for idx, vis_object in enumerate(vis_objects):
            vis_object.plot_data(ax, key=key,
                datapoint=idx,
                individual_lines=individual_lines,
                **plot_data_kwargs)

        finish_plotting_ax(
            ax=ax,
            plot_setting=plot_setting,
            plot_legend=key_with_legend==key and plot_legend,
            legend_kwargs=legend_kwargs,
            **fig_kwargs,
            )


    for i in range(len(plot_settings), len(axs)):
        fig.delaxes(axs.flatten()[i])

    if plt_show:
        plt.show()

def finish_plotting_ax(
    ax,
    plot_setting,
    title_size=22,
    title_loc='center',
    minor_text_size=18,
    legend_text_size=16,
    grid_kwargs={},
    legend_kwargs={},
    ysteps=10,
    plot_legend=True,
    ):
    # -----------------------
    # createa a grid
    # -----------------------
    ax.tick_params(axis='both', which='major', labelsize=minor_text_size)
    if not grid_kwargs:
        grid_kwargs = copy.deepcopy(default_grid_kwargs())
    ax.grid(**grid_kwargs)
    

    # -----------------------
    # set title
    # -----------------------
    title = plot_setting.get("title", '')
    if title:
      ax.set_title(title, fontsize=title_size, loc=title_loc)

    # -----------------------
    # labels (names, sizes)
    # -----------------------
    ylabel = plot_setting.get("ylabel", '')
    xlabel = plot_setting.get("xlabel", '')
    if ylabel: ax.set_ylabel(ylabel)
    if xlabel: ax.set_xlabel(xlabel)

    # set the size of labels
    ax.yaxis.label.set_size(minor_text_size)
    ax.xaxis.label.set_size(minor_text_size)
    # set the size of text displaying the magnitude
    text = ax.xaxis.get_offset_text()
    text.set_size(minor_text_size)

    # -----------------------
    # x/y limits
    # -----------------------
    xlim = plot_setting.get('xlim', None)
    if xlim:
      ax.set_xlim(*xlim)

    ylim = plot_setting.get('ylim', None)
    if ylim:
      ax.set_ylim(*ylim)
      length = ylim[1]-ylim[0]
      step = length/ysteps
      ax.set_yticks(np.arange(ylim[0], ylim[1]+step, step))

    # -----------------------
    # setup legend
    # -----------------------
    _legend_kwargs = {}
    if isinstance (legend_kwargs, str):
        if legend_kwargs.lower() == "none":
            pass
        elif legend_kwargs.lower() == "right":
            _legend_kwargs=dict(
                loc='upper left',
                bbox_to_anchor=(1,1), 
                )
    elif isinstance (legend_kwargs, dict):
        _legend_kwargs.update(legend_kwargs)
    else:
        raise NotImplementedError

    if plot_legend:
      ax.legend(
        **_legend_kwargs,
        prop={'size': legend_text_size})

def default_grid_kwargs():
    return dict(
      linestyle="-.",
      linewidth=.5,
    )



