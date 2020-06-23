import numpy as np

def create_grid(ranges,nbin):
    grids=[]
    for i in range(int(len(ranges)/2)):
        grids.append(np.linspace(ranges[2*i],ranges[2*i+1],nbin))
    xx, yy = np.meshgrid(grids[0], grids[1])
    xy_list = np.concatenate([xx.reshape(nbin ** 2, 1), yy.reshape(nbin ** 2, 1)], axis=-1)
    return xy_list,xx,yy,grids


def create_dataset(data, lag_time=None):
    if lag_time is None:
        ValueError

    if type(data) is list:
        x_t0 = []
        x_tt = []
        for item in data:
            x_t0.append(item[:-lag_time])
            x_tt.append(item[lag_time:])
        x_t0 = np.concatenate(x_t0)
        x_tt = np.concatenate(x_tt)
    elif type(data) is np.ndarray:
        x_t0 = data[:-lag_time]
        x_tt = data[lag_time:]
    else:
        raise TypeError('Data type {} is not supported'.format(type(data)))

    return [x_t0, x_tt]

def plot_eigenfunction(state, bins, fig,xx=None,yy=None, levels=None, ax=None, title=None, 
                                      color_label = None, c_lim=None, xlabel='x', ylabel='v', include_cbar=True):
    #im = ax.contour(xx, yy, signal.medfilt(state.reshape(bins, bins),kernel_size=15), levels=levels, cmap='viridis_r' )
    im = ax.contour(xx, yy, state.reshape(bins, bins), levels=levels, cmap='viridis_r' )
    #ax.set_xlabel(xlabel, fontsize=12)
    #ax.set_ylabel(ylabel, fontsize=12)
    if not title is None:
        ax.set_title(title, fontsize = 12)
    #ax.axes.get_xaxis().set_ticks([])
    #ax.axes.get_yaxis().set_ticks([])
    if include_cbar:
        cbar = fig.colorbar(im, ax=ax)
        if not color_label is None: cbar.set_label(color_label, fontsize=12)
    return ax




