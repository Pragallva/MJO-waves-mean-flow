import cartopy.crs as ccrs
import cartopy.util as cutil
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib
import pylab as py

def plot_map(ax, projection0, CL, grid=True, color='grey'):
    ax.coastlines(color=color, alpha=0.3)
    ax.set_aspect('auto')
    ax.set_xticks([0, 60, 120, 180, 240, 300, 360], crs=projection0())
    ax.set_yticks([-90, -60, -30, 0, 30, 60, 90],   crs=projection0())
    ax.set_extent((0-CL, 360-CL, -90, 90),          crs=projection0())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    if grid:
       ax.gridlines(crs=projection0(), linewidth=2, color='grey', alpha=0.5, linestyle='--', )

def map_contour(lon, lat, field, lrange,  cmap,  projection0, contour, alpha=1, colors='k', linewidths=1.5): 
    cyclic_data, cyclic_lons = cutil.add_cyclic_point(data=field, coord=lon)
    if cmap is None:
        im = contour(cyclic_lons, lat, cyclic_data, lrange, colors=colors, \
                     transform=projection0(), extend='both', alpha=alpha, linewidths=linewidths);
    else:
        im = contour(cyclic_lons, lat, cyclic_data, lrange, cmap=cmap, \
                     transform=projection0(), extend='both', alpha=alpha, linewidths=linewidths);
    return im


def map_subplots(fig, j, projection0, projection, CL, \
                 lon, lat, field, lrange,  cmap,  \
                 contour, alpha=1, colors='k', TITLE='', \
                 mapi=True, grid=True, colorbar=True, coastcolor='grey', \
                 nrow=4, ncol=4, colorbar_labelsize=15):
    
    ax = fig.add_subplot(nrow, ncol, j, projection=projection)
    if mapi:
        plot_map(ax, projection0, CL, grid, coastcolor)
    if contour=='contourf':
        im = map_contour(lon, lat, field, lrange,  cmap,  \
                projection0, ax.contourf, alpha, colors)  
    if contour=='contour':
        im = map_contour(lon, lat, field, lrange,  cmap,  \
                projection0, ax.contour, alpha, colors)
    if colorbar:
        cb = fig.colorbar(im, ax=ax)
        cb.ax.tick_params(labelsize=colorbar_labelsize)
    return ax, im


    