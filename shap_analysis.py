"""
General classes for plotting predictions of
Neutral Density Random Forest Model (Halford/Murphy et al.)
as published in Murphy et al. 2025 (https://doi.org/10.1029/2024SW003928)

Also see MLTDM repository (under MIT license): https://github.com/kylermurphy/mltdm/

"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pandas as pd
import time as pytime
import numpy as np
import numpy.typing as npt
import cartopy.crs as ccrs
from cartopy.feature.nightshade import Nightshade
import datetime as dt
from mltdm.subsol import subsol

fgeo_col = ['1300_02', '43000_09', '85550_13', '94400_18', 'SYM_H index', 
            'AE', 'SatLat', 'cos_SatMagLT', 'sin_SatMagLT']

class PolarBear():

    def __init__(self, n_lat: int = 10, n_mlt: int = 20):
        """
        Parameters
        ----------
        n_lat : int
            Number of points for sat Lat between (0,90)
        n_mlt : int
            Number of points for MLT between (0,24)
        """
        self.n_lat = n_lat
        self.n_mlt = n_mlt
        self.setup_grid()

    def setup_grid(self):
        """
        Sets up needed arrays
        """
        self.radii = np.linspace(0.999,0.001,self.n_lat) # start at 0.01 to avoid pole singularity
        self.satLat_N = np.arccos(self.radii)*180/np.pi
        self.satLat_S = -self.satLat_N

        self.theta = np.linspace(0,2.*np.pi,self.n_mlt)

        self.MLT_all = 24.*self.theta/(2*np.pi)

        self.radMesh, self.thtMesh = np.meshgrid(self.radii, self.theta, indexing='ij')
        self.satlt_mesh_N, self.MLT_mesh = np.meshgrid(self.satLat_N, self.MLT_all, indexing='ij')
        self.satlt_mesh_S = -self.satlt_mesh_N

        self.satLat_N = self.satlt_mesh_N.flatten()
        self.satLat_S = self.satlt_mesh_S.flatten()
        self.cos_theta = np.cos(self.thtMesh.flatten())
        self.sin_theta = np.sin(self.thtMesh.flatten())

        # for GEO map
        self.geoLat = np.linspace(-89.,89.,self.n_lat*2) # is satLat
        self.geoLon = np.linspace(-179.,179., self.n_mlt)
        self.geoLat_msh, self.geoLon_msh = np.meshgrid(self.geoLat, self.geoLon, indexing='ij')


    def make_grid(self, event, north: bool):
        if isinstance(event, pd.Series):
            # need to convert to dataFrame for grid generation
            evt = event.to_frame().transpose()
        else:
            evt = event

        evt = evt[fgeo_col]

        grid = pd.concat([evt]*self.n_lat*self.n_mlt, ignore_index=True)
        satlt_mesh = self.satlt_mesh_N if north else self.satlt_mesh_S

        grid["SatLat"] = self.satLat_N if north else self.satLat_S
        grid["cos_SatMagLT"] = self.cos_theta
        grid["sin_SatMagLT"] = self.sin_theta

        return grid, satlt_mesh

    def make_geo_grid(self, event):
        evt_date = event['DateTime']
        if isinstance(event, pd.Series):
            # need to convert to dataFrame for grid generation
            evt = event.to_frame().transpose()
        else:
            evt = event

        evt = evt[fgeo_col]

        # factor of 2 for both hemispheres (latitudes)
        gridGEO = pd.concat([evt]*2*self.n_lat*self.n_mlt, ignore_index=True)

        # GEO MLT grid requires subsolar point longitude
        _, evt_sbslon = subsol(evt_date)
        GEO_MLT = ((self.geoLon_msh.flatten() - evt_sbslon) / 15. + 12)%24
        cos_MLT = np.cos(GEO_MLT*2*np.pi/24.)
        sin_MLT = np.sin(GEO_MLT*2*np.pi/24.)

        gridGEO["SatLat"] = self.geoLat_msh.flatten()
        gridGEO["cos_SatMagLT"] = cos_MLT
        gridGEO["sin_SatMagLT"] = sin_MLT

        return gridGEO

    def plot_density(self, ax, rf, event, grid, satlat_mesh, north: bool, vr=[0.5,6.5]):
        """Dial plot of density prediction

        Parameters
        ----------
        fig: plt.Figure
            Figure to plot on
        ax : plt.Axes
            Axes to plot on (must have projection = "polar")
        rf : Random Forest
            Random Forest model (from scikit-learn)
        event : Pandas Series or DataFrame
            single event to pass to RF model
        grid : Pandas DataFrame
            from 'self.make_grid()', Holds grid of SatLat, cosMLT, sinMLT
        satlat_mesh : np.NDArray
            Holds SatLat mesh
        north : bool
            Plot north hemisphere? (or south?)
        prefix : str, optional
            File prefix for saving density figure, by default ""
        vr : list, optional
            Min/Max for plotting, by default [0.5,6.5]

        Returns
        -------
        img :
            Handle to Canvas on ax

        Notes
        -----
        'ax' must be instantiated with keyword arg: projection="polar"
        """
        vmin, vmax = vr

        if isinstance(event, pd.Series):
            # need to convert to dataFrame for prediction
            point_satLat = event["SatLat"]
            point_theta = np.arctan2(event["sin_SatMagLT"], event["cos_SatMagLT"])
            evt = event.to_frame().transpose()
        else:
            evt = event
            point_satLat = event["SatLat"].item()
            point_theta = np.arctan2(event["sin_SatMagLT"].item(), event["cos_SatMagLT"].item())

        denPred = rf.predict(evt[fgeo_col])[0]
        pred = rf.predict(grid)

        ax.yaxis.set_major_locator(plt.MaxNLocator(5))

        if north:
            ax.set_rlim(bottom=90, top=0)
        else:
            ax.set_rlim(bottom=-90, top=0)

        img = ax.pcolormesh(self.thtMesh, satlat_mesh, pred.reshape(self.n_lat,self.n_mlt), vmin=vmin, vmax=vmax, cmap=plt.cm.inferno)

        #ax.set_title(f"Predicted density at point: {denPred:.2f}")
        if not(hasattr(self, "dial_lbls")):
            self.dial_lbls = [24./360.*float(item.get_text()[:-1]) for item in ax.get_xticklabels()]
        ax.set_xticklabels(self.dial_lbls)
        if (north == (point_satLat >= 0.)):
            ax.scatter(point_theta, point_satLat, s=10, c="red")

        return img

    def plot_shap(self, ax, event, shap_val, shap_name, satlat_mesh, north: bool, vr=[]):
        if len(vr):
            min_shap, max_shap = vr
        else:
            min_shap = shap_val.min()
            max_shap = shap_val.max()

        if isinstance(event, pd.Series):
            point_satLat = event["SatLat"]
            point_theta = np.arctan2(event["sin_SatMagLT"], event["cos_SatMagLT"])
        else:
            point_satLat = event["SatLat"].item()
            point_theta = np.arctan2(event["sin_SatMagLT"].item(), event["cos_SatMagLT"].item())

        ax.yaxis.set_major_locator(plt.MaxNLocator(5))

        if north:
            ax.set_rlim(bottom=90, top=0)
        else:
            ax.set_rlim(bottom=-90, top=0)

        mn = shap_val.mean()
        img = ax.pcolormesh(self.thtMesh, satlat_mesh, shap_val.reshape(self.n_lat,self.n_mlt), vmin=min_shap, vmax=max_shap, cmap=plt.cm.inferno)
        ax.scatter(point_theta, point_satLat, s=10, c="cyan")
        ax.set_xticklabels(self.dial_lbls)
        ax.set_title(f"Factor: {shap_name}, Mean: {mn:.2f}")

        return img

    def plot_geo_den(self, ax, rf, event, grid, vr=[0.5,6.5]):
        vmin_den,vmax_den = vr

        if isinstance(event, pd.Series):
            point_satLat = event["SatLat"]
            point_theta = np.arctan2(event["sin_SatMagLT"], event["cos_SatMagLT"])
            evt_date = event['DateTime']
        else:
            point_satLat = event["SatLat"].item()
            point_theta = np.arctan2(event["sin_SatMagLT"].item(), event["cos_SatMagLT"].item())
            evt_date = event['DateTime'].item()

        point_MLT = 24.*point_theta/(2*np.pi)

        # Assuming MLT is relative to GEO subsolar longitude
        date = evt_date.strftime("%Y-%m-%d %H:%M:%S")
        _, evt_sbslon = subsol(evt_date)
        point_satLon = (point_MLT - 12)*15. + evt_sbslon

        predGEO = rf.predict(grid)

        ax.coastlines()
        ax.gridlines(draw_labels=True)
        img0 = ax.pcolormesh(self.geoLon_msh, self.geoLat_msh, predGEO.reshape(self.n_lat*2, self.n_mlt), vmin=vmin_den, vmax=vmax_den, cmap=plt.cm.inferno, alpha=0.95)
        ax.scatter(point_satLon, point_satLat, s=10, c="red")
        ax.set_title(f"Density at: {date}", size=16)
        #ax.axvline(evt_sbslon, ls='-.',lw=1.5,color='#FFC300')
        ax.axvline(evt_sbslon, ls='-.',lw=1.5,color='#3fd7f3')
        return img0

    def plot_geo_night(self, ax, event):
        if isinstance(event, pd.Series):
            evt_date = event['DateTime']
        else:
            evt_date = event['DateTime'].item()
        ax.coastlines()
        ax.gridlines(draw_labels=False)
        ax.add_feature(Nightshade(evt_date, alpha=0.5))
