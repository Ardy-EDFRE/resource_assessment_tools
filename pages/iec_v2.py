import matplotlib.pyplot as plt
import rasterio
import rasterio.mask
from rasterio.io import MemoryFile
import pyproj
import pandas as pd
import numpy as np
import geopandas
import math
from functools import reduce
import operator
import rasterio.crs
from sector import Sector
import angle as angle
import csv
import time
import streamlit as st
import folium
from streamlit_folium import folium_static
import warnings
from shapely.geometry import Polygon
import os


warnings.filterwarnings('ignore')


def app():
    st.markdown("# IEC Terrain Assessment")

    st.markdown("**How it works**")

    st.markdown("For each sector, there are rules governing whether or not the sector passes the test")

    st.markdown(f"""    In this step we will:
    1. Find the slope of the plane of best fit given a terrain file
    2. Find the terrain variation of this plane
    3. Return a True/False pass grade for the sector
            """)

    st.markdown("For each sector, there are rules governing whether or not the sector passes the test")

    st.markdown("We need all the files into UTM WGS84 coordinates")

    # BUSINESS LOGIC
    def createResultTxtFiles(outputResultsFileName):
        try:
            csvfile = open(outputResultsFileName, 'w', newline='')
        except:
            csvfile = open(outputResultsFileName, 'wb')
        filewriter = csv.writer(csvfile)
        filewriter.writerow(('target_turbine_fid', 'target_met_fid', 'pairPassesIEC'))
        csvfile.close()

        outputResultsFileDetailsName = outputResultsFileName[:-4] + '_details.csv'
        try:
            csvfile = open(outputResultsFileDetailsName, 'w', newline='')
        except:
            csvfile = open(outputResultsFileDetailsName, 'wb')
        filewriter = csv.writer(csvfile)
        filewriter.writerow(('pairPassesIEC', 'target_turbine_fid', 'target_met_fid', 'center', 'sector', 'angle',
                             'actual_slope_method', 'actual_slope', 'max_slope', 'actual_terrain_variation',
                             'max_terrain_variation'))
        csvfile.close()

    def getParamsFromFile(paramsFiles):
        pairLines = []
        pairsFile = open(paramsFiles["pair_path"], "r+")
        pairLines = pairsFile.readlines()
        pairsFile.close()

        return pairLines

    def createParams(pl):
        plList = pl.split(',')
        params = {}
        params['turbine_shapefile_path'] = paramsFiles['turbine_shapefile_path']
        params['raster_path'] = paramsFiles['raster_path']
        params['target_turbine_fid'] = plList[0]
        params['target_turbine_x'] = float(plList[1])
        params['target_turbine_y'] = float(plList[2])
        params['target_turbine_rotor_diameter'] = float(plList[3])
        params['target_turbine_hub_height'] = float(plList[4])
        params['target_met_fid'] = plList[5]
        params['target_met_x'] = float(plList[6])
        if plList[7][-1:] == '\n':
            params['target_met_y'] = float(plList[7][:-1])
        else:
            params['target_met_y'] = float(plList[7])

        return params

    def results2csv(pairResults, outputResultsFileName):
        # write detail output
        outputResultsFileDetailsName = outputResultsFileName[:-4] + '_details.csv'
        try:
            outputResultsFileDetails = open(outputResultsFileDetailsName, 'a', newline='')
        except:
            outputResultsFileDetails = open(outputResultsFileDetailsName, 'ab')
        filewriter = csv.writer(outputResultsFileDetails)

        pairPassesIEC = 'PASSES'
        for p in pairResults:
            if p['max_terrain_variation'] is None:
                tmpString = (
                    str(p['pass_IEC_Test']), str(p['target_turbine_fid']), str(p['target_met_fid']),
                    str(p['centered_on']),
                    p['name'], str(p['angle']).replace(',', '-'), p['actual_slope_method'], str(p['actual_slope']),
                    str(p['max_slope']))
            else:
                tmpString = (
                    str(p['pass_IEC_Test']), str(p['target_turbine_fid']), str(p['target_met_fid']),
                    str(p['centered_on']),
                    p['name'], str(p['angle']).replace(',', '-'), p['actual_slope_method'], str(p['actual_slope']),
                    str(p['max_slope']),
                    str(p['actual_terrain_variation']), str(round(p['max_terrain_variation'], 4)))
            filewriter.writerow(tmpString)

            if not p['pass_IEC_Test']:
                pairPassesIEC = 'DOES NOT PASS'
        outputResultsFileDetails.close()

        # write summary output
        try:
            outputResultsFile = open(outputResultsFileName, 'a', newline='')
        except:
            outputResultsFile = open(outputResultsFileName, 'ab')
        filewriter = csv.writer(outputResultsFile)
        filewriter.writerow(
            (str(pairResults[0]['target_turbine_fid']), str(pairResults[0]['target_met_fid']), str(pairPassesIEC)))
        outputResultsFile.close()

    def format_sectors(origin, centered_on, include_angles, exclude_angles, target_turbine_fid, target_met_fid, L, H,
                       D):
        sectors = []
        center_sector = [
            Sector(
                name="CENTER",
                target_met_fid=target_met_fid,
                target_turbine_fid=target_turbine_fid,
                actual_slope_method='plane_slope',
                origin=origin,
                centered_on=centered_on,
                angle=None,
                lower_distance_bound=0,
                upper_distance_bound=2 * L,
                include=True,
                max_terrain_variation=1 / 3 * (H - 0.5 * D),
                max_slope=3,
                type='circle',
            )
        ]
        sectors.append(center_sector)

        if include_angles is not None:
            for angle in include_angles:
                include_sectors = [
                    Sector(
                        name="A INCLUDE",
                        target_met_fid=target_met_fid,
                        target_turbine_fid=target_turbine_fid,
                        actual_slope_method='plane_slope',
                        origin=origin,
                        centered_on=centered_on,
                        angle=angle,
                        lower_distance_bound=2 * L,
                        upper_distance_bound=4 * L,
                        include=True,
                        max_terrain_variation=2 / 3 * (H - 0.5 * D),
                        max_slope=5,
                        type='segment'
                    ),
                    Sector(
                        name="B INCLUDE",
                        target_met_fid=target_met_fid,
                        target_turbine_fid=target_turbine_fid,
                        actual_slope_method='plane_slope',
                        origin=origin,
                        centered_on=centered_on,
                        angle=angle,
                        lower_distance_bound=4 * L,
                        upper_distance_bound=8 * L,
                        include=True,
                        max_terrain_variation=H - 0.5 * D,
                        max_slope=10,
                        type='segment'
                    ),
                    Sector(
                        name="C INCLUDE",
                        target_met_fid=target_met_fid,
                        target_turbine_fid=target_turbine_fid,
                        actual_slope_method='maximum_slope',
                        origin=origin,
                        centered_on=centered_on,
                        angle=angle,
                        lower_distance_bound=8 * L,
                        upper_distance_bound=16 * L,
                        include=True,
                        max_terrain_variation=None,
                        max_slope=10,
                        type='segment'
                    )
                ]
                sectors.append(include_sectors)

        for angle in exclude_angles:
            exclude_sectors = [
                Sector(
                    name="A EXCLUDE",
                    target_met_fid=target_met_fid,
                    target_turbine_fid=target_turbine_fid,
                    actual_slope_method='maximum_slope',
                    origin=origin,
                    centered_on=centered_on,
                    angle=angle,
                    lower_distance_bound=2 * L,
                    upper_distance_bound=4 * L,
                    include=False,
                    max_terrain_variation=None,
                    max_slope=10,
                    type='segment'
                )
            ]
            sectors.append(exclude_sectors)

        return sectors

    def transform_by_coordinate_system(df, raster_elevation):
        """
        Using the input CRS and raster data, convert X,Y points to common CRS
        """
        turbine_crs = df.geometry.crs
        raster_crs = raster_elevation.crs  # Pass CRS of image from rasterio
        df['raster_coords'] = df.geometry.apply(
            lambda point: pyproj.transform(turbine_crs, raster_crs, point.x, point.y))
        df['X'] = df['geometry'].apply(lambda point: point.x)
        df['Y'] = df['geometry'].apply(lambda point: point.y)
        # JL shouldn't you assign the raster_crs to the df dataframe itself df.crs = raster_crs
        return df

    def get_turbine_elevation(target, raster_elevation):
        """
        Given a raster of elevation data and a set of data about turbine location,
        find the Z of the turbines base elevation from the its (X, Y) position.
        """
        # read and extract raster
        elevation_band_1 = raster_elevation.read(1)
        # What is the corresponding row and column in our image?
        # row, col = elevation.index(x, y) # spatial --> image coordinates
        return {'Z': float(elevation_band_1[raster_elevation.index(target['X'], target['Y'])])}

    def get_utm_distance(po, p):
        """
        Given Universal Transverse Mercator points p_0 and p, find the Euclidean distance between them
        """
        difX = p['X'] - po['X']
        difY = p['Y'] - po['Y']
        dist = math.sqrt(difX ** 2 + difY ** 2)
        return dist

    def get_sectors(target_turbine, target_met, df):
        """
        For both the target turbine and the target met tower, find a list of
        sectors (angles, expressed in tuple form) which contain a significant obstacle
        and should thus be excluded from analysis.
        """
        df_target_turbine = df[(2 < df['LD_target_turbine']) & (df['LD_target_turbine'] < 20)]
        df_target_met = df[(2 < df['LD_target_met']) & (df['LD_target_met'] < 20)]
        df_target_turbine['alpha'] = df_target_turbine.apply(
            lambda x: alpha_sector(x["LD_target_turbine"]),
            axis=1
        )
        df_target_met['alpha'] = df_target_met.apply(
            lambda x: alpha_sector(x["LD_target_met"]),
            axis=1
        )
        df_target_turbine['direction'] = df_target_turbine.apply(
            lambda x: get_direction_degrees(target_turbine, {'X': x['geometry'].x, 'Y': x['geometry'].y}),
            axis=1
        )
        df_target_met['direction'] = df_target_met.apply(
            lambda x: get_direction_degrees(target_met, {'X': x['geometry'].x, 'Y': x['geometry'].y}),
            axis=1
        )
        df_target_turbine['angle_tuple'] = df_target_turbine.apply(
            lambda x: calc_angles(x["direction"], x["alpha"]),
            axis=1
        )
        df_target_met['angle_tuple'] = df_target_met.apply(
            lambda x: calc_angles(x["direction"], x["alpha"]),
            axis=1
        )
        return pd.concat([df_target_turbine, df_target_met])

    def alpha_sector(LD):
        """
        Calculate alpha, the angle (in radians) which is blocked out by the ratio
        L/D = Distance from object / Diameter of object
        theta = 1.3 * arctan(2.5D/L + 0.15 )
        """
        DL = float(1) / LD
        arctan = math.atan(2.5 * DL + 0.15)
        degree = math.degrees(arctan)
        alpha = 1.3 * degree + 10
        return alpha

    def calc_angles(direction, alpha):
        """
        Given a direction and an alpha, return the angle as a tuple
        """
        alphaOri = direction - alpha / 2
        if (alphaOri < 0):
            alphaOri = 360 + alphaOri
        alphaFin = direction + alpha / 2
        if (alphaFin > 360):
            alphaFin = alphaFin - 360
        return int(round(alphaOri)), int(round(alphaFin))

    def get_direction_degrees(po, p):
        """
        Given two points p_0 and p, find the direction of the vector connection them

        d = (90 - atan2(p.y - p_0.y, p.x - p_0.y) mod 360
        """
        angle = math.degrees(math.atan2(
            p['Y'] - po['Y'],
            p['X'] - po['X']
        ))
        bearing2 = (90 - angle) % 360
        return bearing2

    def condense(list_of_tuples):
        """
        Given a list of tuples, find the overlapping pairs and
        condense them into a shorter list of tuples where the start and end values
        represent the smallest and largest ends of the overlapping segments

        >>> condense([(20, 25), (18, 19), (14, 15), (7, 15), (5, 6), (2, 3), (1, 2)])
        [(1, 3), (5, 6), (7, 15), (18, 19), (20, 25)]
        >>> condense([(1, 2), (2, 11), (5, 6), (5, 10), (7, 15), (14, 15), (18, 19), (20, 25)])
        [(1, 15), (18, 19), (20, 25)]
        >>> condense([(162, 236), (49, 115), (32, 80), (93, 137), (59, 99), (290, 10), (300, 12)])
        [(32, 137), (162, 236), (290, 12)]
        """
        l = sorted(list_of_tuples)
        overlap = lambda x, y: True if y[0] <= x[1] else False
        merge = lambda x, y: tuple((min(x[0], y[0]), max(x[1], y[1])))
        adjust_angle_forward = lambda x: tuple((x[0], x[1] + 360)) if x[1] < x[0] else x
        adjust_angle_backward = lambda x: tuple((x[0], x[1] - 360)) if x[1] > 360 else x
        i = 0
        while i < len(l) - 1:
            l[i] = adjust_angle_forward(l[i])
            l[i + 1] = adjust_angle_forward(l[i + 1])
            if overlap(l[i], l[i + 1]):  # if the ith and i+1th elements overlap
                merged = merge(l[i], l[i + 1])  # merge
                l[i] = merged  # replace original
                del l[i + 1]  # delete i+1th
            else:
                i += 1  # else keep moving
            l[i] = adjust_angle_backward(l[i])

        # manual check for overlapping angles on both sides
        if overlap(l[-1], l[0]) and len(l) > 1:
            merged = tuple((l[-1][0], l[0][1]))
            l[-1] = merged
            del l[0]

        return l

    def evaluate_sector(sector, raster):
        """
        For each sector, there are rules governing whether or not the sector passes the test:
        +----+-------------+-----------------+----------------------------------------+----------------------------+
        |    | Distance    | Maximum Slope   | Maximum Terrain Variation From Plane   | Sector                     |
        |----+-------------+-----------------+----------------------------------------+----------------------------|
        |  0 | (0, 2*L)    | <3% plane       | <1/3 (H - 0.5*D)                       | 360                        |
        |  1 | (2*L, 4*L)  | <5% plane       | <2/3 (H - 0.5*D)                       | Measurement sector         |
        |  2 | (2*L, 4*L)  | <10% max pnt    | Not applicable                         | Outside measurement sector |
        |  3 | (4*L, 8*L)  | <10% plane      | <(H - 0.5*D)                           | Measurement sector         |
        |  4 | (4*L, 8*L)  | Not applicable  | Not applicable                         | Outside measurement sector |
        |  5 | (8*L, 16*L) | <10% max pnt    | <4/3 (H - 0.5*D)                       | Measurement sector         |
        |  6 | (8*L, 16*L) | Not applicable  | Not applicable                         | Outside measurement sector |
        +----+-------------+-----------------+----------------------------------------+----------------------------+

        In this step we will:
        1. Find the slope of the plane of best fit given a terrain file
        2. Find the terrain variation of this plane
        3. Return a True/False pass grade for the sector
        """

        # open the raster
        rasterData, out_transform = rasterio.mask.mask(raster, [sector.polygon], crop=True)
        out_meta = raster.meta.copy()
        out_meta.update(
            {
                "driver": "GTiff",
                "height": rasterData.shape[1],
                "width": rasterData.shape[2],
                "transform": out_transform
            }
        )

        # substitue the whole raster for the raster clipped to the sector extent
        raster = MemoryFile().open(**out_meta)
        raster.write(rasterData)

        #     Saving the Raster
        #     with rasterio.open(r'G:\Projects\USA_North\Lincoln_Valley\05_GIS\053_Data\working\IEC\circle_terrain_2.tif', 'w', **out_meta) as dst:
        #         dst.write_band(1, rasterData[0].astype(rasterio.float32))

        #     sector nodata mask
        #     avoid this
        #     investigate how to fix none value for tif files & why the file gives raster.nodata = None
        #     if raster.nodata:
        #         raster_noDataValue = 0
        #     else:
        #         raster_noDataValue = raster.nodata

        raster_noDataValue = raster.nodata
        nodataMask = (rasterData != raster_noDataValue)

        # raster cell size
        rasterCellSize = out_transform[0]

        # raster extent
        extentL_X, extentU_Y = raster.transform * (0, 0)
        extentR_X, extentL_Y = raster.transform * (raster.width, raster.height)

        # number of cells from the sector.origin to the 0,0
        xCellsFromLeft2Origin = np.absolute(np.floor((sector.origin['X'] - extentL_X) / float(rasterCellSize)))
        yCellsFromLeft2Origin = np.absolute(np.floor((sector.origin['Y'] - extentU_Y) / float(rasterCellSize)))

        # create a X,Y index matrix center in sector.origin
        # used for two things: apply the plane formula and create a distance to origin matrix
        xDimension = raster.width
        yDimension = raster.height
        xxInd = np.arange(0 - xCellsFromLeft2Origin, xDimension - xCellsFromLeft2Origin, 1)
        yyInd = np.arange(0 - yCellsFromLeft2Origin, yDimension - yCellsFromLeft2Origin, 1)
        xxMatrixInd, yyMatrixInd = np.meshgrid(xxInd, yyInd)

        if sector.actual_slope_method == 'plane_slope':
            # use slope of the plane and terrain variation
            slope_interpolated_plane_and_terrain_variation(sector, rasterData, nodataMask, raster_noDataValue,
                                                           rasterCellSize,
                                                           xCellsFromLeft2Origin, yCellsFromLeft2Origin, xxMatrixInd,
                                                           yyMatrixInd)
            # print(sector.actual_slope_method)

        elif sector.actual_slope_method == 'maximum_slope':
            # use maximum slope from the sector.origin to any point of the terrain
            slope_perc_from_origin_to_all_other_pnts(sector, rasterData, nodataMask, rasterCellSize, xxInd, yyInd)
            # print(sector.actual_slope_method)

        else:
            sector.pass_IEC_Test = True
            return sector.to_dict()

        # evaluate if the sector passes the IEC test
        sector.evaluate()

        return sector.to_dict()

    def slope_interpolated_plane_and_terrain_variation(sector, rasterData, nodataMask, raster_noDataValue,
                                                       rasterCellSize,
                                                       xCellsFromLeft2Origin, yCellsFromLeft2Origin, xxMatrixInd,
                                                       yyMatrixInd):
        """
        Create an interpolation plane centered in the sector.origin and calculate
        plane slope and max terrain difference from the plane
        """

        # extract terrain grid values (x,y,z) in grid coordinates centered in the sector.origin (turb or met)
        rasX, rasY, rasZ = get_sector_terrain_as_xyz_grid_surface_centeredInOrigin(sector, rasterData,
                                                                                   raster_noDataValue,
                                                                                   xCellsFromLeft2Origin,
                                                                                   yCellsFromLeft2Origin)

        # Interpolate a plane through the origin
        rasXY = np.column_stack([rasX, rasY])
        coeffs = np.linalg.lstsq(rasXY, rasZ, rcond=None)[0]

        # Recreate the interpolated raster from the regression model (numpy corrdinates)
        outRas = coeffs[0] * xxMatrixInd + coeffs[1] * yyMatrixInd + sector.origin['Z']

        ##    # convert numpy outRas to rasterio raster
        ##
        ##    # use out_meta as a template and change to float because the interpolated values are float
        ##    out_meta['dtype'] = "float64"
        ##    with rasterio.open(r'G:\Projects\USA_West\Oso_Grande\05_GIS\053_Data\working\IEC_container_test\test_6x5\inter_plane.tif', 'w', **out_meta) as dst:
        ##        dst.write(outRas, 1)

        ##    rasterWindow = rasterio.windows.from_bounds(extentL_X, extentL_Y, extentR_X, extentU_Y,
        ##                                                transform=out_meta['transform'], width = out_meta['width'], height = out_meta['height'])
        ##    with rasterio.open(r'G:\Projects\USA_West\Oso_Grande\05_GIS\053_Data\working\IEC_container_test\test_6x5\inter_plane.tif') as dst:
        ##        plane = dst.read(1, window = rasterWindow)

        # outRas comes from rasterData, so they have the same dimensions in both grid and spatial coordiantes
        difDemAdjTrend = np.where(nodataMask, rasterData - outRas, np.NaN)
        sector.actual_above_plane_terrain_variation = np.nanmax(difDemAdjTrend)
        sector.actual_below_plane_terrain_variation = - np.nanmin(difDemAdjTrend)
        sector.actual_terrain_variation = max(np.nanmax(difDemAdjTrend), - np.nanmin(difDemAdjTrend))

        # using gradients
        # https://stackoverflow.com/questions/34003993/generating-gradient-map-of-2d-array
        # gradient is (f(x+1)-f(x-1)/distance, so vgrad[0]**2 should be (vgrad[0]/cell)**2
        vgrad = np.gradient(outRas)
        slp_perc = np.sqrt(
            (vgrad[0].mean() / float(rasterCellSize)) ** 2 + (vgrad[1].mean() / float(rasterCellSize)) ** 2) * 100

        sector.actual_slope = slp_perc

    def slope_perc_from_origin_to_all_other_pnts(sector, rasterData, nodataMask, rasterCellSize, xxInd, yyInd):
        """
        Get the maximum slope in percentage from the sector origin to all other terrain points
        """
        # create a distance to the sector origin grid
        yyIndT = yyInd.reshape(yyInd.shape[0], 1)
        dist_from_origin = rasterCellSize * (np.sqrt((xxInd) ** 2 + (yyIndT) ** 2))

        # calculate the slopes to each grid element
        # we can't divide by 0 in the origin, that's why the np.divide
        difElev = rasterData - np.double(sector.origin['Z'])
        slopes = np.where(nodataMask, np.abs(
            np.divide(difElev, dist_from_origin, out=np.zeros_like(difElev), where=dist_from_origin != 0)), np.NaN)
        sector.actual_slope = np.nanmax(slopes) * 100

    def get_sector_terrain_as_xyz_grid_surface_centeredInOrigin(sector, rasterData, raster_noDataValue,
                                                                xCellsFromLeft2Origin, yCellsFromLeft2Origin):
        """
        Apply polygon mask on raster to select only the sector in question. Then,
        create a terrain array of valid data by filtering out noDataValue and flattening
        the array to a list of [(x0,y0,z0), (x1, y1, z1), ... (xn, yn, zn)] tuples for n points
        in the terrain surface.
        The arrays are numpy indexes and elevations centered in the origin (turbine or met)
        """

        rasX = []
        rasY = []
        rasZ = []

        for row in range(rasterData[0].shape[0]):
            for column in range(rasterData[0].shape[1]):
                if rasterData[0][row, column] != raster_noDataValue:
                    # get index of cells different from nodata
                    rasZ.append(rasterData[0][row, column] - sector.origin['Z'])
                    rasY.append(row)
                    rasX.append(column)

        # move the X and Y of the turb to the origin in array coordinates
        rasX = rasX - xCellsFromLeft2Origin
        rasY = rasY - yCellsFromLeft2Origin

        return rasX, rasY, rasZ

    def write_shapefile(polygons, crs):
        polygons_df = geopandas.GeoDataFrame(crs=crs, geometry=polygons)
        polygons_df.to_file(filename='polygon.shp', driver="ESRI Shapefile")

    def plot_terrain_and_plane(X, Y, Z, sector, coeffs):
        """
        Useful debugging tool. Plot the terrain as X,Y,Z scatter and
        the plane of best fit by meshgrid
        """
        # Create the trend surface using the regression model
        xx, yy = np.meshgrid(
            np.arange(min(X), max(X), 1),
            np.arange(min(Y), max(Y), 1)
        )

        zz = coeffs[0] * xx + coeffs[1] * yy
        plt3d = plt.figure().gca(projection='3d')
        plt3d.plot_surface(xx, yy, zz, alpha=0.2)
        plt3d.scatter(X, Y, Z)
        plt.show()

    def process_pair(params):
        try:
            turbine_shapefile_path = params['turbine_shapefile_path']
            raster_path = params['raster_path']
            target_turbine = {
                'fid': params['target_turbine_fid'],
                'X': params['target_turbine_x'],
                'Y': params['target_turbine_y'],
                'rotor_diameter': params['target_turbine_rotor_diameter'],
                'hub_height': params['target_turbine_hub_height']
            }
            target_met = {
                'fid': params['target_met_fid'],
                'X': params['target_met_x'],
                'Y': params['target_met_y']
            }

        except:
            raise Exception("""
                Could not run task, params malformed or missing keys from param dict:
                    turbine_shapefile_path
                    raster_path
                    target_turbine_fid,
                    target_turbine_x,
                    target_turbine_y,
                    target_turbine_rotor_diameter,
                    target_turbine_hub_height,
                    target_met_fid,
                    target_met_x,
                    target_met_y
            """)

        df = geopandas.read_file(turbine_shapefile_path)
        df['rotor_diameter'] = target_turbine['rotor_diameter']
        df['hub_height'] = target_turbine['hub_height']

        # open the raster
        raster_elevation = rasterio.open(raster_path)

        # read the turbine and met tower elevations from the raster
        target_turbine.update(get_turbine_elevation(target_turbine, raster_elevation))
        target_met.update(get_turbine_elevation(target_met, raster_elevation))

        # remove this and force all inputs to be in the same UTM
        ##        df = transform_by_coordinate_system(df, raster_elevation)

        df['LD_target_turbine'] = df.apply(
            lambda x: get_utm_distance({'X': x['geometry'].x, 'Y': x['geometry'].y}, target_turbine) / x[
                'rotor_diameter'],
            axis=1)
        df['LD_target_met'] = df.apply(
            lambda x: get_utm_distance({'X': x['geometry'].x, 'Y': x['geometry'].y}, target_met) / x['rotor_diameter'],
            axis=1)

        sectors_df = get_sectors(target_turbine, target_met, df)
        sectors_list = sectors_df.angle_tuple.to_list()
        exclude_angles = angle.condense(sectors_list)
        if len(exclude_angles) == 1 and exclude_angles[0][0] == exclude_angles[0][1]:
            # weird case when there is no include angles
            # (ie include_angles=[(28, 28)] should be none. In reality [(28, 28)] is the exclude angle)
            include_angles = None
        else:
            include_angles = angle.invert_angles(exclude_angles)

        L = get_utm_distance(target_turbine, target_met)
        H = target_turbine['hub_height']
        D = target_turbine['rotor_diameter']
        #     print(L, H, D)

        # create sectors based on the angles and L distances
        turbine_sectors = format_sectors(target_turbine, 'Turbine', include_angles, exclude_angles,
                                         target_turbine['fid'], target_met['fid'], L, H, D)

        met_sectors = format_sectors(target_met, 'Met', include_angles, exclude_angles,
                                     target_turbine['fid'], target_met['fid'], L, H, D)

        sectors = reduce(operator.concat, turbine_sectors + met_sectors)
        output = []

        for sector in sectors:
            ##        sector.terrain = raster_elevation
            ##        res = evaluate_sector(sector.to_dict(serialize=True))
            ##        output.append(res)
            res = evaluate_sector(sector, raster_elevation)
            output.append(res)

        return output

    def save_uploaded_file(file_content, file_name):
        """
        Save the uploaded file to a temporary directory
        """
        import tempfile
        import os
        import uuid

        _, file_extension = os.path.splitext(file_name)
        file_id = str(uuid.uuid4())
        file_path = os.path.join(tempfile.gettempdir(), f"{file_id}{file_extension}")

        with open(file_path, "wb") as file:
            file.write(file_content.getbuffer())

        return file_path

    @st.cache
    def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')

    @st.cache
    def save_uploadedfile(uploadedfile):
        with open(os.path.join("tempDir", uploadedfile.name), "wb") as f:
            f.write(uploadedfile.getbuffer())
        return st.success("Saved File:{} to tempDir".format(uploadedfile.name))

    # G:\Projects\USA_North\Livingston_County\03_Wind\030_Associated_Met_Masts\Location Selection\nominated_turbines_v3.csv
    # G:\Projects\USA_North\Livingston_County\05_GIS\053_Data\Turbines_v45_UTM_NAD83_20220330.zip

    # G:\Projects\USA_North\Livingston_County\05_GIS\053_Data\DEM_10m_Large_Clip_NAD_UTM.tif
    # G:\Projects\CAN_West\Red_Rock\03_Wind\032_Other_Analyses\Met_tower_siting\IEC_Lv22_Results_ed2.csv

    # showing the maps
    turbine_map = folium.Map(tiles='OpenStreetMap')
    mets_map = folium.Map(tiles='OpenStreetMap')
    sectors_map = folium.Map(tiles='OpenStreetMap')

    # Add custom base maps to folium
    basemaps = {
        'Google Maps': folium.TileLayer(
            tiles='https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',
            attr='Google',
            name='Google Maps',
            overlay=True,
            control=True
        ),
        'Google Satellite': folium.TileLayer(
            tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
            attr='Google',
            name='Google Satellite',
            overlay=True,
            control=True
        ),
        'Google Terrain': folium.TileLayer(
            tiles='https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}',
            attr='Google',
            name='Google Terrain',
            overlay=True,
            control=True
        ),
        'Google Satellite Hybrid': folium.TileLayer(
            tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
            attr='Google',
            name='Google Satellite',
            overlay=True,
            control=True
        ),
        'Esri Satellite': folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Esri Satellite',
            overlay=True,
            control=True
        )
    }

    # Add custom basemaps
    basemaps['Google Maps'].add_to(mets_map)
    basemaps['Google Satellite Hybrid'].add_to(mets_map)
    basemaps['Google Maps'].add_to(turbine_map)
    basemaps['Google Satellite Hybrid'].add_to(turbine_map)
    basemaps['Google Maps'].add_to(sectors_map)
    basemaps['Google Satellite Hybrid'].add_to(sectors_map)

    # Input CSV
    mets_turbs_pairs = st.sidebar.file_uploader("Upload Met Turbine Pairs file location", type='csv',
                                                help="Please use only the extension .csv")
    if mets_turbs_pairs:
        mets_turbs_pairs = save_uploaded_file(mets_turbs_pairs, mets_turbs_pairs.name)

    # Input Geospatial
    turbine_layout = st.sidebar.file_uploader("Upload Turbine Layout file location", type=['zip', 'kml'],
                                              help="Please zip all your shapefile extensions (.dbf, .cpg, .prj, .shp, .sbn) and upload as .zip or upload with KML as .kml")
    if turbine_layout:
        turbine_layout = save_uploaded_file(turbine_layout, turbine_layout.name)

    # Input Elevation
    elevation_raster = st.sidebar.file_uploader("Upload elevation file location", type='tif',
                                                help="Please use only use .tif. The size of the dem file (till 20km away from the project boundary).")
    if elevation_raster:
        elevation_raster = save_uploaded_file(elevation_raster, elevation_raster.name)

    # Outputs
    outputResultsFileName = st.sidebar.text_input("Write you output file name",
                                                  help="Please use only the extension .csv")

    if turbine_layout and elevation_raster:
        met_pairs_df = geopandas.read_file(mets_turbs_pairs)
        turbine_CRSCheck = geopandas.read_file(turbine_layout)
        raster_CRSCheck = rasterio.open(elevation_raster)

        if turbine_CRSCheck.crs == raster_CRSCheck.crs:
            st.write(f'Input values coordinate systems match - {turbine_CRSCheck.crs} - {raster_CRSCheck.crs}!')
        else:
            st.write(
                'Input values coordinate system do not match, please fix before continuing  -- TURBINE CRS {} &  RASTER '
                'CRS {} '.format(
                    turbine_CRSCheck.crs, raster_CRSCheck.crs))

    # display_met_pairs_map = st.sidebar.checkbox("Display Met Pairs on Map")
    # if display_met_pairs_map:
    #     met_pairs_df = met_pairs_df.astype(
    #         {"target_turbine_x": float, "target_turbine_y": float, "target_met_x": float, "target_met_y": float})
    #
    #     turbines_pairs_df = met_pairs_df.iloc[:, [met_pairs_df.columns.get_loc(c) for c in
    #                                               ['target_turbine_fid', 'target_turbine_x', 'target_turbine_y']]]
    #
    #     turbines_utm = utm.to_latlon(met_pairs_df['target_turbine_x'], met_pairs_df['target_turbine_y'], 14, northern=True)
    #
    #     turbines_pairs_df['target_met_x'] = turbines_utm[0]
    #     turbines_pairs_df['target_met_y'] = turbines_utm[1]
    #
    #     mets_unique_df = met_pairs_df.iloc[:, [met_pairs_df.columns.get_loc(c) for c in ['target_met_fid', 'target_met_x', 'target_met_y']]]
    #
    #     mets_utm = utm.to_latlon(met_pairs_df['target_met_x'], met_pairs_df['target_met_y'], 14, northern=True)
    #
    #     mets_unique_df['target_met_x'] = mets_utm[0]
    #     mets_unique_df['target_met_y'] = mets_utm[1]
    #
    #     turbines_pairs_df['geometry'] = [Point(xy) for xy in
    #                                      zip(turbines_pairs_df.target_turbine_x, turbines_pairs_df.target_turbine_y)]
    #
    #     mets_unique_df['geometry'] = [Point(xy) for xy in
    #                                   zip(mets_unique_df.target_met_x, mets_unique_df.target_met_y)]
    #
    #     turbines_pairs_df = turbines_pairs_df.set_crs(4326, allow_override=True)
    #     mets_unique_df = mets_unique_df.set_crs(4326, allow_override=True)
    #
    #
    #
    #
    #     turbines_pairs_df["long"] = turbines_pairs_df.geometry.x
    #     turbines_pairs_df["lat"] = turbines_pairs_df.geometry.y
    #     mets_unique_df["long"] = mets_unique_df.geometry.x
    #     mets_unique_df["lat"] = mets_unique_df.geometry.y
    #     turbine_pair_points = turbines_pairs_df[["lat", "long"]]
    #     turbine_pairs_list = turbine_pair_points.values.tolist()
    #     mets_points = mets_unique_df[["lat", "long"]]
    #     mets_list = mets_points.values.tolist()
    #
    #     turbines_pairs_cluster = folium.plugins.MarkerCluster().add_to(mets_map)
    #     mets_pairs_cluster = folium.plugins.MarkerCluster().add_to(mets_map)
    #
    #     for point in range(0, len(turbine_pairs_list)):
    #         turbine_icon = folium.features.CustomIcon(
    #             'https://raw.githubusercontent.com/Ardy-EDFRE/resource_assessment_tools/main/images/turbines.png',
    #             icon_size=(40, 40))
    #         folium.Marker(turbine_pairs_list[point],
    #                       popup='Turbine',
    #                       icon=turbine_icon
    #                       ).add_to(turbines_pairs_cluster)
    #
    #     for point in range(0, len(mets_list)):
    #         met_icon = folium.features.CustomIcon(
    #             'https://raw.githubusercontent.com/Ardy-EDFRE/resource_assessment_tools/main/images/met_tower.png',
    #             icon_size=(40, 40))
    #         folium.Marker(mets_list[point],
    #                       popup='Met',
    #                       icon=met_icon
    #                       ).add_to(mets_pairs_cluster)
    #
    #     bounding_box = turbines_pairs_cluster.get_bounds()
    #     mets_map.fit_bounds([bounding_box])
    #     folium_static(mets_map, width=800, height=800)

    display_turbine_layout_map = st.sidebar.checkbox("Display Turbine Layout on Map",
                                                     help="This checkbox will display a map visualization of the input Turbine shapefiles")

    if display_turbine_layout_map:
        turbine_CRSCheck = turbine_CRSCheck.to_crs("EPSG:4326")
        turbines_df = turbine_CRSCheck.loc[turbine_CRSCheck['Alternate'] == 'Primary Turbine']
        mets_df = turbine_CRSCheck.loc[turbine_CRSCheck['Alternate'] == 'Alt']
        turbines_df["long"] = turbines_df.geometry.x
        turbines_df["lat"] = turbines_df.geometry.y
        mets_df["long"] = mets_df.geometry.x
        mets_df["lat"] = mets_df.geometry.y
        turbine_points = turbines_df[["lat", "long"]]
        turbine_list = turbine_points.values.tolist()
        mets_points = mets_df[["lat", "long"]]
        mets_list = mets_points.values.tolist()

        paramsFiles = {"turbine_shapefile_path": turbine_layout,
                       "raster_path": elevation_raster,
                       "pair_path": mets_turbs_pairs}

        # create txt file to hold output results
        createResultTxtFiles(outputResultsFileName)

        # get pairs from file
        pairLines = getParamsFromFile(paramsFiles)

        paired_results_polys = []

        if pairLines:
            line_list = len(pairLines)
            line_count = line_list - 1
            line_count = line_count * -1
            for pl in pairLines[line_count:]:
                params = createParams(pl)
                # run the IEC test on this pair
                pairResults = process_pair(params)
                paired_results_polys.append(pairResults[0]['polygon'])

        turbines_cluster = folium.plugins.MarkerCluster().add_to(turbine_map)
        mets_cluster = folium.plugins.MarkerCluster().add_to(turbine_map)

        for point in range(0, len(turbine_list)):
            turbine_icon = folium.features.CustomIcon(
                'https://raw.githubusercontent.com/Ardy-EDFRE/resource_assessment_tools/main/images/turbines.png',
                icon_size=(40, 40))
            folium.Marker(turbine_list[point],
                          popup="Turbine",
                          icon=turbine_icon).add_to(turbines_cluster)

        for point in range(0, len(mets_list)):
            met_icon = folium.features.CustomIcon(
                'https://raw.githubusercontent.com/Ardy-EDFRE/resource_assessment_tools/main/images/met_tower.png',
                icon_size=(40, 40))
            folium.Marker(mets_list[point],
                          popup='Met',
                          icon=met_icon).add_to(mets_cluster)

        st.write("Sectors Dataframe")

        import shapely
        from shapely.ops import unary_union

        sectors_gdf = unary_union(paired_results_polys)
        sectors_gdf = geopandas.GeoDataFrame(index=[0], crs=4326, geometry=[sectors_gdf])
        sectors_gdf = sectors_gdf.to_crs(epsg='4326', inplace=True)

        st.write(sectors_gdf)

        # folium.GeoJson(data=sectors_gdf['geometry']).add_to(sectors_map)
        # folium_static(sectors_map, width=800, height=800)
        #
        # bounding_box = turbines_cluster.get_bounds()
        # turbine_map.fit_bounds([bounding_box])
        # folium_static(turbine_map, width=800, height=800)

    run_iec = st.sidebar.button("Run IEC Terrain Assessment",
                                help="This will run the process for evaluation sectors and generate an output for display & download")
    if run_iec:
        # process all pairs
        startTime = time.time()

        # testing pairs 110RD 80HH 2.3RD
        # G:\Projects\USA_North\Slate_Creek\03_Wind\035_Operational Analysis\20200302_NPC\SlateCreek_IEC_data.mxd
        # G:\Projects\USA_North\Slate_Creek\03_Wind\035_Operational Analysis\20200302_NPC\iec_SlateCreek_WTGs_nad83utm14\mets_buff_2_3RD.shp

        paramsFiles = {"turbine_shapefile_path": turbine_layout,
                       "raster_path": elevation_raster,
                       "pair_path": mets_turbs_pairs}

        # create txt file to hold output results
        createResultTxtFiles(outputResultsFileName)

        # get pairs from file
        pairLines = getParamsFromFile(paramsFiles)

        counter = 1

        if pairLines:
            line_list = len(pairLines)
            line_count = line_list - 1
            line_count = line_count * -1
            for pl in pairLines[line_count:]:
                params = createParams(pl)
                # run the IEC test on this pair
                pairResults = process_pair(params)

                results2csv(pairResults, outputResultsFileName)
                st.write(
                    '{}/{} Pair turb: {} - met: {}'.format(counter, len(pairLines[1:]), params['target_turbine_fid'],
                                                           params['target_met_fid']))
                counter += 1

        elapsed_time = time.time() - startTime
        elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        st.write("Total time: " + str(elapsed_time))

        sector_results_output = pd.read_csv(outputResultsFileName)
        details_output_csv = pd.read_csv(outputResultsFileName[:-4] + '_details.csv')

        st.write(sector_results_output)
        st.write(details_output_csv)

        convert_csv = convert_df(details_output_csv)

        st.download_button(
            label="Download data as CSV",
            data=convert_csv,
            file_name=outputResultsFileName,
            mime='text/csv',
        )
        #
        # if os.path.expanduser(f'~/Downloads/{outputResultsFileName[:-4]}' + '_details.csv'):
        #     path1 = os.path.expanduser(f'~/Downloads/{outputResultsFileName[:-4]}' + '_details.csv')
        #     csv_abs_path = os.path.dirname(os.path.abspath(outputResultsFileName))
        #     path2 = os.path.expanduser(csv_abs_path)
        #     shutil.move(path1, path2)

    st.image("https://raw.githubusercontent.com/Ardy-EDFRE/resource_assessment_tools/main/images/edf_small_logo.png",
             width=50)

