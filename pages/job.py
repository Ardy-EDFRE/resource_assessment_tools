import rasterio
import rasterio.mask
from rasterio.io import MemoryFile
# from tasks import app
import pyproj
import pandas as pd
import numpy as np
import geopandas
import math
from functools import reduce
import operator
from sector import Sector
import angle as angle
import csv
import time

def createResultTxtFiles(outputResultsFileName):
    try:
        csvfile = open(outputResultsFileName, 'w',newline='')
    except:
        csvfile = open(outputResultsFileName, 'wb')
    filewriter = csv.writer(csvfile)
    filewriter.writerow(('target_turbine_fid','target_met_fid','pairPassesIEC'))
    csvfile.close()

    outputResultsFileDetailsName = outputResultsFileName[:-4]+'_details.csv'
    try:
        csvfile = open(outputResultsFileDetailsName, 'w',newline='')
    except:
        csvfile = open(outputResultsFileDetailsName, 'wb')
    filewriter = csv.writer(csvfile)
    filewriter.writerow(('pairPassesIEC','target_turbine_fid','target_met_fid','center','sector','angle',
                         'actual_slope_method','actual_slope','max_slope','actual_terrain_variation','max_terrain_variation'))
    csvfile.close()

def getParamsFromFile(paramsFiles):
    pairLines = []
    pairsFile = open(paramsFiles["pair_path"],"r+")
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
    params['target_met_fid'] =plList[5]
    params['target_met_x'] =float(plList[6])
    if plList[7][-1:] == '\n':
        params['target_met_y'] = float(plList[7][:-1])
    else:
        params['target_met_y'] = float(plList[7])

    return params

def results2csv(pairResults, outputResultsFileName):
    # write detail output
    outputResultsFileDetailsName = outputResultsFileName[:-4]+'_details.csv'
    try:
        outputResultsFileDetails = open(outputResultsFileDetailsName, 'a',newline='')
    except:
        outputResultsFileDetails = open(outputResultsFileDetailsName, 'ab')
    filewriter = csv.writer(outputResultsFileDetails)

    pairPassesIEC = 'PASSES'
    for p in pairResults:
        if p['max_terrain_variation'] is None:
            tmpString = (str(p['pass_IEC_Test']),str(p['target_turbine_fid']),str(p['target_met_fid']),str(p['centered_on']),
                         p['name'], str(p['angle']).replace(',','-'), p['actual_slope_method'],str(p['actual_slope']),str(p['max_slope']))
        else:
            tmpString = (str(p['pass_IEC_Test']),str(p['target_turbine_fid']),str(p['target_met_fid']),str(p['centered_on']),
                         p['name'],str(p['angle']).replace(',','-'),p['actual_slope_method'],str(p['actual_slope']),str(p['max_slope']),
                         str(p['actual_terrain_variation']),str(round(p['max_terrain_variation'],4)))
        filewriter.writerow(tmpString)

        if not p['pass_IEC_Test']:
            pairPassesIEC = 'DOES NOT PASS'
    outputResultsFileDetails.close()

    # write summary output
    try:
        outputResultsFile = open(outputResultsFileName, 'a',newline='')
    except:
        outputResultsFile = open(outputResultsFileName, 'ab')
    filewriter = csv.writer(outputResultsFile)
    filewriter.writerow((str(pairResults[0]['target_turbine_fid']),str(pairResults[0]['target_met_fid']),str(pairPassesIEC)))
    outputResultsFile.close()

def format_sectors(origin, centered_on, include_angles, exclude_angles, target_turbine_fid, target_met_fid, L, H, D):
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
            upper_distance_bound=2*L,
            include=True,
            max_terrain_variation=1/3*(H-0.5*D),
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
                    lower_distance_bound=2*L,
                    upper_distance_bound=4*L,
                    include=True,
                    max_terrain_variation=2/3*(H-0.5*D),
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
                    lower_distance_bound=4*L,
                    upper_distance_bound=8*L,
                    include=True,
                    max_terrain_variation=H-0.5*D,
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
                    lower_distance_bound=8*L,
                    upper_distance_bound=16*L,
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
                lower_distance_bound=2*L,
                upper_distance_bound=4*L,
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
    raster_crs = raster_elevation.crs # Pass CRS of image from rasterio
    df['raster_coords'] = df.geometry.apply(lambda point: pyproj.transform(turbine_crs, raster_crs, point.x, point.y))
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
    dist = math.sqrt(difX**2 + difY**2)
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
    while i < len(l)-1:
        l[i] = adjust_angle_forward(l[i])
        l[i+1] = adjust_angle_forward(l[i+1])
        if overlap(l[i], l[i + 1]): # if the ith and i+1th elements overlap
            merged = merge(l[i], l[i+1]) # merge
            l[i] = merged # replace original
            del l[i+1] # delete i+1th
        else:
            i+=1 # else keep moving
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

    # sector nodata mask
    raster_noDataValue = raster.nodata
    nodataMask = (rasterData!=raster_noDataValue)

    # raster cell size
    rasterCellSize = out_transform[0]

    # raster extent
    extentL_X, extentU_Y = raster.transform * (0, 0)
    extentR_X, extentL_Y = raster.transform * (raster.width, raster.height)

    # number of cells from the sector.origin to the 0,0
    xCellsFromLeft2Origin = np.absolute(np.floor((sector.origin['X'] - extentL_X)/float(rasterCellSize)))
    yCellsFromLeft2Origin = np.absolute(np.floor((sector.origin['Y'] - extentU_Y)/float(rasterCellSize)))

    # create a X,Y index matrix center in sector.origin
    # used for two things: apply the plane formula and create a distance to origin matrix
    xDimension = raster.width
    yDimension = raster.height
    xxInd = np.arange(0-xCellsFromLeft2Origin, xDimension-xCellsFromLeft2Origin, 1)
    yyInd = np.arange(0-yCellsFromLeft2Origin, yDimension-yCellsFromLeft2Origin, 1)
    xxMatrixInd, yyMatrixInd = np.meshgrid(xxInd, yyInd)

    if sector.actual_slope_method=='plane_slope':
        # use slope of the plane and terrain variation
        slope_interpolated_plane_and_terrain_variation(sector, rasterData, nodataMask, raster_noDataValue, rasterCellSize,
                                                       xCellsFromLeft2Origin, yCellsFromLeft2Origin, xxMatrixInd, yyMatrixInd)

    elif sector.actual_slope_method=='maximum_slope':
        # use maximum slope from the sector.origin to any point of the terrain
        slope_perc_from_origin_to_all_other_pnts(sector, rasterData, nodataMask, rasterCellSize, xxInd, yyInd)

    else:
        sector.pass_IEC_Test = True
        return sector.to_dict()


    # evaluate if the sector passes the IEC test
    sector.evaluate()

    return sector.to_dict()


def slope_interpolated_plane_and_terrain_variation(sector, rasterData, nodataMask, raster_noDataValue, rasterCellSize,
                                                   xCellsFromLeft2Origin, yCellsFromLeft2Origin, xxMatrixInd, yyMatrixInd):
    """
    Create an interpolation plane centered in the sector.origin and calculate
    plane slope and max terrain difference from the plane
    """

    # extract terrain grid values (x,y,z) in grid coordinates centered in the sector.origin (turb or met)
    rasX, rasY, rasZ = get_sector_terrain_as_xyz_grid_surface_centeredInOrigin(sector, rasterData, raster_noDataValue,
                                                                               xCellsFromLeft2Origin, yCellsFromLeft2Origin)

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

    #using gradients
    #https://stackoverflow.com/questions/34003993/generating-gradient-map-of-2d-array
    # gradient is (f(x+1)-f(x-1)/distance, so vgrad[0]**2 should be (vgrad[0]/cell)**2
    vgrad = np.gradient(outRas)
    slp_perc = np.sqrt((vgrad[0].mean()/float(rasterCellSize))**2 + (vgrad[1].mean()/float(rasterCellSize))**2) * 100

    sector.actual_slope = slp_perc

def slope_perc_from_origin_to_all_other_pnts(sector, rasterData, nodataMask, rasterCellSize, xxInd, yyInd):
    """
    Get the maximum slope in percentage from the sector origin to all other terrain points
    """
    # create a distance to the sector origin grid
    yyIndT = yyInd.reshape(yyInd.shape[0],1)
    dist_from_origin = rasterCellSize * (np.sqrt((xxInd)**2 + (yyIndT)**2))

    # calculate the slopes to each grid element
    # we can't divide by 0 in the origin, that's why the np.divide
    difElev = rasterData - np.float(sector.origin['Z'])
    slopes = np.where(nodataMask, np.abs(np.divide(difElev , dist_from_origin, out=np.zeros_like(difElev), where=dist_from_origin!=0)), np.NaN)
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
    Useful debugging tool. Plot a the terrain as X,Y,Z scatter and
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

#@app.task(queue='terrain_assessment', default_retry_delay=10)
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

    df['LD_target_turbine'] = df.apply(lambda x: get_utm_distance({'X':x['geometry'].x,'Y':x['geometry'].y}, target_turbine)/x['rotor_diameter'], axis=1)
    df['LD_target_met'] = df.apply(lambda x: get_utm_distance({'X':x['geometry'].x,'Y':x['geometry'].y}, target_met)/x['rotor_diameter'], axis=1)

    sectors_df = get_sectors(target_turbine, target_met, df)
    sectors_list = sectors_df.angle_tuple.to_list()
    exclude_angles = angle.condense(sectors_list)
    if len(exclude_angles)==1 and exclude_angles[0][0] == exclude_angles[0][1]:
        # weird case when there is no include angles
        # (ie include_angles=[(28, 28)] should be none. In reality [(28, 28)] is the exclude angle)
        include_angles = None
    else:
        include_angles = angle.invert_angles(exclude_angles)

    L = get_utm_distance(target_turbine, target_met)
    H = target_turbine['hub_height']
    D = target_turbine['rotor_diameter']
    #print(L, H, D)

    # create sectors based on the angles and L distances
    turbine_sectors = format_sectors(target_turbine,'Turbine',include_angles,exclude_angles,
                                     target_turbine['fid'],target_met['fid'], L, H, D)

    met_sectors = format_sectors(target_met,'Met',include_angles,exclude_angles,
                                 target_turbine['fid'], target_met['fid'],L, H, D)

    sectors = reduce(operator.concat, turbine_sectors + met_sectors)
    output= []

    for sector in sectors:
##        sector.terrain = raster_elevation
##        res = evaluate_sector(sector.to_dict(serialize=True))
##        output.append(res)
        res = evaluate_sector(sector, raster_elevation)
        output.append(res)

    return output


if __name__ == '__main__':

    processWholeShapeFile = True

    if processWholeShapeFile:
        # process all pairs
        startTime = time.time()

        # testing pairs 110RD 80HH 2.3RD
        # G:\Projects\USA_North\Slate_Creek\03_Wind\035_Operational Analysis\20200302_NPC\SlateCreek_IEC_data.mxd
        # "G:\Projects\USA_North\Slate_Creek\03_Wind\035_Operational Analysis\20200302_NPC\iec_SlateCreek_WTGs_nad83utm14\mets_buff_2_3RD.shp"

        paramsFiles = {"turbine_shapefile_path": r'G:\Projects\USA_North\Slate_Creek\03_Wind\035_Operational Analysis\20200302_NPC\SlateCreek_WTGs_nad83utm14.shp',
                  "raster_path": r'G:\Projects\USA_North\Slate_Creek\05_GIS\053_Data\Raster\dem_clip',
                  "pair_path": r'G:\Projects\USA_North\Slate_Creek\03_Wind\035_Operational Analysis\20200302_NPC\iec_SlateCreek_WTGs_nad83utm14\pairs.csv'}

        # create txt file to hold output results
        outputResultsFileName = r"C:\Users\jose.castillejo\Desktop\JL_rag-tools-iec-ed2-terrain-assessment\assets\pairs_results.csv"
        createResultTxtFiles(outputResultsFileName)

        # get pairs from file
        pairLines = getParamsFromFile(paramsFiles)
        counter = 1
        for pl in pairLines[1:]:
            params = createParams(pl)

            # run the IEC test on this pair
            pairResults = process_pair(params)

            results2csv(pairResults, outputResultsFileName)
            print('{}/{} Pair turb: {} - met: {}'.format(counter, len(pairLines[1:]),params['target_turbine_fid'], params['target_met_fid']))
            counter+=1

        elapsed_time = time.time() - startTime
        elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        print ("Total time: " + str(elapsed_time))

    else:
        # process a single pair
        params = {
            "turbine_shapefile_path": r'C:\Users\jose.castillejo\Desktop\rag-tools-iec-ed2-terrain-assessment\data\turbines.shp',
            "raster_path": r'C:\Users\jose.castillejo\Desktop\rag-tools-iec-ed2-terrain-assessment\data\dem_10m',
            'target_turbine_fid': 0,
            'target_turbine_x': 651730.929,
            'target_turbine_y': 2917381.201,
            'target_turbine_rotor_diameter': 120.0,
            'target_turbine_hub_height': 92.0,
            'target_met_fid': 0,
            'target_met_x': 651896.787,
            'target_met_y': 2917155.117
            }

        process_pair(params)

    print (' **** END ****')