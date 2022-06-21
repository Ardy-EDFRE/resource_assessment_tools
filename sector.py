import shapely.geometry
import math
import numpy as np


class Sector(object):
    def __init__(
        self,
        name=None,
        origin=None,
        target_met_fid=None,
        target_turbine_fid=None,
        hub_height=None,
        diameter=None,
        centered_on=None,
        angle=None,
        L=None,
        lower_distance_bound=None,
        upper_distance_bound=None,
        include=True,
        actual_slope_method=None,
        max_terrain_variation=None,
        max_slope=None,
        actual_terrain_variation=None,
        actual_above_plane_terrain_variation=None,
        actual_below_plane_terrain_variation=None,
        actual_slope=None,
        terrain_assessment=None,
        slope_assessment=None,
        pass_IEC_Test=None,
        type='segment',
        polygon=None,

    ):
        self._type = type
        self._origin = origin
        self._target_met_fid = target_met_fid
        self._target_turbine_fid = target_turbine_fid
        self._name = name
        self._hub_height = hub_height
        self._diameter = diameter
        self._type = type
        self._centered_on = centered_on
        self._angle = angle
        self._L = L
        self._lower_distance_bound = lower_distance_bound
        self._upper_distance_bound = upper_distance_bound
        self._include = include
        self._actual_slope_method = actual_slope_method
        self._max_terrain_variation = max_terrain_variation
        self._actual_above_plane_terrain_variation = actual_above_plane_terrain_variation
        self._actual_below_plane_terrain_variation = actual_below_plane_terrain_variation
        self._actual_terrain_variation = actual_terrain_variation
        self._max_slope = max_slope
        self._actual_slope = actual_slope
        self._terrain_assessment = terrain_assessment
        self._slope_assessment = slope_assessment
        self._pass_IEC_Test = pass_IEC_Test
        self._get_polygon = polygon
        # self._polygon = polygon
        self._coords = None
        self._plane = None


    @property
    def origin(self):
        if self._origin is not None:
            return self._origin
        else:
            raise Exception('Sector needs an origin')

    @origin.setter
    def origin(self, value):
        self._origin = value

    @property
    def centered_on(self):
        if self._centered_on is not None:
            return self._centered_on

    @property
    def include(self):
        if self._include is not None:
            return self._include
        else:
            raise Exception('Sector needs an include')

    @property
    def actual_slope_method(self):
        if self._actual_slope_method is not None:
            return self._actual_slope_method
        else:
            raise Exception('Sector needs actual_slope_method to be defined')

    @property
    def L(self):
        if self._L is not None:
            return self._L
        else:
            raise Exception('Sector needs a L distance')

    @property
    def lower_distance_bound(self):
        if self._lower_distance_bound is not None:
            return self._lower_distance_bound
        else:
            raise Exception('Sector needs a lower distance bound')

    @property
    def upper_distance_bound(self):
        if self._upper_distance_bound is not None:
            return self._upper_distance_bound
        else:
            raise Exception('Sector needs an upper distance bound')

    @property
    def max_terrain_variation(self):
        return self._max_terrain_variation

    @property
    def max_slope(self):
        if self._max_slope is not None:
            return self._max_slope
        else:
            raise Exception('Sector needs a max slope')

    @property
    def actual_slope(self):
        if self._actual_slope:
            return round(self._actual_slope, 4)

    @actual_slope.setter
    def actual_slope(self, value):
        if isinstance(value, float):
            self._actual_slope=value
        else:
            raise ValueError("actual_slope must be type 'float'")

    @property
    def actual_terrain_variation(self):
        if self._actual_terrain_variation:
            return round(self._actual_terrain_variation, 4)

    @actual_terrain_variation.setter
    def actual_terrain_variation(self, value):
        if isinstance(value, float):
            self._actual_terrain_variation=value
        else:
            raise ValueError("actual_terrain_variation must be type 'float'")
        
    @property
    def actual_above_plane_terrain_variation(self):
        if self._actual_above_plane_terrain_variation:
            return round(self._actual_above_plane_terrain_variation, 4)

    @actual_above_plane_terrain_variation.setter
    def actual_above_plane_terrain_variation(self, value):
        if isinstance(value, float):
            self._actual_above_plane_terrain_variation=value
        else:
            raise ValueError("actual_above_plane_terrain_variation must be type 'float'")

    @property
    def actual_below_plane_terrain_variation(self):
        if self._actual_below_plane_terrain_variation:
            return round(self._actual_below_plane_terrain_variation, 4)

    @actual_below_plane_terrain_variation.setter
    def actual_below_plane_terrain_variation(self, value):
        if isinstance(value, float):
            self._actual_below_plane_terrain_variation=value
        else:
            raise ValueError("actual_below_plane_terrain_variation must be type 'float'")

    @property
    def pass_IEC_Test(self):
        if self._pass_IEC_Test:
            return self._pass_IEC_Test

    @property
    def get_polygon(self):
        """
        Given an angle, a target turbine (with attached metadata),
        and a sector distance (eg. 2L, 4L, etc.) create an annulus section
        which geometrically represents the sector
        """
        if self._type=='segment':
            sector_vertex_array = []

            # initial bearing
            arc_origin_lower_distance_bound = create_point_from_distance_and_angle(
                self._origin['X'],
                self._origin['Y'],
                self._lower_distance_bound,
                self.angle[0]
            )
            arc_origin_upper_distance_bound = create_point_from_distance_and_angle(
                self.origin['X'],
                self.origin['Y'],
                self._upper_distance_bound,
                self.angle[0]
            )
            sector_vertex_array.append(arc_origin_lower_distance_bound)
            sector_vertex_array.append(arc_origin_upper_distance_bound)

            # external arc
            tmpAngle0 = self.angle[0]
            if self.angle[0]!=self.angle[1]:
                tmpAngle1=self.angle[1]
            else:
                tmpAngle1 = self.angle[0] + 360
                
            intermediate_angle_list = get_angle_range(tmpAngle0, tmpAngle1, 0.2)
            intermediate_angle_list = intermediate_angle_list[1:]

            for intermediate_point in intermediate_angle_list:
                intermediate_point = rotate_point(
                    math.radians(intermediate_point),
                    self.origin['X'],
                    self.origin['Y'],
                    self.origin['X'],
                    self.origin['Y'] + self._upper_distance_bound
                )
                sector_vertex_array.append(intermediate_point)

            # final bearing
            arc_final_lower_distance_bound = create_point_from_distance_and_angle(
                self.origin['X'],
                self.origin['Y'],
                self._lower_distance_bound,
                tmpAngle1
            )
            arc_final_upper_distance_bound = create_point_from_distance_and_angle(
                self.origin['X'],
                self.origin['Y'],
                self._upper_distance_bound,
                tmpAngle1
            )
            sector_vertex_array.append(arc_final_lower_distance_bound)
            sector_vertex_array.append(arc_final_upper_distance_bound)

            # internal arc
            #intermediate_angle_list.reverse()
            intermediate_angle_list = intermediate_angle_list[::-1]
            for intermediate_point in intermediate_angle_list:
                intermediate_point = rotate_point(
                    math.radians(intermediate_point),
                    self.origin['X'],
                    self.origin['Y'],
                    self.origin['X'],
                    self.origin['Y'] + self._lower_distance_bound
                )
                sector_vertex_array.append(intermediate_point)

            # close and create the polygon
            sector_vertex_array.append(arc_origin_lower_distance_bound)
            origin = shapely.geometry.Point(self.origin['X'], self.origin['Y'])
            center_circle = origin.buffer(1)
            sector_vertex_array += [(p[0], p[1]) for p in center_circle.exterior.coords]
            sector_polygon = shapely.geometry.Polygon((p[0], p[1]) for p in sector_vertex_array)
            return sector_polygon
        else:
            origin = shapely.geometry.Point(self.origin['X'], self.origin['Y'])
            circle = origin.buffer(self._upper_distance_bound)
            return circle

    @property
    def polygon(self):
        if self._polygon is not None:
            return self._polygon
        else:
            self._polygon = self.get_polygon()
            return self._polygon

    @polygon.setter
    def polygon(self, value):
        if isinstance(value, shapely.geometry.polygon.Polygon):
            self._polygon=value
        else:
            raise ValueError("Sector.polygon must be of type shapely.geometry.polygon.Polygon")

    @property
    def coords(self):
        if self._polygon is not None:
            self._coords = [tuple((x,y)) for x,y in self._polygon.exterior.coords]
            return self._coords
        else:
            self._polygon = self.get_polygon()
            return self.coords

    @property
    def angle(self):
        if self._type == 'circle':
            return (0, 360)
        elif self._type == 'segment':
            return self._angle
        else:
            raise AttributeError('Sector "type" needs to be set to either "segment" or "circle"')

    def get_polygon_old(self):
        """
        Given an angle, a target turbine (with attached metadata),
        and a sector distance (eg. 2L, 4L, etc.) create an annulus section
        which geometrically represents the sector
        """
        sector_vertex_array = []

        # initial bearing
        arc_origin_lower_distance_bound = create_point_from_distance_and_angle(
            self._origin['X'],
            self._origin['Y'],
            self._lower_distance_bound,
            self.angle[0]
        )
        arc_origin_upper_distance_bound = create_point_from_distance_and_angle(
            self.origin['X'],
            self.origin['Y'],
            self._upper_distance_bound,
            self.angle[0]
        )
        sector_vertex_array.append(arc_origin_lower_distance_bound)
        sector_vertex_array.append(arc_origin_upper_distance_bound)

        # external arc
        intermediate_angle_list = get_angle_range(self.angle[0], self.angle[1], 0.2)
        intermediate_angle_list = intermediate_angle_list[1:]

        for intermediate_point in intermediate_angle_list:
            intermediate_point = rotate_point(
                math.radians(intermediate_point),
                self.origin['X'],
                self.origin['Y'],
                self.origin['X'],
                self.origin['Y'] + self._upper_distance_bound
            )
            sector_vertex_array.append(intermediate_point)

        # final bearing
        arc_final_lower_distance_bound = create_point_from_distance_and_angle(
            self.origin['X'],
            self.origin['Y'],
            self._lower_distance_bound,
            self.angle[1]
        )
        arc_final_upper_distance_bound = create_point_from_distance_and_angle(
            self.origin['X'],
            self.origin['Y'],
            self._upper_distance_bound,
            self.angle[1]
        )
        sector_vertex_array.append(arc_final_lower_distance_bound)
        sector_vertex_array.append(arc_final_upper_distance_bound)

        # internal arc
        #intermediate_angle_list.reverse()
        intermediate_angle_list = intermediate_angle_list[::-1]
        for intermediate_point in intermediate_angle_list:
            intermediate_point = rotate_point(
                math.radians(intermediate_point),
                self.origin['X'],
                self.origin['Y'],
                self.origin['X'],
                self.origin['Y'] + self._lower_distance_bound
            )
            sector_vertex_array.append(intermediate_point)

        # close and create the polygon
        sector_vertex_array.append(arc_origin_lower_distance_bound)
        sector_polygon = shapely.geometry.Polygon((p[0], p[1]) for p in sector_vertex_array)
        return sector_polygon

    def evaluate(self):
        """
        Evaluate sector
        """
        if self._actual_slope_method=='plane_slope':
            # interpolation plane and terrain variation
            if self._max_terrain_variation:
                if self.actual_terrain_variation > self._max_terrain_variation:
                    # Fail if actual > max allowed
                    self._terrain_assessment = False
                else:
                    self._terrain_assessment = True

            if self._max_slope:
                if self.actual_slope > self._max_slope:
                    # Fail if actual > max allowed
                    self._slope_assessment = False
                else:
                    self._slope_assessment = True

            if self._terrain_assessment and self._slope_assessment:
                self._pass_IEC_Test = True
            else:
                self._pass_IEC_Test = False
                
        elif self._actual_slope_method=='maximum_slope':
            # slope to all points
            if self.actual_slope > self._max_slope:
                # Fail if actual > max allowed
                self._slope_assessment = False
                self._pass_IEC_Test = False
            else:
                self._slope_assessment = True
                self._pass_IEC_Test = True

        else:
            # no requirements for this sector
            self._pass_IEC_Test = True
            

    def to_dict(self):
        return {
            "origin" : self._origin,
            "name": self._name,
            "target_met_fid": self._target_met_fid,
            "target_turbine_fid": self._target_turbine_fid,
            "hub_height" : self._hub_height,
            "diameter" : self._diameter,
            "centered_on": self._centered_on,
            "angle" : self._angle,
            "lower_distance_bound" : self._lower_distance_bound,
            "upper_distance_bound" : self._upper_distance_bound,
            "include" : self._include,
            "actual_slope_method": self._actual_slope_method,
            "max_terrain_variation" : self._max_terrain_variation,
            "max_slope" : self._max_slope,
            "actual_terrain_variation": self.actual_terrain_variation,
            "actual_above_terrain_variation": self.actual_above_plane_terrain_variation,
            "actual_below_terrain_variation": self.actual_below_plane_terrain_variation,
            "actual_slope": self.actual_slope,
            "terrain_assessment": self._terrain_assessment,
            "slope_assessment": self._slope_assessment,
            "pass_IEC_Test": self._pass_IEC_Test,
            "type" : self._type,
            "polygon": self._polygon
        }

def create_point_from_distance_and_angle(x, y, distance, angle):
    """
    Given a distance and angle, create an xy point
    """
    disp_x, disp_y = (distance * math.sin(math.radians(angle)), distance * math.cos(math.radians(angle)))
    pnt = (x + disp_x, y + disp_y)
    return pnt

def rotate_point(angle_rad, x0, y0, x, y):
    """
    Given an angle in radians and a pair of (x0, y0), (x, y)
    coords, rotate them about the origin and create a new point
    """
    angle_rad = angle_rad * -1.
    x = x - x0
    y = y - y0
    rotated_point = (
        x0 + x*math.cos(angle_rad) - y*math.sin(angle_rad),
        y0 + x*math.sin(angle_rad) + y*math.cos(angle_rad)
    )
    return rotated_point

def get_angle_range(angle_0, angle_1, step):
    """
    Given a range from angle_0 to angle_1, return a list of intemediary angles
    split on step size.
    >>> get_angle_range(0, 90, 45)
    array([0, 45, 90])
    >>> get_angle_range(0, 90, 10)
    array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
    >>> get_angle_range(90, 40, 60)
    array([90, 150, 210, 270, 330, 30, 40])
    """
    if angle_0 > angle_1:
        angle_1 += 360
        arange = np.arange(angle_0, angle_1, step)
        out = np.array([])
        for x in arange:
            if x > 360:
                out = np.append(out, [x - 360])
            else:
                out = np.append(out, [x])
        angle_1 -= 360
    else:
        out = np.arange(angle_0, angle_1, step)
    return np.append(out, [angle_1])
