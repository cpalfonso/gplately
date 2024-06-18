import logging
import re
import warnings
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Union,
)

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.axes import Axes
from matplotlib.cbook import CallbackRegistry
from matplotlib.transforms import Affine2D
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPolygon,
    Point,
    Polygon,
    box,
)
from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry
from shapely.ops import linemerge, substring

from .io_utils import get_geometries as _get_geometries

logger = logging.getLogger("gplately")


def _project_geometry(geometry, projection, transform=None):
    """Project shapely geometries onto a certain Cartopy CRS map projection.

    Uses a coordinate system ("transform"), if given.

    Parameters
    ----------
    geometry : shapely.geometry.base.BaseGeometry
        An instance of a shapely geometry.
    projection : cartopy.crs.Transform,
        The projection of the plot.
    transform : cartopy.crs.Transform or None, default None
        If the plot is projected, a `transform` value is usually needed.
        Frequently, the appropriate value is an instance of
        `cartopy.crs.PlateCarree`.

    Returns
    -------
    projected : list
        The provided shapely geometries projected onto a Cartopy CRS map projection.
    """
    if transform is None:
        transform = ccrs.PlateCarree()
    result = [projection.project_geometry(geometry, transform)]
    projected = []
    for i in result:
        if isinstance(i, BaseMultipartGeometry):
            projected.extend(list(i.geoms))
        else:
            projected.append(i)
    return projected


def _calculate_triangle_vertices(
    geometries,
    width,
    spacing,
    height,
    polarity,
):
    """Generate vertices of subduction teeth triangles.

    Triangle bases are set on shapely BaseGeometry trench instances with their apexes
    pointing in directions of subduction polarity. Triangle dimensions are set by a
    specified width, spacing and height (either provided by the user or set as default
    values from _tessellate_triangles). The teeth are returned as shapely polygons.

    Parameters
    ----------
    geometries : list of shapely geometries (instances of the
        shapely.geometry.base.BaseGeometry or shapely.geometry.base.BaseMultipartGeometry
        class)
        Trench geometries projected onto a certain map projection (using a
        coordinate system if specified), each with identified subduction polarities.
        Teeth triangles will be generated only on the BaseGeometry instances.
    width : float
        The (approximate) width of the subduction teeth. If a projection is
        used, this value will be in projected units.
    spacing : float,
        The spacing between the subduction teeth. As with
        `width` and `height`, this value should be given in projected units.
    height : float, default None
        The height of the subduction teeth. This value should also be given in projected
        units.
    polarity : {"left", "right"}
        The subduction polarity of the shapely geometries.

    Returns
    -------
    triangles : list of shapely polygons
        The subduction teeth generated along the supplied trench geometries.
    """
    if isinstance(geometries, BaseGeometry):
        geometries = [geometries]
    triangles = []
    for geometry in geometries:
        if not isinstance(geometry, BaseGeometry):
            continue

        length = geometry.length
        tessellated_x = []
        tessellated_y = []

        for distance in np.arange(spacing, length, spacing):
            point = Point(geometry.interpolate(distance))
            tessellated_x.append(point.x)
            tessellated_y.append(point.y)
        tessellated_x = np.array(tessellated_x)
        tessellated_y = np.array(tessellated_y)

        for i in range(len(tessellated_x) - 1):
            normal_x = tessellated_y[i] - tessellated_y[i + 1]
            normal_y = tessellated_x[i + 1] - tessellated_x[i]
            normal = np.array((normal_x, normal_y))
            normal_mag = np.sqrt((normal**2).sum())
            if normal_mag == 0:
                continue
            normal *= height / normal_mag
            midpoint = np.array((tessellated_x[i], tessellated_y[i]))
            if polarity == "right":
                normal *= -1.0
            apex = midpoint + normal

            next_midpoint = np.array((tessellated_x[i + 1], tessellated_y[i + 1]))
            line_vector = np.array(next_midpoint - midpoint)
            line_vector_mag = np.sqrt((line_vector**2).sum())
            line_vector /= line_vector_mag
            triangle_point_a = midpoint + width * 0.5 * line_vector
            triangle_point_b = midpoint - width * 0.5 * line_vector
            triangle_points = np.array(
                (
                    triangle_point_a,
                    triangle_point_b,
                    apex,
                )
            )
            triangles.append(Polygon(triangle_points))
    return triangles


def _parse_polarity(polarity):
    """Ensure subduction polarities have valid strings as labels - either "left", "l", "right" or "r".

    The geometries' subduction polarities are either provided by the user in plot_subduction_teeth
    or found automatically in a geopandas.GeoDataFrame column by _find_polarity_column, if such a
    column exists.

    Parameters
    ----------
    polarity : {"left", "l", "right", "r"}
        The subduction polarity of the geometries (either set by the user or found automatically
        from the geometries' data frame).

    Returns
    -------
    polarity : {"left", "right"}
        Returned if the provided polarity string is one of {"left", "l", "right", "r"}. "l" and "r"
        are classified and returned as "left" and "right" respectively.

    Raises
    ------
    TypeError
        If the provided polarity is not a string type.
    ValueError
        If the provided polarity is not valid ("left", "l", "right" or "r").
    """
    if not isinstance(polarity, str):
        raise TypeError("Invalid `polarity` argument type: {}".format(type(polarity)))
    if polarity.lower() in {"left", "l"}:
        polarity = "left"
    elif polarity.lower() in {"right", "r"}:
        polarity = "right"
    else:
        valid_args = {"left", "l", "right", "r"}
        err_msg = "Invalid `polarity` argument: {}".format(
            polarity
        ) + "\n(must be one of: {})".format(valid_args)
        raise ValueError(err_msg)
    return polarity


def _find_polarity_column(columns):
    """Search for a 'polarity' column in a geopandas.GeoDataFrame to extract subduction
    polarity values.

    Subduction polarities can be used for tessellating subduction teeth.

    Parameters
    ----------
    columns : geopandas.GeoDataFrame.columns.values instance
        A list of geopandas.GeoDataFrame column header strings.

    Returns
    -------
    column : list
        If found, returns a list of all subduction polarities ascribed to the supplied
        geometry data frame.
    None
        if a 'polarity' column was not found in the data frame. In this case, subduction
        polarities will have to be manually provided to plot_subduction_teeth.

    """
    pattern = "polarity"
    for column in columns:
        if re.fullmatch(pattern, column) is not None:
            return column
    return None


def _parse_geometries(geometries):
    """Resolve a geopandas.GeoSeries object into shapely BaseGeometry and/or
    BaseMutipartGeometry instances.

    Parameters
    ----------
    geometries : geopandas.GeoDataFrame, sequence of shapely geometries, or str
        If a `geopandas.GeoDataFrame` is given, its geometry attribute
        will be used. If `geometries` is a string, it must be the path to
        a file, which will be loaded with `geopandas.read_file`. Otherwise,
        `geometries` must be a sequence of shapely geometry objects (instances
        of the `shapely.geometry.base.BaseGeometry` class).

    Returns
    -------
    out : list
        Resolved shapely BaseMutipartGeometry and/or BaseGeometry instances.
    """
    geometries = _get_geometries(geometries)
    if isinstance(geometries, gpd.GeoSeries):
        geometries = list(geometries)

    # Explode multi-part geometries
    # Weirdly the following seems to be faster than
    # the equivalent explode() method from GeoPandas:
    out = []
    for i in geometries:
        if isinstance(i, BaseMultipartGeometry):
            out.extend(list(i.geoms))
        else:
            out.append(i)
    return out


def _clean_polygons(data, projection):
    data = gpd.GeoDataFrame(data)
    data = data.explode(ignore_index=True)

    if data.crs is None:
        data.crs = ccrs.PlateCarree()

    if isinstance(
        projection,
        (
            ccrs._RectangularProjection,
            ccrs._WarpedRectangularProjection,
        ),
    ):
        central_longitude = _meridian_from_projection(projection)
        dx = 1.0e-3
        dy = 5.0e-2
        rects = (
            box(
                central_longitude - 180,
                -90,
                central_longitude - 180 + dx,
                90,
            ),
            box(
                central_longitude + 180 - dx,
                -90,
                central_longitude + 180,
                90,
            ),
            box(
                central_longitude - 180,
                -90 - dy * 0.5,
                central_longitude + 180,
                -90 + dy * 0.5,
            ),
            box(
                central_longitude - 180,
                90 - dy * 0.5,
                central_longitude + 180,
                90 + dy * 0.5,
            ),
        )
        rects = gpd.GeoDataFrame(
            {"geometry": rects},
            geometry="geometry",
            crs=ccrs.PlateCarree(),
        )
        data = data.overlay(rects, how="difference")

    projected = data.to_crs(projection)

    # If no [Multi]Polygons, return projected data
    for geom in projected.geometry:
        if isinstance(geom, (Polygon, MultiPolygon)):
            break
    else:
        return projected

    proj_width = np.abs(projection.x_limits[1] - projection.x_limits[0])
    proj_height = np.abs(projection.y_limits[1] - projection.y_limits[0])
    min_distance = np.mean((proj_width, proj_height)) * 1.0e-4

    boundary = projection.boundary
    if np.array(boundary.coords).shape[1] == 3:
        boundary = type(boundary)(np.array(boundary.coords)[:, :2])
    return _fill_all_edges(projected, boundary, min_distance=min_distance)


def _fill_all_edges(data, boundary, min_distance=None):
    data = gpd.GeoDataFrame(data).explode(ignore_index=True)

    def drop_func(geom):
        if hasattr(geom, "exterior"):
            geom = geom.exterior
        coords = np.array(geom.coords)
        return np.all(np.abs(coords) == np.inf)

    to_drop = data.geometry.apply(drop_func)
    data = (data[~to_drop]).copy()

    def filt_func(geom):
        if hasattr(geom, "exterior"):
            geom = geom.exterior
        coords = np.array(geom.coords)
        return np.any(np.abs(coords) == np.inf) or (
            min_distance is not None and geom.distance(boundary) < min_distance
        )

    to_fix = data.index[data.geometry.apply(filt_func)]
    for index in to_fix:
        fixed = _fill_edge_polygon(
            data.geometry.at[index],
            boundary,
            min_distance=min_distance,
        )
        data.geometry.at[index] = fixed
    return data


def _fill_edge_polygon(geometry, boundary, min_distance=None):
    if isinstance(geometry, BaseMultipartGeometry):
        return type(geometry)(
            [_fill_edge_polygon(i, boundary, min_distance) for i in geometry.geoms]
        )
    if not isinstance(geometry, Polygon):
        geometry = Polygon(geometry)
    coords = np.array(geometry.exterior.coords)

    segments_list = []
    segment = []
    for x, y in coords:
        if (np.abs(x) == np.inf or np.abs(y) == np.inf) or (
            min_distance is not None and boundary.distance(Point(x, y)) <= min_distance
        ):
            if len(segment) > 1:
                segments_list.append(segment)
                segment = []
            continue
        segment.append((x, y))
    if len(segments_list) == 0:
        return geometry
    segments_list = [LineString(i) for i in segments_list]

    out = []
    for i in range(-1, len(segments_list) - 1):
        segment_before = segments_list[i]
        point_before = Point(segment_before.coords[-1])

        segment_after = segments_list[i + 1]
        point_after = Point(segment_after.coords[0])

        d0 = boundary.project(point_before, normalized=True)
        d1 = boundary.project(point_after, normalized=True)
        boundary_segment = substring(boundary, d0, d1, normalized=True)

        if boundary_segment.length > 0.5 * boundary.length:
            if d1 > d0:
                seg0 = substring(boundary, d0, 0, normalized=True)
                seg1 = substring(boundary, 1, d1, normalized=True)
            else:
                seg0 = substring(boundary, d0, 1, normalized=True)
                seg1 = substring(boundary, 0, d1, normalized=True)

            if isinstance(seg0, Point) and isinstance(seg1, LineString):
                boundary_segment = seg1
            elif isinstance(seg1, Point) and isinstance(seg0, LineString):
                boundary_segment = seg0
            else:
                boundary_segment = linemerge([seg0, seg1])

        if i == -1:
            out.append(segment_before)
        out.append(boundary_segment)
        if i != len(segments_list) - 2:
            out.append(segment_after)

    return Polygon(np.vstack([i.coords for i in out]))


def _meridian_from_ax(ax):
    if hasattr(ax, "projection") and isinstance(ax.projection, ccrs.Projection):
        proj = ax.projection
        return _meridian_from_projection(projection=proj)
    return 0.0


def _meridian_from_projection(projection):
    x = np.mean(projection.x_limits)
    y = np.mean(projection.y_limits)
    return ccrs.PlateCarree().transform_point(x, y, projection)[0]


def _transform_distance_axes(d, ax, inverse=False):
    axes_bbox = ax.get_position()
    fig_bbox = ax.figure.bbox_inches  # display units (inches)

    axes_width = axes_bbox.width * fig_bbox.width
    axes_height = axes_bbox.height * fig_bbox.height

    # Take mean in case they're different somehow
    xlim = ax.get_xlim()
    x_factor = np.abs(xlim[1] - xlim[0]) / axes_width  # map units per display unit
    ylim = ax.get_ylim()
    y_factor = np.abs(ylim[1] - ylim[0]) / axes_height
    factor = 0.5 * (x_factor + y_factor)
    if inverse:
        return d / factor
    return d * factor


def plot_subduction_teeth(
    geometries,
    size,
    polarity=None,
    spacing=None,
    projection="auto",
    ax=None,
    **kwargs,
):
    """Add subduction teeth to a plot.

    The subduction polarity used for subduction teeth can be specified
    manually or detected automatically if `geometries` is a
    `geopandas.GeoDataFrame` object with a `polarity` column.

    Parameters
    ----------
    geometries : geopandas.GeoDataFrame, sequence of shapely geometries, or str
        If a `geopandas.GeoDataFrame` is given, its geometry attribute
        will be used. If `geometries` is a string, it must be the path to
        a file, which will be loaded with `geopandas.read_file`. Otherwise,
        `geometries` must be a sequence of shapely geometry objects (instances
        of the `shapely.geometry.base.BaseGeometry` class).
    size : float, default: 6.0
        Teeth size in points (alias: `markersize`).
    polarity : {"left", "l", "right", "r", None}, default None
        The subduction polarity of the geometries. If no polarity is provided,
        and `geometries` is a `geopandas.GeoDataFrame`, this function will
        attempt to find a `polarity` column in the data frame and use the
        values given there. If `polarity` is not manually specified and no
        appropriate column can be found, an error will be raised.
    spacing : float, optional
        Teeth spacing, in display units (usually inches). The default
        of `None` will choose a value based on the teeth size.
    projection : cartopy.crs.Transform, "auto", or None, default "auto"
        The projection of the plot. If the plot has no projection, this value
        can be explicitly given as `None`. The default value is "auto", which
        will acquire the projection automatically from the plot axes.
    ax : matplotlib.axes.Axes, or None, default None
        The axes on which the subduction teeth will be drawn. By default,
        the current axes will be acquired using `matplotlib.pyplot.gca`.
    **kwargs
        Any further keyword arguments will be passed to
        `gplately.SubductionTeeth`.

    Raises
    ------
    ValueError
        If `width` <= 0, or if `polarity` is an invalid value or could not
        be determined.
    """
    if kwargs.pop("width", None) is not None:
        warnings.warn(
            "`width` argument is deprecated; use `size` instead",
            DeprecationWarning,
        )
    if kwargs.pop("height", None) is not None:
        warnings.warn(
            "`height` argument is deprecated; use `aspect` ("
            "height / width) instead",
            DeprecationWarning,
        )

    if ax is None:
        ax = plt.gca()

    if projection == "auto":
        try:
            projection = ax.projection
        except AttributeError:
            projection = None
    elif isinstance(projection, str):
        raise ValueError("Invalid projection: {}".format(projection))

    if polarity is None:
        polarity_column = _find_polarity_column(geometries.columns.values)
        if polarity_column is None:
            raise ValueError(
                "Could not automatically determine polarity; "
                + "it must be defined manually instead."
            )
        left = _parse_geometries(
            geometries[geometries[polarity_column].str.lower().isin({"left", "l"})]
        )
        right = _parse_geometries(
            geometries[geometries[polarity_column].str.lower().isin({"right", "r"})]
        )
    else:
        polarity = _parse_polarity(polarity)
        if polarity == "left":
            left = _parse_geometries(geometries)
            right = []
        else:
            left = []
            right = _parse_geometries(geometries)
    return SubductionTeeth(
        left=left,
        right=right,
        ax=ax,
        size=size,
        spacing=spacing,
        **kwargs
    )


class SubductionTeeth:
    """Add subduction zone teeth to a map."""
    def __init__(
        self,
        left: Sequence[Union[LineString, MultiLineString]],
        right: Sequence[Union[LineString, MultiLineString]],
        ax: Optional[Union[Axes, GeoAxes]] = None,
        size: float = 6.0,
        aspect: float = 1.0,
        spacing: Optional[float] = None,
        color="black",
        **kwargs
    ):
        """Add subduction zone teeth to a map.

        Parameters
        ----------
        left, right : sequence of LineString or MultiLineString
            Shapely geometries representing the left- and right-polarity
            subduction zones.

        ax : matplotlib Axes or cartopy GeoAxes, optional
            The axes on which to plot the subduction zone teeth. If not specified,
            will use the current axes.

        size : float, default: 6.0
            Teeth size in points (alias: `markersize`).

        aspect : float, default: 1.0
            Aspect ratio of teeth triangles (height / width).

        spacing : float, optional
            Teeth spacing, in display units (usually inches). The default
            of `None` will choose a value based on the teeth size.

        color : str, default='black'
            The colour of the teeth (`markerfacecolor` and `markeredgecolor`).

        **kwargs :
            Further keyword arguments are passed to `matplotlib.pyplot.plot`.
            See `matplotlib` keyword arguments
            [here](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html).
        """
        if ax is None:
            ax = plt.gca()
        self._ax = ax
        self._left = self._explode_geometries(left)
        self._left_projected = None
        self._right = self._explode_geometries(right)
        self._right_projected = None
        self._aspect = float(aspect)
        self._teeth = []

        if spacing is not None:
            spacing = float(spacing)
        self._spacing = spacing

        self._plot_kw = dict(kwargs)
        keys = self._plot_kw.keys()
        if "markersize" not in keys:
            self._plot_kw["markersize"] = size
        if "facecolor" in keys:
            self._plot_kw["markerfacecolor"] = self._plot_kw.pop("facecolor")
        if "edgecolor" in keys:
            self._plot_kw["markeredgecolor"] = self._plot_kw.pop("edgecolor")
        if "markerfacecolor" not in keys:
            self._plot_kw["markerfacecolor"] = color
        if "markeredgecolor" not in keys:
            self._plot_kw["markeredgecolor"] = color

        self._triangle = mpath.Path(
            vertices=[
                (-0.5, 0),
                (0.5, 0),
                (0, self.aspect),
                (-0.5, 0),
            ]
        )

        self.ax.set_xlim(*self.ax.get_xlim())
        self.ax.set_ylim(*self.ax.get_ylim())
        self._draw_teeth()

        self._callbacks = CallbackRegistry()
        self._callback_ids = set()

        def callback_func(ax):
            return self._draw_teeth(ax)

        for event in ("xlim_changed", "ylim_changed"):
            self._callback_ids.add(self._callbacks.connect(event, callback_func))
        self._ax.callbacks = self._callbacks

    def __del__(self):
        for callback_id in self._callback_ids:
            self._callbacks.disconnect(callback_id)
        del self._ax.callbacks

    def _draw_teeth(self, ax=None):
        if ax is None:
            ax = self.ax

        if self._teeth is not None:
            for i in self._teeth:
                i.remove()
        self._teeth = []

        spacing = _transform_distance_axes(self.spacing, self.ax)
        left = self.left_projected
        right = self.right_projected

        domain = (ax.transData.inverted().transform_bbox(ax.bbox))
        domain = box(domain.x0, domain.y0, domain.x1, domain.y1)
        left = domain.intersection(left)
        right = domain.intersection(right)

        for polarity, geometries in zip(
            ("left", "right"),
            (left, right),
        ):
            geometries = self._explode_geometries(geometries)
            for geometry in geometries:
                if not isinstance(geometry, BaseGeometry):
                    continue
                if geometry.is_empty:
                    continue

                length = geometry.length
                geom_points = [Point(i) for i in geometry.coords]
                cumlen = np.concatenate(
                    (
                        [0.0],
                        np.cumsum(
                            [
                                geom_points[i + 1].distance(geom_points[i])
                                for i in range(len(geom_points) - 1)
                            ]
                        )
                    )
                )
                for distance in np.arange(spacing, length, spacing):
                    point = Point(geometry.interpolate(distance))
                    after = int(np.where(cumlen >= distance)[0][0])
                    before = after - 1
                    p1 = geom_points[before]
                    p2 = geom_points[after]
                    normal = np.array([p1.y - p2.y, p2.x - p1.x])
                    if polarity == "right":
                        normal *= -1.0
                    angle = np.arctan2(*(normal[::-1]))
                    marker = self._triangle.transformed(
                        Affine2D().rotate_deg(-90).rotate(angle)
                    )
                    p = ax.plot(point.x, point.y, marker=marker, **self.plot_kw)
                    self._teeth.extend(p)

    @staticmethod
    def _get_default_spacing(markersize: float, aspect: float) -> float:
        """Default spacing is approximately 2 times triangle width."""
        width = np.sqrt(2 * markersize / aspect)  # approximately
        width /= 72  # convert from points to inches
        spacing = 2 * width
        return spacing

    @staticmethod
    def _explode_geometries(geometries):
        if isinstance(geometries, BaseGeometry):
            geometries = [geometries]
        out = []
        for geometry in geometries:
            if geometry.is_empty:
                continue
            if isinstance(geometry, BaseMultipartGeometry):
                out.extend(
                    [
                        i for i in list(geometry.geoms)
                        if isinstance(i, BaseGeometry) and not i.is_empty
                    ]
                )
            else:
                out.append(geometry)
        return out

    @property
    def ax(self):
        return self._ax

    @property
    def figure(self):
        return self.ax.figure

    @property
    def projection(self) -> Optional[ccrs.Projection]:
        if hasattr(self.ax, "projection"):
            return self.ax.projection
        return None

    @property
    def left(self) -> List[LineString]:
        return self._left

    @property
    def left_projected(self) -> List[LineString]:
        if self.projection is None:
            return self.left
        if self._left_projected is None:
            projected = [self.projection.project_geometry(i) for i in self.left]
            projected = [
                i for i in projected
                if isinstance(i, BaseGeometry) and not i.is_empty
            ]
            projected = self._explode_geometries(projected)
            if len(projected) == 0:
                self._left_projected = []
                return self._left_projected
            merged = linemerge(projected)
            if hasattr(merged, "geometries"):
                self._left_projected = list(merged.geometries)
            else:
                self._left_projected = [merged]
        return self._left_projected

    @property
    def right(self) -> List[LineString]:
        return self._right

    @property
    def right_projected(self) -> List[LineString]:
        if self.projection is None:
            return self.right
        if self._right_projected is None:
            projected = [self.projection.project_geometry(i) for i in self.right]
            projected = [
                i for i in projected
                if isinstance(i, BaseGeometry) and not i.is_empty
            ]
            projected = self._explode_geometries(projected)
            if len(projected) == 0:
                self._right_projected = []
                return self._right_projected
            merged = linemerge(projected)
            if hasattr(merged, "geometries"):
                self._right_projected = list(merged.geometries)
            else:
                self._right_projected = [merged]
        return self._right_projected

    @property
    def spacing(self) -> float:
        if self._spacing is None:
            return self._get_default_spacing(
                self.plot_kw["markersize"],
                self.aspect,
            )
        return self._spacing

    @spacing.setter
    def spacing(self, x: Optional[float]):
        if x is not None:
            x = float(x)
        self._spacing = x
        self._draw_teeth()

    @property
    def aspect(self):
        return self._aspect

    @property
    def plot_kw(self) -> Dict[str, Any]:
        return self._plot_kw
