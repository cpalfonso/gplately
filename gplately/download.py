""" Functions for downloading assorted plate reconstruction data to use with GPlately's
main objects. Files are stored in the user's cache and can be reused after being
downloaded once. 

These data have been created and used in plate reconstruction models and studies, and
are available from public web servers (like EarthByte's webDAV server, or the GPlates 
2.3 sample dataset library).

"""
import pooch as _pooch
from pooch import os_cache as _os_cache
from pooch import retrieve as _retrieve
from pooch import HTTPDownloader as _HTTPDownloader
from pooch import Unzip as _Unzip
from pooch import Decompress as _Decompress
from matplotlib import image as _image
from .data import DataCollection
import gplately as _gplately
import pygplates as _pygplates
import re as _re
import numpy as _np

def _fetch_from_web(url):
    """Download file(s) in a given url to the 'gplately' cache folder. Processes
    compressed files using either Pooch's Unzip (if .zip) or Decompress (if .gz, 
    .xz or .bz2)."""
    def pooch_retrieve(url, processor):
        """Downloads file(s) from a URL using Pooch."""
        fnames = _retrieve(
            url=url,
            known_hash=None,  
            downloader=_HTTPDownloader(progressbar=True),
            path=_os_cache('gplately'),
            processor=processor)
        return fnames

    archive_formats = tuple([".gz", ".xz", ".bz2"])
    if url.endswith(".zip"):
        fnames = pooch_retrieve(url, processor=_Unzip())
    elif url.endswith(archive_formats):
        fnames = pooch_retrieve(url, processor=_Decompress())
    else:
        fnames = pooch_retrieve(url, processor=None)
    return fnames


def _collect_file_extension(fnames, file_extension):
    """Searches cached directory for filenames with a specified extension(s)."""
    sorted_fnames = []
    file_extension=tuple(file_extension)
    for file in fnames:
        if file.endswith(file_extension):
            sorted_fnames.append(file)
    return sorted_fnames


def _str_in_folder(fnames, strings_to_include=None, strings_to_ignore=None):
    fnames_to_ignore = []
    fnames_to_include = []
    sorted_fnames = []
    for i, fname in enumerate(fnames):
        parent_directory = '/'.join(fname.split("/")[:-1])
        if strings_to_ignore is not None:
            for s in strings_to_ignore:
                if s in parent_directory:
                    fnames_to_ignore.append(fname)
            sorted_fnames = list(set(fnames) - set(fnames_to_ignore))

    if strings_to_include is not None:
        for fname in sorted_fnames:
            parent_directory = '/'.join(fname.split("/")[:-1])
            for s in strings_to_include:
                if s in parent_directory:
                    fnames_to_include.append(fname)
        sorted_fnames = list(set(sorted_fnames).intersection(set(fnames_to_include)))
    return sorted_fnames


def _str_in_filename(fnames, strings_to_include=None, strings_to_ignore=None):
    sorted_fnames = []
    if strings_to_ignore is not None:
        for f in fnames:
            f = f.split("/")[-1]
            check = [s for s in strings_to_ignore if s.lower() in f.lower()]
    if strings_to_include is not None:
        for s in strings_to_include:
            for f in fnames:
                fname = f.split("/")[-1]
                if s.lower() in fname.lower():
                    sorted_fnames.append(f)
    return sorted_fnames


def _check_gpml_or_shp(fnames):
    """For topology features, returns GPML by default. Searches for ESRI Shapefiles 
    instead if GPML files not found."""
    sorted_fnames = []
    for file in fnames:
        if file.endswith(".gpml"):
            sorted_fnames.append(file)
        elif file.endswith(".shp"):
            sorted_fnames.append(file)
    return sorted_fnames


def _remove_hash(fname):
    """Removes hashes (32 character file IDs) from cached filenames."""
    split_paths = fname.split("-")
    cache_path = split_paths[0][:-32]
    new_path = cache_path + "-".join(split_paths[1:])
    return new_path


def _order_filenames_by_time(fnames):
    """Orders filenames in a list from present day to deeper geological time if they
    are labelled by time."""
    # Collect all digits in each filename.
    filepath_digits=[]
    for i, file in enumerate(fnames):
        digits = []
        for element in _re.split('([0-9]+)', _remove_hash(file)):
            if element.isdigit():
                digits.append(int(str(element)))
        filepath_digits.append(digits)

    # Ignore digits common to all full file paths. This leaves behind the files' 
    # geological time label.
    geological_times = []
    filepath_digits = _np.array(filepath_digits).T
    for digit_array in filepath_digits:
        if not all(digit == digit_array[0] for digit in digit_array):
            geological_times.append(digit_array)

    # If files have geological time labels, allocate indices to the current filename order, 
    # and sort files from recent to deep geological time.
    if geological_times:
        sorted_geological_times = sorted(
            enumerate(geological_times[0]), 
            key=lambda x: x[1]
        )
        sorted_geological_time_indices = [geo_time[0] for geo_time in sorted_geological_times]
        filenames_sorted = [fnames[index] for index in sorted_geological_time_indices]
    else:
        # If given filenames do not have a time label, return them as is.
        filenames_sorted = fnames
    return filenames_sorted


def _collection_sorter(fnames, string_identifier):
    """If multiple file collections or plate reconstruction models are downloaded from
    a single zip folder, only return the needed model. 

    The plate models that need separating are listed."""

    needs_sorting = [
        "merdith2021",
        "scotese2008",
        "golonka2007",
        "clennett2020",
        "johansson2018",
        "whittaker2015"
    ]
    if string_identifier.lower() in needs_sorting:
        studyname = _re.findall(r'[A-Za-z]+|\d+', string_identifier)[0]
        newfnames = []
        for files in fnames:
            if studyname not in files:
                continue
            newfnames.append(files)
        return newfnames
    else:
        return fnames


def _match_filetype_to_extension(filetype):
    extensions = []
    if filetype == "netCDF":
        extensions.append(".nc")
    elif filetype == "jpeg":
        extensions.append(".jpg")
    elif filetype == "png":
        extensions.append(".png")
    elif filetype == "TIFF":
        extensions.append(".tif")
    return extensions


class DataServer(object):
    """Uses [Pooch](https://www.fatiando.org/pooch/latest/) to download plate reconstruction 
    feature data from plate models and other studies that are stored on web servers 
    (e.g. EarthByte's [webDAV server](https://www.earthbyte.org/webdav/ftp/Data_Collections/)). 
    
    If the `DataServer` object and its methods are called for the first time, i.e. by:

        # string identifier to access the Muller et al. 2019 model
        gDownload = gplately.download.DataServer("Muller2019")

    all requested files are downloaded into the user's 'gplately' cache folder only _once_. If the same
    object and method(s) are re-run, the files will be re-accessed from the cache provided they have not been 
    moved or deleted. 

    Currently, `DataServer` supports a number of plate reconstruction models. To call the object,
    supply a `file_collection` string from one of the following models:

    * __[Müller et al. 2019](https://www.earthbyte.org/muller-et-al-2019-deforming-plate-reconstruction-and-seafloor-age-grids-tectonics/):__ 

        file_collection = `Muller2019`
    
        Information
        -----------
        * Downloadable files: a `rotation_model`, `topology_features`, `static_polygons`, `coastlines`, `continents`, `COBs`, and
        seafloor `age_grids` from 0 to 250 Ma. 
        * Maximum reconstruction time: 250 Ma

        Citations
        ---------
        Müller, R. D., Zahirovic, S., Williams, S. E., Cannon, J., Seton, M., 
        Bower, D. J., Tetley, M. G., Heine, C., Le Breton, E., Liu, S., Russell, S. H. J., 
        Yang, T., Leonard, J., and Gurnis, M. (2019), A global plate model including 
        lithospheric deformation along major rifts and orogens since the Triassic. 
        Tectonics, vol. 38, https://doi.org/10.1029/2018TC005462.
            

    * __Müller et al. 2016__:

        file_collection = `Muller2016`

        Information
        -----------
        * Downloadable files: a `rotation_model`, `topology_features`, `static_polygons`, `coastlines`, and
        seafloor `age_grids` from 0-230 Ma. 
        * Maximum reconstruction time: 230 Ma

        Citations
        ---------
        * Müller R.D., Seton, M., Zahirovic, S., Williams, S.E., Matthews, K.J.,
        Wright, N.M., Shephard, G.E., Maloney, K.T., Barnett-Moore, N., Hosseinpour, M., 
        Bower, D.J., Cannon, J., InPress. Ocean basin evolution and global-scale plate 
        reorganization events since Pangea breakup, Annual Review of Earth and Planetary 
        Sciences, Vol 44, 107-138. DOI: 10.1146/annurev-earth-060115-012211.


    * __[Merdith et al. 2021](https://zenodo.org/record/4485738#.Yhrm8hNBzA0)__: 

        file_collection = `Merdith2021`

        Information
        -----------
        * Downloadable files: a `rotation_model`, `topology_features`, `static_polygons`, `coastlines`
        and `continents`.
        * Maximum reconstruction time: 1 Ga (however, `PlotTopologies` correctly visualises up to 410 Ma) 

        Citations: 
        ----------
        * Merdith et al. (in review), 'A continuous, kinematic full-plate motion model
        from 1 Ga to present'. 
        * Andrew Merdith. (2020). Plate model for 'Extending Full-Plate Tectonic Models 
        into Deep Time: Linking the Neoproterozoic and the Phanerozoic ' (1.1b) [Data set]. 
        Zenodo. https://doi.org/10.5281/zenodo.4485738


    * __Cao et al. 2020__: 

        file_collection = `Cao2020`

        Information
        -----------
        * Downloadable files: `rotation_model`, `topology_features`, `static_polygons`, `coastlines`
        and `continents`.
        * Maximum reconstruction time: 1 Ga

        Citations
        ---------
        * Toy Billion-year reconstructions from Cao et al (2020). 
        Coupled Evolution of Plate Tectonics and Basal Mantle Structure Tectonics, 
        doi: 10.1029/2020GC009244


    - __Mather et al. 2021__ : 

        file_collection = `Mather2021`
        
        Information
        -----------
        * Downloadable files: `rotation_model`, `topology_features`, and `coastlines`
        * Maximum reconstruction time: 170 Ma

        Citations
        ---------
        * Mather, B., Müller, R.D.,; Alfonso, C.P., Seton, M., 2021, Kimberlite eruption 
        driven by slab flux and subduction angle. DOI: 10.5281/zenodo.5769002


    - __Seton et al. 2012__:

        file_collection = `Seton2012`

        Information
        -----------
        * Downloadable files: `rotation_model`, `topology_features`, `coastlines`,
        `COBs`, and paleo-age grids (0-200 Ma)
        * Maximum reconstruction time: 200 Ma

        Citations
        ---------
        * M. Seton, R.D. Müller, S. Zahirovic, C. Gaina, T.H. Torsvik, G. Shephard, A. Talsma, 
        M. Gurnis, M. Turner, S. Maus, M. Chandler, Global continental and ocean basin reconstructions 
        since 200 Ma, Earth-Science Reviews, Volume 113, Issues 3-4, July 2012, Pages 212-270, 
        ISSN 0012-8252, 10.1016/j.earscirev.2012.03.002.


    - __Matthews et al. 2016__: 

        file_collection = `Matthews2016`

        Information
        -----------
        * Downloadable files: `rotation_model`, `topology_features`, `static_polygons`, `coastlines`,
        and `continents`
        * Maximum reconstruction time(s): 410-250 Ma, 250-0 Ma

        Citations
        ---------
        * Matthews, K.J., Maloney, K.T., Zahirovic, S., Williams, S.E., Seton, M.,
        and Müller, R.D. (2016). Global plate boundary evolution and kinematics since the 
        late Paleozoic, Global and Planetary Change, 146, 226-250. 
        DOI: 10.1016/j.gloplacha.2016.10.002


    - __Merdith et al. 2017__: 

        file_collection = `Merdith2017`

        Information
        -----------
        * Downloadable files: `rotation_files` and `topology_features`
        * Maximum reconstruction time: 410 Ma

        Citations
        ---------
        * Merdith, A., Collins, A., Williams, S., Pisarevskiy, S., Foden, J., Archibald, D. 
        and Blades, M. et al. 2016. A full-plate global reconstruction of the Neoproterozoic. 
        Gondwana Research. 50: pp. 84-134. DOI: 10.1016/j.gr.2017.04.001


    - __Li et al. 2008__: 

        file_collection = `Li2008`

        Information
        -----------
        * Downloadable files: `rotation_model` and `static_polygons`
        * Maximum reconstruction time: 410 Ma

        Citations
        ---------
        * Rodinia reconstruction from Li et al (2008), Assembly, configuration, and break-up 
        history of Rodinia: A synthesis. Precambrian Research. 160. 179-210. 
        DOI: 10.1016/j.precamres.2007.04.021.


    - __Pehrsson et al. 2015__: 

        file_collection = `Pehrsson2015`

        Information
        -----------
        * Downloadable files: `rotation_model` and `static_polygons`
        * Maximum reconstruction time: N/A

        Citations
        ---------
        * Pehrsson, S.J., Eglington, B.M., Evans, D.A.D., Huston, D., and Reddy, S.M., (2015),
        Metallogeny and its link to orogenic style during the Nuna supercontinent cycle. Geological 
        Society, London, Special Publications, 424, 83-94. DOI: https://doi.org/10.1144/SP424.5


    - __Torsvik and Cocks et al. 2017__: 

        file_collection = `TorsvikCocks2017`

        Information
        -----------
        * Downloadable files: `rotation_model`, and `coastlines`
        * Maximum reconstruction time: 410 Ma

        Citations
        ---------
        * Torsvik, T., & Cocks, L. (2016). Earth History and Palaeogeography. Cambridge: 
        Cambridge University Press. doi:10.1017/9781316225523


    - __Young et al. 2019__: 

        file_collection = `Young2019`

        Information
        -----------
        * Downloadable files: `rotation_model`, `topology_features`, `static_polygons`, `coastlines`
        and `continents`.
        * Maximum reconstruction time: 410-250 Ma, 250-0 Ma
        
        Citations
        ---------
        * Young, A., Flament, N., Maloney, K., Williams, S., Matthews, K., Zahirovic, S.,
        Müller, R.D., (2019), Global kinematics of tectonic plates and subduction zones since the late 
        Paleozoic Era, Geoscience Frontiers, Volume 10, Issue 3, pp. 989-1013, ISSN 1674-9871,
        DOI: https://doi.org/10.1016/j.gsf.2018.05.011.


    - __Scotese et al. 2008__: 

        file_collection = `Scotese2008`

        Information
        -----------
        * Downloadable files: `rotation_model`, `static_polygons`, and `continents`
        * Maximum reconstruction time: 
        
        Citations
        ---------
        * Scotese, C.R. 2008. The PALEOMAP Project PaleoAtlas for ArcGIS, Volume 2, Cretaceous 
        paleogeographic and plate tectonic reconstructions. PALEOMAP Project, Arlington, Texas.


    - __Golonka et al. 2007__: 

        file_collection = `Golonka2007`

        Information
        -----------
        * Downloadable files: `rotation_model`, `static_polygons`, and `continents`
        * Maximum reconstruction time: 410 Ma
        
        Citations
        ---------
        * Golonka, J. 2007. Late Triassic and Early Jurassic palaeogeography of the world. 
        Palaeogeography, Palaeoclimatology, Palaeoecology 244(1–4), 297–307.


    - __Clennett et al. 2020 (based on Müller et al. 2019)__: 

        file_collection = `Clennett2020_M2019`

        Information
        -----------
        * Downloadable files: `rotation_model`, `topology_features`, `continents` and `coastlines`
        * Maximum reconstruction time: 250 Ma
        
        Citations
        ---------
        * Clennett, E.J., Sigloch, K., Mihalynuk, M.G., Seton, M., Henderson, M.A., Hosseini, K.,
        Mohammadzaheri, A., Johnston, S.T., Müller, R.D., (2020), A Quantitative Tomotectonic Plate 
        Reconstruction of Western North America and the Eastern Pacific Basin. Geochemistry, Geophysics, 
        Geosystems, 21, e2020GC009117. DOI: https://doi.org/10.1029/2020GC009117


    - __Clennett et al. 2020 (rigid topological model based on Shephard et al, 2013)__: 

        file_collection = `Clennett2020_S2013`

        Information
        -----------
        * Downloadable files: `rotation_model`, `topology_features`, `continents` and `coastlines`
        * Maximum reconstruction time: 
        
        Citations
        ---------
        * Clennett, E.J., Sigloch, K., Mihalynuk, M.G., Seton, M., Henderson, M.A., Hosseini, K.,
        Mohammadzaheri, A., Johnston, S.T., Müller, R.D., (2020), A Quantitative Tomotectonic Plate 
        Reconstruction of Western North America and the Eastern Pacific Basin. Geochemistry, Geophysics, 
        Geosystems, 21, e2020GC009117. DOI: https://doi.org/10.1029/2020GC009117

    """
    def __init__(self, file_collection):

        self.file_collection = file_collection
        self.data_collection = DataCollection(self.file_collection)


    def get_plate_reconstruction_files(self):
        """Downloads and constructs a `rotation model`, a set of `topology_features` and
        and a set of `static_polygons` needed to call the `PlateReconstruction` object.

        Returns
        -------
        rotation_model : instance of <pygplates.RotationModel>
            A rotation model to query equivalent and/or relative topological plate rotations
            from a time in the past relative to another time in the past or to present day.
        topology_features : instance of <pygplates.FeatureCollection>
            Point, polyline and/or polygon feature data that are reconstructable through 
            geological time.
        static_polygons : instance of <pygplates.FeatureCollection>
            Present-day polygons whose shapes do not change through geological time. They are
            used to cookie-cut dynamic polygons into identifiable topological plates (assigned 
            an ID) according to their present-day locations.

        Notes
        -----
        This method accesses the plate reconstruction model ascribed to the `file_collection` string passed 
        into the `DataServer` object. For example, if the object was called with `"Muller2019"`:

            gDownload = gplately.download.DataServer("Muller2019")
            rotation_model, topology_features, static_polygons = gDownload.get_plate_reconstruction_files()

        the method will download a `rotation_model`, `topology_features` and `static_polygons` from the 
        Müller et al. (2019) plate reconstruction model. Once the reconstruction objects are returned, 
        they can be passed into:

            model = gplately.reconstruction.PlateReconstruction(rotation_model, topology_features, static_polygons)

        * Note: If the requested plate model does not have a certain file(s), a message will be printed 
        to alert the user. For example, using `get_plate_reconstruction_files()`
        for the Torsvik and Cocks (2017) plate reconstruction model yields the printed message:

                No topology features in TorsvikCocks2017. No FeatureCollection created - unable to 
                plot trenches, ridges and transforms.
                No continent-ocean boundaries in TorsvikCocks2017.

        """

        rotation_filenames = []
        rotation_model = []
        topology_filenames = []
        topology_features = _pygplates.FeatureCollection()
        static_polygons= _pygplates.FeatureCollection()
        static_polygon_filenames = []

        # Locate all plate reconstruction files from GPlately's DataCollection
        database = DataCollection.plate_reconstruction_files(self)

        # Set to true if we find the given collection in our database
        found_collection = False
        for collection, url in database.items():

            # Only continue if the user's chosen collection exists in our database
            if self.file_collection.lower() == collection.lower():
                found_collection = True
                if len(url) == 1:
                    fnames = _collection_sorter(
                        _fetch_from_web(url[0]), self.file_collection
                    )
                    rotation_filenames = _str_in_folder(
                        _collect_file_extension(fnames, [".rot"]),
                        strings_to_ignore=DataCollection.rotation_strings_to_ignore(self)
                    )

                    #print(rotation_filenames)
                    rotation_model = _pygplates.RotationModel(rotation_filenames)

                    topology_filenames = _collect_file_extension(
                        _str_in_folder(
                            _str_in_filename(fnames, 
                                strings_to_include=DataCollection.dynamic_polygon_strings_to_include(self)
                            ), 
                            strings_to_ignore=DataCollection.dynamic_polygon_strings_to_ignore(self)
                        ),
                        [".gpml", ".gpmlz"]
                    )
                    #print(topology_filenames)
                    for file in topology_filenames:
                        topology_features.add(_pygplates.FeatureCollection(file))

                    static_polygon_filenames = _check_gpml_or_shp(
                        _str_in_folder(
                            _str_in_filename(fnames, 
                                strings_to_include=DataCollection.static_polygon_strings_to_include(self)
                            ),
                            strings_to_ignore=DataCollection.static_polygon_strings_to_ignore(self)
                        )
                    )
                    for stat in static_polygon_filenames:
                        static_polygons.add(_pygplates.FeatureCollection(stat))
                    #print(static_polygons)
                else:
                    for file in url[0]:
                        rotation_filenames.append(
                            _collect_file_extension(
                                _fetch_from_web(file), [".rot"])
                        )
                        rotation_model = _pygplates.RotationModel(rotation_filenames)

                    for file in url[1]:
                        topology_filenames.append(
                            _collect_file_extension(
                                _fetch_from_web(file), [".gpml"])
                        )
                        for file in topology_filenames:
                            topology_features.add(
                                _pygplates.FeatureCollection(file)
                            )

                    for file in url[2]:
                        static_polygon_filenames.append(
                            _check_gpml_or_shp(
                                _str_in_folder(
                                    _str_in_filename(_fetch_from_web(url[0]), 
                                        strings_to_include=DataCollection.static_polygon_strings_to_include(self)
                                    ),    
                                        strings_to_ignore=DataCollection.static_polygon_strings_to_ignore(self)
                                )
                            )   
                        )
                        for stat in static_polygon_filenames:
                            static_polygons.add(_pygplates.FeatureCollection(stat))
                break

        if found_collection is False:
            raise ValueError("{} is not in GPlately's DataServer.".format(self.file_collection))

        if not rotation_filenames:
            print("No .rot files in {}. No rotation model created.".format(self.file_collection))
            rotation_model = []
        if not topology_filenames:
            print("No topology features in {}. No FeatureCollection created - unable to plot trenches, ridges and transforms.".format(self.file_collection))
            topology_features = []
        if not static_polygons:
            print("No static polygons in {}.".format(self.file_collection))
            static_polygons = []

        return rotation_model, topology_features, static_polygons


    def get_topology_geometries(self):
        """Uses Pooch to download coastline, continent and COB (continent-ocean boundary)
        Shapely geometries from the requested plate model. These are needed to call the `PlotTopologies`
        object and visualise topological plates through time.

        Returns
        -------
        coastlines : instance of <pygplates.FeatureCollection>
            Present-day global coastline Shapely polylines cookie-cut using static polygons. Ready for
            reconstruction to a particular geological time and for plotting.

        continents : instance of <pygplates.FeatureCollection>
            Cookie-cutting Shapely polygons for non-oceanic regions (continents, inta-oceanic arcs, etc.)
            ready for reconstruction to a particular geological time and for plotting.

        COBs : instance of <pygplates.FeatureCollection>
            Shapely polylines resolved from .shp and/or .gpml topology files that represent the 
            locations of the boundaries between oceanic and continental crust.
            Ready for reconstruction to a particular geological time and for plotting.

        Notes
        -----
        This method accesses the plate reconstruction model ascribed to the `file_collection` 
        string passed into the `DataServer` object. For example, if the object was called with
        `"Muller2019"`:

            gDownload = gplately.download.DataServer("Muller2019")
            coastlines, continents, COBs = gDownload.get_topology_geometries()

        the method will attempt to download `coastlines`, `continents` and `COBs` from the Müller
        et al. (2019) plate reconstruction model. If found, these files are returned as individual 
        pyGPlates Feature Collections. They can be passed into:

            gPlot = gplately.plot.PlotTopologies(gplately.reconstruction.PlateReconstruction, time, continents, coastlines, COBs)

        to reconstruct features to a certain geological time. The `PlotTopologies`
        object provides simple methods to plot these geometries along with trenches, ridges and 
        transforms (see documentation for more info). Note that the `PlateReconstruction` object 
        is a parameter.

        * Note: If the requested plate model does not have a certain geometry, a
        message will be printed to alert the user. For example, if `get_topology_geometries()` 
        is used with the `"Matthews2016"` plate model, the workflow will print the following 
        message: 

                No continent-ocean boundaries in Matthews2016.
        """

        # Locate all topology geometries from GPlately's DataCollection
        database = DataCollection.topology_geometries(self)

        coastlines = []
        continents = []
        COBs = []
        
        # Find the requested plate model data collection
        found_collection = False
        for collection, url in database.items():

            if self.file_collection.lower() == collection.lower():
                found_collection = True

                if len(url) == 1:
                    # Some plate models do not have reconstructable geometries i.e. Li et al. 2008
                    if url[0] is None:
                        break
                    else:
                        fnames = _collection_sorter(
                            _fetch_from_web(url[0]), self.file_collection
                        )
                        coastlines = _check_gpml_or_shp(
                            _str_in_folder(
                                _str_in_filename(
                                    fnames,
                                    strings_to_include=DataCollection.coastline_strings_to_include(self)
                                ), 
                                strings_to_ignore=DataCollection.coastline_strings_to_ignore(self)
                            )
                        )
                        continents = _check_gpml_or_shp(
                            _str_in_folder(
                                _str_in_filename(
                                    fnames, 
                                    strings_to_include=DataCollection.continent_strings_to_include(self)
                                ), 
                                strings_to_ignore=DataCollection.continent_strings_to_ignore(self)
                            )
                        )
                        COBs = _check_gpml_or_shp(
                            _str_in_folder(
                                _str_in_filename(
                                    fnames,
                                    strings_to_include=DataCollection.COB_strings_to_include(self)
                                ), 
                                strings_to_ignore=DataCollection.COB_strings_to_ignore(self)
                            )
                        )
                else:
                    for file in url[0]:
                        if url[0] is not None:
                            coastlines.append(_str_in_filename(
                                _fetch_from_web(file), 
                                strings_to_include=["coastline"])
                            )
                            coastlines = _check_gpml_or_shp(coastlines)
                        else:
                            coastlines = []

                    for file in url[1]:
                        if url[1] is not None:
                            continents.append(_str_in_filename(
                                _fetch_from_web(file), 
                                strings_to_include=["continent"])
                            )
                            continents = _check_gpml_or_shp(continents)
                        else:
                            continents = []

                    for file in url[2]:
                        if url[2] is not None:
                            COBs.append(_str_in_filename(
                                _fetch_from_web(file), 
                                strings_to_include=["cob"])
                            )
                            COBs = _check_gpml_or_shp(COBs)
                        else:
                            COBs = []
                break

        if found_collection is False:
            raise ValueError("{} is not in GPlately's DataServer.".format(self.file_collection))

        if not coastlines:
            print("No coastlines in {}.".format(self.file_collection))
            coastlines_featurecollection = []
        else:
            #print(coastlines)
            coastlines_featurecollection = _pygplates.FeatureCollection()
            for coastline in coastlines:
                coastlines_featurecollection.add(_pygplates.FeatureCollection(coastline))
        
        if not continents:
            print("No continents in {}.".format(self.file_collection))
            continents_featurecollection = []
        else:
            #print(continents)
            continents_featurecollection = _pygplates.FeatureCollection()
            for continent in continents:
                continents_featurecollection.add(_pygplates.FeatureCollection(continent))
        
        if not COBs:
            print("No continent-ocean boundaries in {}.".format(self.file_collection))
            COBs_featurecollection = []
        else:
            #print(COBs)
            COBs_featurecollection = _pygplates.FeatureCollection()
            for COB in COBs:
                COBs_featurecollection.add(_pygplates.FeatureCollection(COB))
        
        geometries = coastlines_featurecollection, continents_featurecollection, COBs_featurecollection
        return geometries


    def get_age_grid(self, time):
        """Downloads seafloor and paleo-age grids from the plate reconstruction model (`file_collection`)
        passed into the `DataServer` object. Stores grids in the "gplately" cache.

        Currently, `DataServer` supports the following age grids:

        * __Muller et al. 2019__

            * `file_collection` = `Muller2019`
            * Time range: 0-250 Ma
            * Seafloor age grid rasters in netCDF format.

        * __Muller et al. 2016__
            
            * `file_collection` = `Muller2016`
            * Time range: 0-240 Ma
            * Seafloor age grid rasters in netCDF format. 

        * __Seton et al. 2012__

            * `file_collection` = `Seton2012`
            * Time range: 0-200 Ma
            * Paleo-age grid rasters in netCDF format.

        
        Parameters
        ----------
        time : int, or list of int, default=None
            Request an age grid from one (an integer) or multiple reconstruction times (a
            list of integers).

        Returns
        -------
        raster_array : MaskedArray
            A masked array containing the netCDF4 age grid ready for plotting or for
            passing into GPlately's `Raster` object for raster manipulation.

        Raises
        -----
        ValueError
            If `time` (a single integer, or a list of integers representing reconstruction
            times to extract the age grids from) is not passed.

        Notes
        -----
        The first time that `get_age_grid` is called for a specific time(s), the age grid(s) 
        will be downloaded into the GPlately cache once. Upon successive calls of `get_age_grid`
        for the same reconstruction time(s), the age grids will not be re-downloaded; rather, 
        they are re-accessed from the same cache provided the age grid(s) have not been moved or deleted. 

        Examples
        --------
        if the `DataServer` object was called with the `Muller2019` `file_collection` string:

            gDownload = gplately.download.DataServer("Muller2019")

        `get_age_grid` will download seafloor age grids from the Müller et al. (2019) plate 
        reconstruction model for the geological time(s) requested in the `time` parameter. 
        If found, these age grids are returned as masked arrays. 

        For example, to download  Müller et al. (2019) seafloor age grids for 0Ma, 1Ma and
        100 Ma:

            age_grids = gDownload.get_age_grid([0, 1, 100])
            
        """
        age_grids = []
        age_grid_links = DataCollection.netcdf4_age_grids(self, time)
        for link in age_grid_links:
            age_grid_file = _fetch_from_web(link)
            age_grid = _gplately.grids.read_netcdf_grid(age_grid_file)
            age_grids.append(age_grid)

        if not age_grids:
            raise ValueError("{} netCDF4 age grids not found.".format())

        if len(age_grids) == 1:
            return age_grids[0]
        else: 
            return age_grids


    def get_raster(self, raster_id_string=None):
        """Downloads assorted raster data that are not associated with the plate 
        reconstruction models supported by GPlately's `DataServer`. Stores rasters in the 
        "gplately" cache.

        Currently, `DataServer` supports the following rasters and images:

        * __[ETOPO1](https://www.ngdc.noaa.gov/mgg/global/)__: 
            * Filetypes available : TIF, netCDF (GRD)
            * `raster_id_string` = `"ETOPO1_grd"`, `"ETOPO1_tif"` (depending on the requested format)
            * A 1-arc minute global relief model combining lang topography and ocean bathymetry.
            * Citation: doi:10.7289/V5C8276M


        Parameters
        ----------
        raster_id_string : str, default=None
            A string to identify which raster to download.

        Returns
        -------
        raster_filenames : ndarray
            An ndarray of the cached raster. This can be plotted using `matplotlib.pyplot.imshow` on
            a `cartopy.mpl.GeoAxis` GeoAxesSubplot (see example below).

        Raises
        ------
        ValueError
            * if a `raster_id_string` is not supplied.

        Notes
        -----
        Rasters obtained by this method are (so far) only reconstructed to present-day. 

        Examples
        --------
        To download ETOPO1 and plot it on a Mollweide projection:

            import gplately
            import numpy as np
            import matplotlib.pyplot as plt
            import cartopy.crs as ccrs

            gdownload = gplately.DataServer("Muller2019")
            etopo1 = gdownload.get_raster("ETOPO1_tif")
            fig = plt.figure(figsize=(18,14), dpi=300)
            ax = fig.add_subplot(111, projection=ccrs.Mollweide(central_longitude = -150))
            ax2.imshow(etopo1, extent=[-180,180,-90,90], transform=ccrs.PlateCarree()) 

        """
        from matplotlib import image
        if raster_id_string is None:
            raise ValueError(
                "Please specify which raster to download."
            )
        #filetype = "."+"_".split(raster_id_string)[-1]

        archive_formats = tuple([".gz", ".xz", ".bz2"])
        # Set to true if we find the given collection in database
        found_collection = False
        raster_filenames = []
        database = DataCollection.rasters(self)

        for collection, zip_url in database.items():
            # Isolate the raster name and the file type
            #raster_name = collection.split("_")[0]
            #raster_type = "."+collection.split("_")[-1]
            if (raster_id_string.lower() == collection.lower()):
                raster_filenames = _fetch_from_web(zip_url[0])
                found_collection = True
                break

        if found_collection is False:
            raise ValueError("{} not in collection database.".format(raster_id_string))
        else:
            raster_matrix = image.imread(raster_filenames)
        return raster_matrix


    def get_feature_data(self, feature_data_id_string=None):
        """Downloads assorted geological feature data from web servers (i.e. 
        [GPlates 2.3 sample data](https://www.earthbyte.org/gplates-2-3-software-and-data-sets/))
        into the "gplately" cache.

        Currently, `DataServer` supports the following feature data:

        * __Large igneous provinces from Johansson et al. (2018)__

            Information
            -----------
            * Formats: .gpmlz
            * `feature_data_id_string` = `Johansson2018`

            Citations
            ---------
            * Johansson, L., Zahirovic, S., and Müller, R. D., In Prep, The 
            interplay between the eruption and weathering of Large Igneous Provinces and 
            the deep-time carbon cycle: Geophysical Research Letters.


        - __Large igneous province products interpreted as plume products from Whittaker 
        et al. (2015)__.

            Information
            -----------
            * Formats: .gpmlz, .shp
            * `feature_data_id_string` = `Whittaker2015`
            
            Citations
            ---------
            * Whittaker, J. M., Afonso, J. C., Masterton, S., Müller, R. D., 
            Wessel, P., Williams, S. E., & Seton, M. (2015). Long-term interaction between 
            mid-ocean ridges and mantle plumes. Nature Geoscience, 8(6), 479-483. 
            doi:10.1038/ngeo2437.


        - __Seafloor tectonic fabric (fracture zones, discordant zones, V-shaped structures, 
        unclassified V-anomalies, propagating ridge lineations and extinct ridges) from 
        Matthews et al. (2011)__

            Information
            -----------
            * Formats: .gpml
            * `feature_data_id_string` = `SeafloorFabric`

            Citations
            ---------
            * Matthews, K.J., Müller, R.D., Wessel, P. and Whittaker, J.M., 2011. The 
            tectonic fabric of the ocean basins. Journal of Geophysical Research, 116(B12): 
            B12109, DOI: 10.1029/2011JB008413. 


        - __Present day surface hotspot/plume locations from Whittaker et al. (2013)__

            Information
            -----------
            * Formats: .gpmlz
            * `feature_data_id_string` = `Hotspots`

            Citation
            --------
            * Whittaker, J., Afonso, J., Masterton, S., Müller, R., Wessel, P., 
            Williams, S., and Seton, M., 2015, Long-term interaction between mid-ocean ridges and 
            mantle plumes: Nature Geoscience, v. 8, no. 6, p. 479-483, doi:10.1038/ngeo2437.

        
        Parameters
        ----------
        feature_data_id_string : str, default=None
            A string to identify which feature data to download to the cache (see list of supported
            feature data above).

        Returns
        -------
        feature_data_filenames : instance of <pygplates.FeatureCollection>, or list of instance <pygplates.FeatureCollection>
            If a single set of feature data is downloaded, a single pyGPlates `FeatureCollection` 
            object is returned. Otherwise, a list containing multiple pyGPlates `FeatureCollection` 
            objects is returned (like for `SeafloorFabric`). In the latter case, feature reconstruction 
            and plotting may have to be done iteratively.

        Raises
        ------
        ValueError
            If a `feature_data_id_string` is not provided.

        Examples
        --------
        For examples of plotting data downloaded with `get_feature_data`, see GPlately's sample 
        notebook 05 - Working With Feature Geometries [here](https://github.com/GPlates/gplately/blob/master/Notebooks/05-WorkingWithFeatureGeometries.ipynb).
        """
        if feature_data_id_string is None:
            raise ValueError(
                "Please specify which feature data to fetch."
            )

        database = DataCollection.feature_data(self)

        found_collection = False
        for collection, zip_url in database.items():
            if feature_data_id_string.lower() == collection.lower():
                found_collection = True
                feature_data_filenames = _collection_sorter(
                    _collect_file_extension(
                    _fetch_from_web(zip_url[0]), [".gpml", ".gpmlz"]
                    ),
                    collection
                )

                break

        feat_data = _pygplates.FeatureCollection()
        if len(feature_data_filenames) == 1:
                feat_data.add(_pygplates.FeatureCollection(feature_data_filenames[0]))
                return feat_data
        else:    
            feat_data=[]
            for file in feature_data_filenames:
                feat_data.append(_pygplates.FeatureCollection(file))
            return feat_data
    