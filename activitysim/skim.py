# ActivitySim
# See full license in LICENSE.txt.

import logging

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


class Skim(object):
    """
    Container for skim arrays.

    Parameters
    ----------
    data : 2D array
    offset : int, optional
        An optional offset that will be added to origin/destination
        values to turn them into array indices.
        For example, if zone IDs are 1-based, an offset of -1
        would turn them into 0-based array indices.

    """
    def __init__(self, data, offset=None):
        self.data = np.asanyarray(data)
        self.offset = offset

    def get(self, orig, dest):
        """
        Get impedence values for a set of origin, destination pairs.

        Parameters
        ----------
        orig : 1D array
        dest : 1D array

        Returns
        -------
        values : 1D array

        """
        # only working with numpy in here
        orig = np.asanyarray(orig)
        dest = np.asanyarray(dest)
        out_shape = orig.shape

        # filter orig and dest to only the real-number pairs
        notnan = ~(np.isnan(orig) | np.isnan(dest))
        orig = orig[notnan].astype('int')
        dest = dest[notnan].astype('int')

        if self.offset:
            orig = orig + self.offset
            dest = dest + self.offset

        result = self.data[orig, dest]

        # add the nans back to the result
        out = np.empty(out_shape)
        out[notnan] = result
        out[~notnan] = np.nan

        return out


class Skims(object):
    """
    A skims object is a wrapper around multiple skim objects,
    where each object is identified by a key.  It operates like a
    dictionary - i.e. use brackets to add and get skim objects - but also
    has information on how to lookup against the skim objects.
    Specifically, this object has a dataframe, a left_key and right_key.
    It is assumed that left_key and right_key identify columns in df.  The
    parameter df is usually set by the simulation itself as it's a result of
    interacting choosers and alternatives.

    When the user calls skims[key], key is an identifier for which skim
    to use, and the object automatically looks up impedances of that skim
    using the specified left_key column in df as the origin and
    the right_key column in df as the destination.  In this way, the user
    does not do the O-D lookup by hand and only specifies which skim to use
    for this lookup.  This is the only purpose of this object: to
    abstract away the O-D lookup and use skims by specifying which skim
    to use in the expressions.

    Note that keys are any hashable object, not just strings.  So calling
    skim[('AM', 'SOV')] is valid and useful.
    """

    def __init__(self):
        self.skims = {}
        self.left_key = "TAZ"
        self.right_key = "TAZ_r"
        self.df = None

    def set_keys(self, left_key, right_key):
        """
        Set the left and right keys.

        Parameters
        ----------
        left_key : String
            The left key (origin) column in the dataframe
        right_key : String
            The right key (destination) column in the dataframe

        Returns
        --------
        Nothing
        """
        self.left_key = left_key
        self.right_key = right_key
        return self

    def set_df(self, df):
        """
        Set the dataframe

        Parameters
        ----------
        df : DataFrame
            The dataframe which contains the origin and destination ids

        Returns
        -------
        Nothing
        """
        self.df = df

    def lookup(self, skim):
        """
        Generally not called by the user - use __getitem__ instead

        Parameters
        ----------
        skim: Skim
            The skim object to perform the lookup using df[left_key] as the
            origin and df[right_key] as the destination

        Returns
        -------
        impedances: pd.Series
            A Series of impedances which are elements of the Skim object and
            with the same index as df
        """
        assert self.df is not None, "Call set_df first"
        s = skim.get(self.df[self.left_key],
                     self.df[self.right_key])
        return pd.Series(s, index=self.df.index)

    def set_3d(self, key, key_3d, value):
        """
        If you want to use the Skims3D object below, you will need to do that
        explicitly by setting first the key which will be used by __getattr__
        and second the key that relates to the 3rd dimension of the dataframe.

        Parameters
        ----------
        key : String or any hashable
            Will be accessible using __getitem__ in Skims3d
        key_3d : String or any hashable
            Relates to the 3rd dimension lookup column set by Skims3D
        value : Skim
            the skim object for these keys
        """
        self.skims[(key, key_3d)] = value

    def get_3d(self, key, key_3d):
        """
        If you want

        Parameters
        ----------
        key : String or any hashable
            Will be accessible using __getitem__ in Skims3d
        key_3d : String or any hashable
            Relates to the 3rd dimension lookup column set by Skims3D

        Returns
        -------
        skims : Skim
            the skim object for these keys
        """
        return self.skims[(key, key_3d)]

    def __setitem__(self, key, value):
        """
        Set an available skim object

        Parameters
        ----------
        key : hashable
             The key (identifier) for this skim object
        value : Skim
             The skim object

        Returns
        -------
        Nothing
        """
        self.skims[key] = value

    def __getitem__(self, key):
        """
        Get the (df implicit) lookup for an available skim object

        Parameters
        ----------
        key : hashable
             The key (identifier) for this skim object

        Returns
        -------
        impedances: pd.Series
            A Series of impedances which are elements of the Skim object and
            with the same index as df
        """
        # FIXME - misleading and confusing that this getter is NOT symmetrical to __setitem__
        # get_skim below is the symmetrical getter corresponding to __setitem__
        return self.lookup(self.skims[key])

    def get_skim(self, key):
        """
        Get an available skim object (not the lookup)

        Parameters
        ----------
        key : hashable
             The key (identifier) for this skim object

        Returns
        -------
        skim: Skim
             The skim object
        """
        # FIXME - misleading that this (and NOT __getitem__) is symmetrical to __setitem__ setter
        return self.skims[key]


class SkimStack(object):

    def __init__(self, skims):

        self.skims_data = {}
        self.skim_keys_to_indexes = {}

        # pass to make dictionary of dictionaries where highest level is unique
        # first items of the tuples and the 2nd level is the second items of
        # the tuples
        for key, value in skims.skims.iteritems():
            if not isinstance(key, tuple) or not len(key) == 2:
                logger.debug("Skims3D __init__ skipping key: %s" % key)
                continue
            skim_key1, skim_key2 = key
            # logger.debug("SkimStack init key: key1='%s' key2='%s'" % (skim_key1, skim_key2))
            # FIXME - this is just an object assignment not an actual copy?
            self.skims_data.setdefault(skim_key1, {})[skim_key2] = value.data

        # second pass to turn the each highest level value into a 3D array
        # with a dictionary to make second level keys to indexes
        for skim_key1, value in self.skims_data.iteritems():
            # FIXME - this actually creates new stacked data
            self.skims_data[skim_key1] = np.dstack(value.values())
            self.skim_keys_to_indexes[skim_key1] = dict(zip(value.keys(), range(len(value))))

        logger.info("SkimStack.__init__ loaded and stacked %s skims for %s keys"
                    % (len(skims.skims.keys()), len(self.skim_keys_to_indexes.keys())))

    def key_count(self):
        return len(self.skim_keys_to_indexes.keys())

    def contains(self, key):
        return key in self.skims_data

    def get(self, key):
        return self.skims_data[key], self.skim_keys_to_indexes[key]


class Skims3D(object):
    """
    A Skims3D object wraps a skim objects to add an additional wrinkle of
    lookup functionality.  Upon init the separate skims objects are
    processed into a 3D matrix so that lookup of the different skims can
    be performed quickly for each row in the dataframe.  In this very
    particular formulation, the keys are assumed to be tuples with two
    elements - the second element of which will be taken from the
    different rows in the dataframe.  The first element can then be
    dereferenced like an array.  This is useful, for instance, to have a
    certain skim vary by time of day - the skims are set with keys of
    ('SOV', 'AM"), ('SOV', 'PM') etc.  The time of day is then taken to
    be different for every row in the tours table, and the 'SOV' portion
    of the key can be used in __getitem__.

    To be more explicit, the input is a dictionary of Skims objects, each of
    which contains a 2D matrix.  These are stacked into a 3D matrix with a
    mapping of keys to indexes which is applied using pandas .map to a third
    column in the object dataframe.  The three columns - left_key and
    right_key from the Skims object and skim_key from this one, are then used to
    dereference the 3D matrix.  The tricky part comes in defining the key which
    matches the 3rd dimension of the matrix, and the key which is passed into
    __getitem__ below (i.e. the one used in the specs).  By convention,
    every key in the Skims object that is passed in MUST be a tuple with 2
    items.  The second item in the tuple maps to the items in the dataframe
    referred to by the skim_key column and the first item in the tuple is
    then available to pass directly to __getitem__.  This is now made
    explicit by adding the set_3d and get_3d methods in the Skims object which
    take the two keys independently and convert to the tuple internally.
    The sum conclusion of this is that in the specs, you can say something
    like out_skim['SOV'] and it will automatically dereference the 3D matrix
    using origin, destination, and time of day.

    Parameters
    ----------
    skims: Skims
        This is the Skims object to wrap
    skim_key : str
        This identifies the column in the dataframe which is used to
        select among Skim object using the SECOND item in each tuple (see
        above for a more complete description)
    offset : int, optional
        A single offset must be used for all Skim objects - previous
        offsets will be ignored
    """

    def __init__(self, left_key, right_key, skim_key, offset=None, stack=None):
        self.left_key = left_key
        self.right_key = right_key
        self.offset = offset
        self.skim_key = skim_key
        self.df = None
        self.stack = None

        # lazy load support - enabled via call to set_omx
        self.lazy_load = False
        self.omx = None

        if stack is not None:
            logger.info("Skims3D.__init__ loading %s skims from stack." % stack.key_count())
            self.stack = stack

    def set_df(self, df):
        """
        Set the dataframe

        Parameters
        ----------
        df : DataFrame
            The dataframe which contains the origin and destination ids

        Returns
        -------
        Nothing
        """
        self.df = df

    def __getitem__(self, key):
        """
        Get an available skim object

        Parameters
        ----------
        key : hashable
             The key (identifier) for this skim object

        Returns
        -------
        skim: Skim
             The skim object
        """

        assert self.df is not None, "Call set_df first"
        origins = self.df[self.left_key].astype('int')
        destinations = self.df[self.right_key].astype('int')
        if self.offset:
            origins = origins + self.offset
            destinations = destinations + self.offset

        if self.stack.contains(key):
            stacked_skim_data, skim_keys_to_indexes = self.stack.get(key)
        elif self.lazy_load:
            stacked_skim_data, skim_keys_to_indexes = self._load_stacked_skim_from_disk(key)
        assert stacked_skim_data is not None, "Skims3D key %s missing" % key

        skim_indexes = self.df[self.skim_key].map(skim_keys_to_indexes).astype('int')

        ret = pd.Series(stacked_skim_data[origins, destinations, skim_indexes], self.df.index)

        if not self.stack.contains(key):
            # FIXME - is there any point to doing this?
            del stacked_skim_data
            del skim_keys_to_indexes

        return ret

    """
    So these three function allow the use of reading skims directly from the OMX
    file - ON DISK - rather than storing all your skims in memory.  This
    comes about well, first, because I run out of memory on my machine and on
    Travis when reading all the skims into memory, and second, that with the
    exception of the distance matrix, we really only use each skim 1-2 times
    each and pretty much all in the mode choice model.  And even though each
    skim for 1454 zone system is only about 16MB, we have about 300 skim files
    which can get large pretty fast (although I think it should be manageable
    even still.  So the job here is to build the 3D skims file, stacking the
    skims for different time periods into a single 3D matrix (origin,
    destination, and time period).  Unfortunately this doesn't run as fast as I
    thought it might - I actually think the stacking is pretty slow especially
    Anyway, this should be considered a "low memory" mode.  It is not right now
    working very well (I mean it works, just very slowly).
    """

    def get_from_omx(self, key, v):
        # treat this as a callback - override depending on how you store skims in the omx file
        #
        # from activitysim import skim as askim
        # from types import MethodType
        # askim.Skims3D.get_from_omx = MethodType(get_from_omx, None, askim.Skims3D)

        omx_key = key + '__' + v
        # print "my_get_from_omx - key: '%s' v: '%s', omx_key: '%s'" % (key, v, omx_key)
        return self.omx[omx_key]

    def _load_stacked_skim_from_disk(self, key):

        # get list of unique second-tuple-item keys (aka skim_key2)
        uniq = self.df[self.skim_key].unique()

        # print "_load_stacked_skim_from_disk key = '%s' key2 = %s " % (key, uniq)

        skims_data = np.dstack([self.get_from_omx(key, v) for v in uniq])
        skim_keys_to_indexes = {i: v for i, v in zip(uniq, range(len(uniq)))}

        return skims_data, skim_keys_to_indexes

    def set_omx(self, omx):

        self.lazy_load = omx is not None
        self.omx = omx
