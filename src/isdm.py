from math import pi, atan
import warnings
import math
import sys
import shutil
import pickle
import os
import numpy as np
import pylru as lru

try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError

try:
    FileExistsError
except NameError:
    FileExistsError = IOError

TWO_PI = 44.0/7

def configure(n=1024, r_range=(0,15), contains_std_threshold=5, loc_name='pam'):
    global r, ndim, exp_dist, exp_dist_std, memory_location, _axis_length
    global memory_location, r_min, r_max, delta_theta
    global contains_threshold, same_vector_distance_threshold

    r = r_range
    ndim = n
    exp_dist = ndim*(r[1]-r[0]+1)/4
    exp_dist_std = (ndim*((((r[1]-r[0]+1)**2)+8)/48))**0.5
    memory_location = loc_name


    _axis_length = r[1]-r[0]+1
    r_max = r[1]
    r_min = r[0]
    delta_theta = TWO_PI / _axis_length
    same_vector_distance_threshold = ndim/10
    contains_threshold = contains_std_threshold

configure(1024)

def set_modularity(new_r):
    global r, _axis_length, r_max, r_min
    r = new_r
    _axis_length = r[1]-r[0]+1
    r_max = r[1]
    r_min = r[0]


def set_dimensionality(new_ndim):
    global ndim, same_vector_distance_threshold
    ndim = new_ndim
    same_vector_distance_threshold = ndim/10


sin = dict([(key, np.sin(key * delta_theta)) for key in range(_axis_length)])
cos = dict([(key, np.cos(key * delta_theta)) for key in range(_axis_length)])


class ModularDimension(object):


    def __init__(self, val):
        self.mag = 1
        self.theta = val * delta_theta
        self.value = val

    @classmethod
    def from_polar(cls, mag, theta):
        value = round(theta/delta_theta) % _axis_length
        obj = cls(value)
        obj.mag = mag
        obj.theta = theta
        return obj

    @property
    def theta(self):
        return self.__theta

    @theta.setter
    def theta(self, theta_value):
        self.__theta = theta_value
        self.value = round(theta_value/delta_theta) % _axis_length

    @property
    def value(self):
        return self.__value

    @value.setter
    def value(self, v):
        self.__value = v

    def set_vector(self, val):
        self.mag = 1
        self.theta = val * delta_theta

    def set_vector(self, val, mag):
        self.mag = mag
        self.theta = val * delta_theta

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self+other

    def __add__(self, v):
        resultant = ModularDimension(0)
        y = v.mag * math.sin(v.theta) + self.mag * math.sin(self.theta)
        x = v.mag * math.cos(v.theta) + self.mag * math.cos(self.theta)
        if x == 0:
            if y > 0:
                theta = (TWO_PI/4)
            else:
                theta = (TWO_PI*3/4)
        else:
            theta = atan(abs(y/x))

        if y < 0 < x:
            theta = TWO_PI-theta
        elif y < 0 and x < 0:
            theta = TWO_PI/2+theta
        elif y > 0 > x:
            theta = TWO_PI/2-theta

        if x == 0 and y == 0:
            # insert the chance logic
            theta = self.theta + TWO_PI/4

        resultant.mag = (y ** 2 + x ** 2) ** 0.5
        resultant.theta = theta
        return resultant
#
#
# def add_two_dimensions(value1, magnitude1, value2, magnitude2):
#     # x and y components are flipped for visualizing the vectors on a clock style circle with 0 at the top
#     x = magnitude1*cos[value1] + magnitude2*cos[value2]
#     y = magnitude1*sin[value1] + magnitude2*sin[value2]
#
#     if x == 0:
#         if y > 0:
#             result = (TWO_PI/4)
#         else:
#             result = (TWO_PI*(3.0/4))
#     else:
#         result = atan(abs(y/x))
#
#     if y < 0 < x:
#         result = TWO_PI - result
#     elif y < 0 and x < 0:
#         result = TWO_PI / 2 + result
#     elif x < 0 < y:
#         result = TWO_PI / 2 - result
#
#     if x == 0 and y == 0:
#         # insert the chance logic
#         result = atan(sin[value1]/cos[value1]) + TWO_PI / 4
#
#     magnitude = (value1**2+value2**2)**.5
#
#     return round(result / delta_theta) % _axis_length, magnitude

# Check add operation
# for a, b in itertools.combinations(range(16), 2):
#    print(a, b, add_two_dimensions(a, 1, b, 1)[0])


class MCRVector(object):

    __slots__ = ['_dims', '_factor', '_magnitudes_internal', '_group_list']

    def __init__(self, dims, factor=1., _mag=np.array([1]*ndim), group_list=[]):
        if len(dims) != ndim:
            raise ValueError('Invalid length of vector')

        if not isinstance(dims, np.ndarray):
            self._dims = np.array(dims)
        else:
            self._dims = dims

        self._factor = factor
        self._magnitudes_internal = _mag
        self._group_list = group_list

    @property
    def _magnitudes(self):
        magnitudes = self._magnitudes_internal
        self._magnitudes_internal = [1] * ndim
        return magnitudes

    @classmethod
    def random_vector(cls, factor=1., _mag=np.array([1]*ndim)):
        _dims = np.random.random_integers(r_min, r_max, ndim)
        return cls(_dims, factor=factor, _mag=_mag)

    def __str__(self):
        return str(self._dims)

    def __len__(self):
        return len(self._dims)

    def __invert__(self):
        dims_copy = [(_axis_length-dim) % _axis_length for dim in self.dims]
        return MCRVector(np.array(dims_copy))

    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return MCRVector(self.dims.copy(), factor=self.factor*other)
        else:
            return MCRVector(np.apply_along_axis(lambda dim: dim % _axis_length, 0, self.dims+other.dims))

    @staticmethod
    def _addMCR(mcrs):
        mcrR = list()
        for i in range(np.size(mcrs[0])):
            mcrR.append(0)
            values = [ModularDimension(mcr[i]) for mcr in mcrs]
            mcrR[i] = (sum(values)).value % _axis_length
        result = MCRVector(mcrR, group_list=mcrs)
        return result

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self+other

    def __add__(self, other):
        # Optimization to prevent a lot of multiplications because _factor is mostly 1
        #This solution may make add to work but MCR can only loose its group after binding/mul
        new_group = []
        if len(self._group_list) > 0:
            new_group.extend(self._group_list)
        else:
            new_group.append(self)
        if len(other._group_list) > 0:
            new_group.extend(other._group_list)
        else:
            new_group.append(other)
        return MCRVector._addMCR(new_group)

    def __mod__(self, other):
        return MCRVector.distance_between(self, other)

    def __getitem__(self, item):
        return self._dims[item]

    def distance(self, other):
        return MCRVector.distance_between(self, other)

    def probe(self, other):
        if len(self._group_list) > 20 or len(other._group_list) > 20:
            warnings.warn('Only tested for maximum goup size of 21')

        return (self % other) < (exp_dist-contains_threshold*exp_dist_std)

    def __contains__(self, item):
        return self.probe(item)


    @staticmethod
    def distance_between(x, y):
        diff1 = (x.dims-y.dims) % _axis_length
        diff2 = (y.dims-x.dims) % _axis_length
        mins = np.minimum(diff1, diff2)
        manhattan_distance = np.sum(mins)
        return manhattan_distance

    @property
    def dims(self):
        return self._dims

    @property
    def factor(self):
        return self._factor

    def get_noisy_copy(self, pct=.1):
        chosen_dims = np.random.choice(range(ndim), size=int(ndim*pct))
        new_dims = self.dims.copy()
        for i in chosen_dims:
            new_dims[i] = np.random.random_integers(r_min, r_max)
        return MCRVector(new_dims, factor=self._factor)


class HardLocation(MCRVector):

    __slots__ = ['_name', '_dims', '_factor', '_magnitudes_internal']

    def __init__(self, name):
        _vector = MCRVector.random_vector().dims.copy()
        super(HardLocation, self).__init__(_vector)
        self._name = name

    @staticmethod
    def get_empty_counter():
        return [np.array([0]*_axis_length) for _ in range(ndim)]

    def create_counters(self):
        counters = HardLocation.get_empty_counter()
        IntegerSDM.store_counters(self.name, counters)
        return counters

    def write(self, word):
        counters = self.counters
        for i, dim in enumerate(word.dims):
            counters[i][dim] += 1
        self._update_counters(counters)

    def add_counters(self, counters):
        resultant_counters = []
        self_counters = self.counters
        for i, counter_dim in enumerate(counters):
            resultant_counters.append(counters[i]+self_counters[i])

        return resultant_counters

    def _update_counters(self, counters):
        IntegerSDM.store_counters(self.name, counters)

    @property
    def counters(self):
        counters = IntegerSDM.retrieve_counters(self.name)
        if counters is None:
            counters = self.create_counters()
        return counters

    @property
    def name(self):
        return self._name

    def __hash__(self):
        return hash(self.name)


class IntegerSDM(object):

    cache = lru.lrucache(1000)
    cached = True

    @staticmethod
    def retrieve_counters(hard_location_name):
        if IntegerSDM.cached and hard_location_name in IntegerSDM.cache:
            return IntegerSDM.cache[hard_location_name]
        try:
            mem_file = open(os.path.join(memory_location, hard_location_name), 'rb')
        except FileNotFoundError:
            return None
        counter = pickle.load(mem_file)
        if IntegerSDM.cached:
            IntegerSDM.cache[hard_location_name] = counter
        return counter

    @staticmethod
    def store_counters(hard_location_name, counters):
        if IntegerSDM.cached:
            IntegerSDM.cache[hard_location_name] = counters
        mem_file = open(os.path.join(memory_location, hard_location_name), 'wb')
        pickle.dump(counters, mem_file)

    @classmethod
    def access_sphere_radius(cls, ndim=ndim, r=r):
        # phi_inv = scipy.stats.norm.ppf(0.001) the calculation used below
        phi_inv = -3.0902323061678132  # for p=0.001 phi_inv(p) = -3.0902... , where p is the proportion of hard locations enclosed
        r_ = _axis_length
        access_sphere_radius = ((ndim*(r_**2+8)/48)**.5)*phi_inv+((ndim*r_)/4)
        return access_sphere_radius

    def __init__(self, n_hard_locations):
        self.hard_locations = [HardLocation(name='location'+str(i)) for i in range(n_hard_locations)]
        self.access_sphere_radius = IntegerSDM.access_sphere_radius()
        if _axis_length % 2 != 0:
            raise Exception('r should be even')

        try:
            os.mkdir(memory_location)
            created = True
        except FileExistsError:
            print("Location", os.path.join(os.getcwd(), memory_location),
                  " already stores a memory. Run from a different location by changing isdm.memory_location."
                  "Or remove the folder specified.")
            created = False
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise
        self._created = created

    @staticmethod
    def _has_converged(prev_distances):
        if len(prev_distances) > 1:
            if prev_distances[-1] < same_vector_distance_threshold:
                print(prev_distances)
                print("read by convergence")
                return True
        return False

    @staticmethod
    def _has_diverged(prev_distances):
        if np.mean([prev_distances[i]-prev_distances[i-1] for i in range(1, len(prev_distances))]) > 0:
            return True
        return False

    def _read_once(self, address):
        locations_in_radius = self.hard_locations_in_radius(address)

        if len(locations_in_radius) < 0:
            raise Exception("cannot be read")
        # adding the counters of locations in radius
        total = HardLocation.get_empty_counter()
        for location in locations_in_radius:
            total = location.add_counters(total)

        # creating the word from max of counters
        word = list()
        for counter in total:
            word.append(np.argmax(counter))
        word = MCRVector(np.array(word))
        return word

    def read(self, address, prev_distances=None):
        if prev_distances is None:
            prev_distances = []

        if self._has_converged(prev_distances):
            return address
        if self._has_diverged(prev_distances):
            return None

        word = self._read_once(address)
        prev_distances.append(address.distance(word))
        return self.read(word, prev_distances)

    def write(self, word):
        locations_in_radius = self.hard_locations_in_radius(word)

        if len(locations_in_radius) < 1:
            raise Exception("cannot be written")
        for location in locations_in_radius:
            location.write(word)

    def __del__(self):

        if hasattr(self, '_created'):
            if self._created:
                shutil.rmtree(memory_location)

    def hard_locations_in_radius(self, vector, radius=-1):
        if radius == -1:
            radius = self.access_sphere_radius

        hard_locations_in_radius = [location for location in self.hard_locations
                                    if vector.distance(location) < radius]

        return hard_locations_in_radius


class NPIntegerSDM(IntegerSDM):
    """ Faster Numpy based ISDM"""
    def __init__(self, n_hard_locations):
        super(NPIntegerSDM, self).__init__(n_hard_locations)
        hard_locations = np.array([location.dims for location in self.hard_locations])
        self._hard_locations_tensor = hard_locations

    def hard_locations_in_radius(self, vector, radius=-1):
        if not isinstance(vector, MCRVector):
            raise ValueError('Input vector must be an object of MCRVector')
        if radius == -1:
            radius = self.access_sphere_radius
        mod1_diff = (self._hard_locations_tensor-vector.dims) % _axis_length
        mod2_diff = (vector.dims-self._hard_locations_tensor) % _axis_length
        min_diff = np.minimum(mod1_diff, mod2_diff)
        distances = np.sum(min_diff, axis=1)
        in_radius = distances < radius
        locations = np.where(in_radius)
        return [self.hard_locations[location] for location in locations[0]]
#
# class TFIntegerSDM(IntegerSDM):
#
#     import tensorflow as tf
#     def __init__(self, n_hard_locations):
#         super(TFIntegerSDM, self).__init__(n_hard_locations)
#         hard_locations = np.array([location.dims for location in self.hard_locations])
#         graph = tf.Graph()
#         with graph.as_default():
#             self._hard_locations_tensor = tf.constant(hard_locations)
#             self._query_vector = tf.placeholder(dtype=tf.int32, shape=(ndim))
#             self._query_radius = tf.placeholder(dtype=tf.int32)
#             mod1_diff = (self._hard_locations_tensor-self._query_vector) % _axis_length
#             mod2_diff = (self._query_vector-self._hard_locations_tensor) % _axis_length
#             min_diff = tf.minimum(mod1_diff, mod2_diff)
#             distances = tf.reduce_sum(min_diff, axis=1)
#             in_radius = distances < self._query_radius
#             self._locations = tf.where(in_radius)
#         self._distance_checker = tf.Session(graph=graph)
#
#     def hard_locations_in_radius(self, vector, radius=-1):
#         if not isinstance(vector, MCRVector):
#             raise ValueError('Input vector must be an object of MCRVector')
#         if radius == -1:
#             radius = self.access_sphere_radius
#
#         locations = self._distance_checker.run(self._locations, feed_dict={self._query_vector: vector.dims,
#                                                                            self._query_radius: radius})
#         return [self.hard_locations[location_index[0]] for location_index in locations]

pam = IntegerSDM(100000)

def example1():
    cake = MCRVector.random_vector()
    ram = MCRVector.random_vector()
    apple = MCRVector.random_vector()
    sita = MCRVector.random_vector()
    eat = ram*cake + sita*apple

    cake.distance(ram)
    print(eat.distance(ram))

    pam.write(cake)
    pam.write(ram)
    pam.write(apple)
    pam.write(sita)

    noisy_ram = ram.get_noisy_copy()
    read_ram = pam.read(noisy_ram)
    noisy_ram.distance(ram)
    v1 = eat*(~ram)
    print(v1.distance(ram))
    print(v1.distance(eat))
    v1 = pam.read(v1)
    print(v1.distance(cake))
    print(v1.distance(sita))
    print(cake.distance(sita))


def create():
    v = MCRVector.random_vector()
    pam.write(v)
    return v


if __name__ == '__main__':

    thanos = create()
    scarlett_witch = create()
    luke = create()
    sith = create()
    villian = create()
    hero = create()

