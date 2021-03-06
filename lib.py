import numpy as np
import pyuff
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.optimize import curve_fit
from tqdm import tqdm


class Experiment:
    def __init__(self, raw_file, modeset_file=None, geometry=None):
        """
        Initialization of Experiment instance. Takes the data of LDV experiment and modal data of this experiment.
        Experiment should be performed with equidistant regular rectangular grid of measuring points.
        :param raw_file: str, Path to raw .uff experimental file.
        :param modeset_file: str,  Path to Simcenter Testlab .unv mode shape file.
        :param geometry: tuple of ints, dimensions of the experimental grid (optional).
        """
        print('Reading experimental file...')
        self.raw_data_file: pyuff.UFF = pyuff.UFF(raw_file)
        print('Experimental file has been read.')
        self.raw_data: np.array = None
        self.velocities: np.array = None
        self.exp_freqs: list = []
        self.eigenfreqs: list = []
        self.DBRs: dict = {}
        self.mode_shapes: np.array = None
        self.xs: list = []
        self.ys: list = []
        self.rs: np.array = None
        self.x_length: int = 0
        self.y_length: int = 0
        if geometry:
            self.extract_geometry(geometry)
        else:
            self.extract_geometry()
        self.extract_data_blocks()
        if modeset_file:
            print('Reading modeset file...')
            self.mode_shapes_file: pyuff.UFF = pyuff.UFF(modeset_file)
            print('Experimental file has been read.')
            self.extract_eigenfreqs()
            self.construct_mode_shapes()
        self.coeffs: np.array = None
        self.WDs: np.array = None
        self.WBP: np.array = None
        print("Experimental data processed successfully.")

    def extract_data_blocks(self):
        """
        Extracts experimental data (frequencies and vibrovelocities) from uff file.
        :return: None
        """
        print('Extracting experimental data...')
        indices = []
        set_types = self.raw_data_file.get_set_types()
        pbar = tqdm(total=len(set_types))
        for i, each in enumerate(set_types):
            pbar.update(1)
            if each == 58:
                indices.append(i)
        pbar.close()
        data = self.raw_data_file.read_sets(indices)
        df = pd.DataFrame(data)
        self.exp_freqs = data[0]['x']
        self.raw_data = np.stack(df[df['id1'] == 'Transfer Function H1']['data'].to_numpy())
        try:
            self.velocities = np.stack(df[df['id1'] == 'Response Linear Spectrum']['data'].to_numpy())
        except ValueError:
            self.velocities = None

    def extract_eigenfreqs(self):
        """
        Extracts eigenfrequencies from Testlab mode shape file.
        :return: None
        """
        print('Extracting mode shapes data...')
        indices = []
        if not self.mode_shapes_file:
            raise AttributeError("Modeset file was not specified for this experiment")
        set_types = self.mode_shapes_file.get_set_types()
        pbar = tqdm(total=len(set_types))
        for i, each in enumerate(set_types):
            if each == 55:
                indices.append(i)
            pbar.update(1)
        pbar.close()
        data = self.mode_shapes_file.read_sets(indices)
        eigenfreqs = []
        for each in data:
            if "Residuals" in each['id1']:
                pass
            else:
                _, test, _ = each['id4'].split(",")
                _, _, test = test.split(" ")
                test, _ = test.split("(")
                eigenfreqs.append(float(test))
        self.eigenfreqs = eigenfreqs

    def set_modeset_file(self, modeset_file: str):
        """
        Sets the mode shapes file for the experiment.
        :param modeset_file: str, path to modeset.unv file.
        :return: None.
        """
        self.mode_shapes_file: pyuff.UFF = pyuff.UFF(modeset_file)

    def extract_geometry(self, geometry: tuple = None) -> tuple:
        """
        Extracts (tries to) dimensions of experimental grid from raw .uff file.
        :param geometry: (optional) tuple of ints, for the specification of dimensions if the extraction is unsuccessful
        :return: tuple of ints, width and height of the experimental grid (lengths of x and y axes of rectangular grid).
        """
        if geometry:
            self.x_length = geometry[0]
            self.y_length = geometry[1]
            return self.x_length, self.y_length
        data = self.raw_data_file.read_sets(2)
        self.xs = data['x']
        self.ys = data['y']
        i = 0
        step = max(self.ys[0], self.ys[1]) - min(self.ys[0], self.ys[1])
        y_min = min(self.ys)
        for each in self.ys:
            if each - y_min < step:
                i += 1
        length = len(self.xs)
        if (length / i).is_integer():
            self.y_length = i
        elif (length / (i - 1)).is_integer():
            self.y_length = i - 1
        elif (length / (i + 1)).is_integer():
            self.y_length = i + 1
        else:
            print('Error! Unable to detect geometry. Set geometry manually.')
        self.x_length = int(length / self.y_length)
        print(f"Detected geometry: {self.x_length}x{self.y_length} points")
        return self.x_length, self.y_length

    def construct_mode_shapes(self):
        """
        Constructs modeshapes as amplitudes of FRFs of each scanning point (at frequencies, closest to eigenfrequencies)
        :return: None
        """
        print('Constructing mode shapes...')
        modeshapes = {}
        work_freq_indices = self.associate_frequencies()
        for index in work_freq_indices:
            modeshape = np.empty(0)
            for point in self.raw_data:
                modeshape = np.append(modeshape, point[index])
            modeshape = modeshape.reshape(self.y_length, self.x_length)
            modeshapes[int(np.around(self.exp_freqs[index]))] = modeshape
        print('Mode shapes constructed.')
        self.mode_shapes = modeshapes

    def stack_mode_shapes(self, dims: int = 2, flip: bool = False, dtype: str = '') -> np.array:
        """
        Stacks mode shapes into one 2D-array or 3D-array.
        :param dims: int, specifies the dimensions for output array. Another words, specifies if the mode shapes should
        be represented as vector or as matrix.
        :param flip: bool, specifies if the mode shapes should be flipped along the horizontal axis.
        :param dtype: str, options: empty str '' - complex value of FRF will be stacked, 'amplitude' - only amplitudes will be
         stacked, 'phase' - only phases will be stacked.
        :return: np.array, 2D or 3D-array with mode shapes, where mode shapes are along axis 0 either as vector or as
        matrix.
        """
        if dims == 2:
            modeset = np.empty((0, self.y_length * self.x_length))
        elif dims == 3:
            modeset = np.empty((0, self.y_length, self.x_length))
        else:
            raise ValueError('The given number of axes is not correct. Dims = 2 (with flattened modal vectors) or '
                             'dims = 3 (with modal shapes of experimental grid dimensions) should be used.')
        for key, value in self.mode_shapes.items():
            if dtype == 'amplitude':
                if not flip:
                    modeset = np.append(modeset, [np.abs(value).reshape(-1) if dims == 2 else np.abs(value)], axis=0)
                else:
                    modeset = np.append(modeset, [np.flip(np.abs(value), axis=1).reshape(-1) if dims == 2 else
                                                  np.flip(np.abs(value), axis=1)], axis=0)
            elif dtype == 'phase':
                if not flip:
                    modeset = np.append(modeset, [convert_to_phase(value).reshape(-1) if dims == 2 else
                                                  convert_to_phase(value)], axis=0)
                else:
                    modeset = np.append(modeset, [np.flip(convert_to_phase(value), axis=1).reshape(-1) if dims == 2 else
                                                  np.flip(convert_to_phase(value), axis=1)], axis=0)
            elif dtype == '':
                if not flip:
                    modeset = np.append(modeset, [value.reshape(-1) if dims == 2 else value], axis=0)
                else:
                    modeset = np.append(modeset, [np.flip(value, axis=1).reshape(-1) if dims == 2 else
                                                  np.flip(value, axis=1)], axis=0)
        return modeset

    def visualize_mode_shape(self, frequency: int, **kwargs):
        """
        Visualizes mode shape, either specified with frequency, or with number.
        :param frequency: int, frequency or number of mode (if the given number is less than 150, it will be interpreted
        as number of mode shape, otherwise - as frequency.
        :param kwargs: optional arguments for matplotlib (plt.plot)
        :return: None
        """
        if not isinstance(frequency, list):
            frequency = [frequency]
        for each in frequency:
            fig, ax = plt.subplots()
            plt.set_cmap('jet')
            if each < 150:
                freq = list(self.mode_shapes.keys())[each]
                im = ax.imshow(np.rot90(np.abs(self.mode_shapes[freq])), **kwargs)
                fig.colorbar(im)
                plt.title(f'Modeshape at frequency {freq} Hz')
            else:
                try:
                    im = ax.imshow(np.rot90(np.abs(self.mode_shapes[each])), **kwargs)
                    fig.colorbar(im)
                    plt.title(f'Modeshape at frequency {each} Hz')
                except KeyError:
                    print('No such eigenfrequency. Try another one.')
                    print('Extracted eifgenfrequencies: ', list(self.mode_shapes.keys()))

    def calculate_dbrs(self, coords):
        """
        Calculates DBRs for each mode shape, taking the rectangle with coordinates, specified in coords variable,
        as defected zone. Stores each calculation in the dict with coordinates of window and DBR values.
        :param coords: list of tuples of ints, specifies the rectangle of defected zone.
        :return: np.array, array with dbr values. One value for each modeshape.
        """
        dbrs = np.empty(0)
        total_area = self.x_length * self.y_length
        defected_area = (coords[1][0] - coords[0][0]) * (coords[2][1] - coords[0][1])
        for freq, mode_shape in self.mode_shapes.items():
            defected_region_magnitudes = 0
            intact_region_magnitudes = 0
            for y in range(coords[0][1], coords[2][1]):
                for x in range(coords[0][0], coords[1][0]):
                    defected_region_magnitudes += np.abs(mode_shape[x][y])
                    intact_region_magnitudes = np.sum(np.abs(mode_shape)) - defected_region_magnitudes
            dbrs = np.append(dbrs, defected_region_magnitudes * (total_area - defected_area) /
                             (intact_region_magnitudes * defected_area))
        coords_str = ''
        for i in coords:
            coords_str += str(i) + ' '
        self.DBRs[coords_str] = dbrs
        plt.plot(dbrs)
        return dbrs

    def move_and_calculate_dbr_window(self, size, step):
        """
        Iteratively calculates DBRs for the given size of rectangle with specified step.
        :param size: int or tuple of ints, size of zone to be considered as defected (square if int, rectangle if
        tuple of ints)
        :param step: The step with which the window of defected area is moved along the specimen.
        :return: 2-D np array, where each row is the DBR values for all mode shapes. The number of rows equals to the
        number of unique positions of the window for the given combination of step and size.
        """
        if isinstance(step, tuple):
            step_x = step[0]
            step_y = step[1]
        elif isinstance(step, int):
            step_x = step_y = step
        else:
            raise ValueError("Error! Step was filled incorrectly.")

        if isinstance(size, int):
            size = (size, size)

        dbrs = np.empty((0, len(self.mode_shapes)))
        if size[0] >= self.x_length:
            raise ValueError("Error! The requested window x-size is greater than the experimental geometry.")
        if size[1] >= self.y_length:
            raise ValueError("Error! The requested window y-size is greater than the experimental geometry.")
        for y in range(0, self.x_length - size[1] - 1, step_x):
            for x in range(0, self.y_length - size[0] - 1, step_y):
                print("Calculating DBR for coordinates ",
                      [(x, y), (x + size[0], y), (x, y + size[1]), (x + size[0], y + size[1])])
                dbrs = np.append(dbrs,
                                 [self.calculate_dbrs([(x, y), (x + size[0], y),
                                                       (x, y + size[1]), (x + size[0], y + size[1])])], axis=0)
        return dbrs

    def interpolate_one_modeset(self, grid_x, grid_y, method='nearest'):
        print("Started interpolation")
        zs_interpolated = np.zeros((0, grid_x.shape[0] * grid_x.shape[1]))
        for each in range(len(self.mode_shapes)):
            interpolation = griddata((self.xs, self.ys), self.mode_shapes[each], (grid_x, grid_y), method=method). \
                reshape(-1)
            zs_interpolated = np.append(zs_interpolated, [interpolation], axis=0)
        print("Interpolation complete")
        return zs_interpolated

    def find_index_from_frequency(self, frequency):
        """
        Performs binary search to find an index of the given frequency.
        :param frequency: int, frequency for which the index will be returned.
        :return: int, index for the given frequency.
        """
        return np.searchsorted(self.eigenfreqs, frequency)

    def associate_frequencies(self):
        """
        Associates the eigenfrequencies with the closest experimental frequency.
        :return: Indices of experimental frequencies that are associated with mode shapes.
        """
        eigenfreqs_indices = np.empty(0, dtype=np.int32)
        for each in self.eigenfreqs:
            ind = np.searchsorted(self.exp_freqs, each)
            if abs(self.exp_freqs[ind] - each) < abs(self.exp_freqs[ind - 1] - each):
                eigenfreqs_indices = np.append(eigenfreqs_indices, ind)
            else:
                eigenfreqs_indices = np.append(eigenfreqs_indices, ind - 1)
        return eigenfreqs_indices

    def get_raw_data(self):
        """
        Gives an access for the raw experimental FRFs.
        :return: 2D numpy array, with FRFs on all measured frequencies.
        """
        return self.raw_data

    def get_velocities(self):
        """
        Gives an access for the raw experimental velocities.
        :return: 2D numpy array, with FRFs on all measured frequencies.
        """
        return self.velocities

    def get_exp_freqs(self):
        """
        Gives access for the experimental frequencies.
        :return: list, list of experimental frequencies
        """
        return self.exp_freqs

    def get_eigenfreqs(self):
        """
        Gives an access for the eigenfrequencies, extracted from the unv testlab file.
        :return: list, list with eigenfrequencies.
        """
        return self.eigenfreqs

    def fit_curves(self, pzt_location: tuple) -> np.array:
        """
        Calculates the WD functions for each spectral line available.
        :param pzt_location: tuple of ints, coordinates of PZT location.
        :return: 2D-np.array, the array with exponential decay function coefficients for each spectral line
        """
        pzt_x, pzt_y = pzt_location
        coeffs = np.empty((0, 3))
        rs = np.empty(0)
        wds = np.empty((0, self.velocities.shape[0]))
        for point_number in range(len(self.velocities)):
            x = int(point_number/self.x_length)
            y = self.x_length - (point_number - int(x*self.x_length))
            r = np.sqrt((x-pzt_x)**2 + (y-pzt_y)**2)
            rs = np.append(rs, r)
        self.rs = rs
        init_popt, _ = curve_fit(exponential_decay, rs, np.abs(self.velocities.T[int(self.velocities.T.shape[0]/2)])**2,
                                 maxfev=5000)
        for each in tqdm(self.velocities.T):
            popt, _ = curve_fit(exponential_decay, rs, np.abs(each)**2, p0=init_popt, maxfev=5000)
            coeffs = np.append(coeffs, [popt], axis=0)
            wd_temp = exponential_decay(rs, popt[0], popt[1], popt[2])
            wds = np.append(wds, [wd_temp], axis=0)
        self.coeffs = coeffs
        self.WDs = wds
        return coeffs, wds

    def calculate_wbp(self, **kwargs) -> np.array:
        """
        This function calculates and visualizes Weighted Band Power (WBP) of the given experiment.
        :return: 2D np.array with WBP of experiment.
        """
        WBP = np.zeros(self.raw_data.shape[0])
        for number, each in enumerate(self.raw_data.T):
            WBP += np.abs(each)**2/self.WDs[number]
        fig, ax = plt.subplots()
        plt.set_cmap('jet')
        im = ax.imshow(np.rot90(WBP.reshape(self.y_length, self.x_length)), **kwargs)
        fig.colorbar(im)
        plt.title("Weighted Band Power")
        return WBP

    def plot_fit_curve(self, line):
        fig, ax = plt.subplots()
        plt.set_cmap('jet')
        plt.plot(self.rs, exponential_decay(self.rs, self.coeffs[line][0], self.coeffs[line][1], self.coeffs[line][2]))
        plt.scatter(self.rs, np.abs(self.velocities.T[line])**2)
        plt.title(f'Decay function of spectral line number {line}')


def mac(modeset1, modeset2, title: str = None):
    """
    Calculates MAC matrix for the given sets of real modal vectors. Applicable only for float values.
    :param modeset1: 2D numpy array of floats, set of modal vectors of the first experiment, rows are modal vectors.
    :param modeset2: 2D numpy array of floats, set of modal vectors of the first experiment, rows are modal vectors.
    :param title: str, title for plot with MAC.
    :return: 2D numpy array of floats, MAC matrix.
    """
    if isinstance(modeset1[0][0], np.complex) or isinstance(modeset2[0][0], np.complex):
        raise ValueError('One of modesets is complex. Use function complex_mac instead!')
    if modeset1.shape[0] > modeset2.shape[0]:
        shape = modeset2.shape[0]
    else:
        shape = modeset1.shape[0]
    res = np.zeros((shape, shape))
    for i in range(shape):
        for j in range(shape):
            res[i][j] = (np.abs(np.dot(modeset1[i].T, modeset2[j])) ** 2) \
                        / (np.dot(modeset1[i].T, modeset1[i]) * np.dot(modeset2[j].T, modeset2[j]))
    fig, ax = plt.subplots()
    plt.set_cmap('jet')
    im = ax.imshow(res, vmin=0, vmax=1)
    plt.colorbar(im)
    plt.gca().invert_yaxis()
    if title:
        plt.title(title)
    return res


def complex_mac(modeset1: np.array, modeset2: np.array, region: tuple = None, title: str = None) -> np.array:
    """
    Calculates MAC matrix for the given sets of complex modal vectors. Applicable only for complex values.
    :param modeset1: 2D numpy array of floats, set of modal vectors of the first experiment, rows are modal vectors.
    :param modeset2: 2D numpy array of floats, set of modal vectors of the first experiment, rows are modal vectors.
    :param region: tuple, a tuple of tuples (coordinates of the rectangle's diagonal for which the MAC is performed)
    :param title: str, title for plot with MAC.
    :return: 2D numpy array of floats, MAC matrix.
    """
    if isinstance(modeset1[0][0], np.float32) or isinstance(modeset2[0][0], np.float32):
        raise ValueError('One of modesets is not complex. Use function mac instead!')
    if region is not None and (modeset1.ndim != 3 or modeset2.ndim != 3):
        raise ValueError('The given modesets are not 3-dims. Regional MAC requires 3D stacking of modeshapes.')
    if modeset1.shape[0] > modeset2.shape[0]:
        shape = modeset2.shape[0]
    else:
        shape = modeset1.shape[0]
    res = np.zeros((shape, shape))
    for i in range(shape):
        for j in range(shape):
            if region:
                res[i][j] = (np.abs(np.vdot(modeset2[j, region[0][1]:region[1][1], region[0][0]:region[1][0]].
                                            reshape(-1),
                                            modeset1[i, region[0][1]:region[1][1], region[0][0]:region[1][0]].
                                            reshape(-1))) ** 2) / np.real(
                    np.vdot(modeset1[i, region[0][1]:region[1][1], region[0][0]:region[1][0]].reshape(-1),
                            modeset1[i, region[0][1]:region[1][1], region[0][0]:region[1][0]].reshape(-1)) *
                    np.vdot(modeset2[j, region[0][1]:region[1][1], region[0][0]:region[1][0]].reshape(-1),
                            modeset2[j, region[0][1]:region[1][1], region[0][0]:region[1][0]].reshape(-1)))
            elif modeset1.ndim == 2 and modeset2.ndim == 2:
                res[i][j] = (np.abs(np.vdot(modeset2[j], modeset1[i].T)) ** 2) / np.real(
                    np.vdot(modeset1[i], modeset1[i]) * np.vdot(modeset2[j], modeset2[j]))
            else:
                res[i][j] = (np.abs(np.vdot(modeset2[j].reshape(-1), modeset1[i].reshape(-1))) ** 2) / np.real(
                    np.vdot(modeset1[i].reshape(-1), modeset1[i].reshape(-1)) * np.vdot(modeset2[j].reshape(-1),
                                                                                          modeset2[j].reshape(-1)))
    if not region:
        fig, ax = plt.subplots()
        plt.set_cmap('jet')
        im = ax.imshow(res, vmin=0, vmax=1)
        plt.colorbar(im)
        plt.gca().invert_yaxis()
        if title:
            plt.title(title)
    return res


def slide_grid_and_calculate_mac(modeset1: np.array, modeset2: np.array, direction: tuple,
                                 title: str = None, mac_is_complex: bool = True) -> np.array:
    """
    :param modeset1: 3D numpy array of floats, set of modal vectors of the first experiment, rows are modal vectors.
    :param modeset2: 3D numpy array of floats, set of modal vectors of the first experiment, rows are modal vectors.
    :param direction: tuple, specifying how to move one matrix. First value for x axis, second for y.
    :param mac_is_complex: bool, if True - uses the complex_mac function to calculate MAC matrix. Otherwise - uses mac
    function for abs amplitudes comparison.
    :param title: title for MAC plot.
    :return: 2D numpy array of floats, MAC matrix.
    """
    if modeset1.ndim == 2 or modeset2.ndim == 2:
        raise ValueError("Error! The given modesets must be 3D arrays (mode shapes must be represented as matrices)")
    if direction[1] == 0 and direction[0] == 0:
        modeset1_slided = modeset1[:, :, :]
        modeset2_slided = modeset2[:, :, :]
    elif direction[0] == 0:
        modeset1_slided = modeset1[:, :, direction[1]:]
        modeset2_slided = modeset2[:, :, :-direction[1]]
    elif direction[1] == 0:
        modeset1_slided = modeset1[:, :-direction[0], :]
        modeset2_slided = modeset2[:, direction[0]:, :]
    else:
        modeset1_slided = modeset1[:, :-direction[0], direction[1]:]
        modeset2_slided = modeset2[:, direction[0]:, :-direction[1]]
    shape1 = modeset1_slided.shape
    shape2 = modeset2_slided.shape
    if mac_is_complex:
        return complex_mac(
            modeset1_slided.reshape((shape1[0], shape1[1]*shape1[2])),
            modeset2_slided.reshape((shape2[0], shape2[1]*shape2[2])), title=title)
    else:
        return mac(
            np.abs(modeset1_slided.reshape((shape1[0], shape1[1] * shape1[2]))),
            np.abs(modeset2_slided.reshape((shape2[0], shape2[1] * shape2[2]))), title=title)


def sliding_mac_procedure(modeset1: np.array,  modeset2: np.array, window: tuple) -> None:
    macs = np.empty((0, min(modeset1.shape[0], modeset2.shape[0]), min(modeset1.shape[0], modeset2.shape[0])))
    window_height = window[1]
    y_steps = int(modeset1.shape[1]/window_height)
    window_width = window[0]
    x_steps = int(modeset1.shape[2]/window_width)
    for y in range(y_steps-1):
        for x in range(x_steps-1):
            mac = complex_mac(modeset1, modeset2, region=((window_width*x, window_height*y),
                                                          (window_width*(x+1), window_height*(y+1))))
            macs = np.append(macs, [mac], axis=0)
    fig, axs = plt.subplots(x_steps-1, y_steps-1)
    plt.set_cmap('jet')
    for m in range(y_steps-1):
        for n in range(x_steps-1):
            print(m, n, m*(x_steps-1)+n)
            if isinstance(axs, np.ndarray):
                im = axs[n][m].imshow(macs[m*(x_steps-1)+n])
                axs[n][m].invert_yaxis()
                axs[n][m].axis('off')
            else:
                im = axs.imshow(macs[0])
                axs.invert_yaxis()


def convert_to_phase(values: np.array) -> np.array:
    """
    Takes the array of FRF values and extracts phase from each element.
    :param values: np.array with complex values of FRF.
    :return: np.array, an array of :values: shape where each element is phase in angles.
    """
    phases = np.arctan2(np.imag(values), np.real(values))
    return phases * 180 / np.pi


def exponential_decay(x: float or np.array, a: float, b: float, c: float) -> float:
    """
    Calculates the exponential decay for the given x and coefficients.
    :param x: variable.
    :param a: coefficient a.
    :param b: coefficient b.
    :param c: coefficient c.
    :return: the value of function for the given set of values.
    """
    return a * np.exp(-b * x) + c


def biexponential_decay(x: float or np.array, a1: float, b1: float, c: float, a2: float, b2: float) -> float:
    """
    Calculates the biexponential decay for the given x and coefficients.
    :param x: variable.
    :param a1: coefficient a1.
    :param b1: coefficient b1.
    :param c: coefficient c.
    :param a2: coefficient a2.
    :param b2: coefficient b2.
    :return: the value of function for the given set of values.
    """
    return a1 * np.exp(-b1 * x) + c + a2 *np.exp(-b2 * x)