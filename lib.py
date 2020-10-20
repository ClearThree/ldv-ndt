import numpy as np
import pyuff
import pandas as pd
import matplotlib.pyplot as plt


class Experiment:
    def __init__(self, raw_file, modeset_file, geometry=None):
        """
        Initialization of Experiment instance. Takes the data of LDV experiment and modal data of this experiment.
        Experiment should be performed with equidistant regular rectangular grid of measuring points.
        :param raw_file: str, Path to raw .uff experimental file.
        :param modeset_file: str,  Path to Simcenter Testlab .unv mode shape file.
        :param geometry: tuple of ints, dimensions of the experimental grid (optional).
        """
        self.raw_data_file = pyuff.UFF(raw_file)
        self.mode_shapes_file = pyuff.UFF(modeset_file)
        self.raw_data = None
        self.exp_freqs = []
        self.eigenfreqs = []
        self.dbrs = {}
        self.mode_shapes = None
        self.average_values = np.empty(0)
        self.x_length = 0
        self.y_length = 0
        self.extract_data_blocks()
        self.extract_eigenfreqs()
        if geometry:
            self.extract_geometry(geometry)
        else:
            self.extract_geometry()
        self.construct_mode_shapes()
        self.DBRs = {}
        print("Experimental data processed successfully.")

    def extract_data_blocks(self):
        """
        Extracts experimental data (frequencies and vibrovelocities) from uff file.
        :return: None
        """
        print('Extracting experimental data...')
        indices = []
        for i, each in enumerate(self.raw_data_file.get_set_types()):
            if each == 58:
                indices.append(i)
        data = self.raw_data_file.read_sets(indices)
        df = pd.DataFrame(data)
        self.exp_freqs = data[0]['x']
        self.raw_data = df['data']

    def extract_eigenfreqs(self):
        """
        Extracts eigenfrequencies from Testlab mode shape file.
        :return: None
        """
        print('Extracting mode shapes data...')
        indices = []
        for i, each in enumerate(self.mode_shapes_file.get_set_types()):
            if each == 55:
                indices.append(i)
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

    def extract_geometry(self, geometry=None):
        """
        Extracts (tries to) dimensions of experimental grid from raw .uff file.
        :param geometry: (optional) tuple of ints, for the specification of dimensions if the extraction is unsuccessful
        :return: int, int, width and height of the experimental grid (lengths of x and y axes of rectangular grid).
        """
        if geometry:
            self.x_length = geometry[0]
            self.y_length = geometry[1]
            return self.x_length, self.y_length
        data = self.mode_shapes_file.read_sets(2)
        y = data['y']
        i = 0
        step = max(y[0], y[1]) - min(y[0], y[1])
        y_min = min(y)
        for each in y:
            if each - y_min < step:
                i += 1
        length = len(self.raw_data)
        print(i)
        print(length)
        if (length / i).is_integer():
            self.y_length = i
        elif (length / (i-1)).is_integer():
            self.y_length = i-1
        elif (length / (i+1)).is_integer():
            self.y_length = i+1
        else:
            print('Error! Unable to detect geometry. Set geometry manually.')
        self.x_length = int(length/self.y_length)
        return self.x_length, self.y_length

    def construct_mode_shapes(self):
        """
        Constructs modeshapes as amplitudes of FRFs of each scanning point (at frequencies, closest to eigenfrequencies)
        :return: None
        """
        print('Constructing mode shapes...')
        modeshapes = {}
        average_values = np.empty(0)
        work_freq_indices = self.associate_frequencies()
        for index in work_freq_indices:
            modeshape = np.empty(0)
            for point in self.raw_data:
                modeshape = np.append(modeshape, point[index])
            average_values = np.append(average_values, np.average(np.abs(modeshape)))
            modeshape = modeshape.reshape(self.y_length, self.x_length)
            modeshapes[int(np.around(self.exp_freqs[index]))] = modeshape
        self.mode_shapes = modeshapes
        self.average_values = average_values

    def visualize_modeshape(self, frequency, **kwargs):
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
            if each < 150:
                freq = list(self.mode_shapes.keys())[each]
                fig, ax = plt.subplots()
                plt.set_cmap('jet')
                im = ax.imshow(np.rot90(np.abs(self.mode_shapes[freq])), **kwargs)
                fig.colorbar(im)
                plt.title(f'Modeshape at frequency {freq} Hz')
            else:
                try:
                    fig, ax = plt.subplots()
                    plt.set_cmap('jet')
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
        total_area = self.x_length*self.y_length
        defected_area = (coords[1][0]-coords[0][0]) * (coords[2][1]-coords[0][1])
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
        for y in range(0, self.x_length - size[1]-1, step_x):
            for x in range(0, self.y_length - size[0]-1, step_y):
                print("Calculating DBR for coordinates ",
                      [(x, y), (x+size[0], y), (x, y+size[1]), (x+size[0], y+size[1])])
                dbrs = np.append(dbrs,
                                 [self.calculate_dbrs([(x, y), (x+size[0], y), (x, y+size[1]), (x+size[0], y+size[1])])],
                                 axis=0)
        return dbrs

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
        :return: Indices of experimental frequencies that are associated with the mode shapes.
        """
        eigenfreqs_indices = np.empty(0, dtype=np.int32)
        for each in self.eigenfreqs:
            ind = np.searchsorted(self.exp_freqs, each)
            if abs(self.exp_freqs[ind]-each) < abs(self.exp_freqs[ind-1]-each):
                eigenfreqs_indices = np.append(eigenfreqs_indices, ind)
            else:
                eigenfreqs_indices = np.append(eigenfreqs_indices, ind-1)
        return eigenfreqs_indices

    def get_raw_data(self):
        """
        Gives an access for the raw_data UFF object.
        :return: obj, pyuff.UFF object with raw experimental data.
        """
        return self.raw_data

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
