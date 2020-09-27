import numpy as np
import pyuff
import pandas as pd
import matplotlib.pyplot as plt


class Experiment:
    def __init__(self, raw_file, modeset_file, geometry=None):
        self.raw_data_file = pyuff.UFF(raw_file)
        self.mode_shapes_file = pyuff.UFF(modeset_file)
        self.raw_data = None
        self.exp_freqs = []
        self.eigenfreqs = []
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
        print("Experimental data processed successfully.")

    def extract_data_blocks(self):
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

    def visualize_normalized_modeshape(self, frequency, **kwargs):
        if not isinstance(frequency, list):
            frequency = [frequency]
        for each in frequency:
            if each < 150:
                freq = list(self.mode_shapes.keys())[each]
                fig, ax = plt.subplots()
                plt.set_cmap('jet')
                im = ax.imshow(np.rot90(np.abs(self.mode_shapes[freq]) /
                                        self.average_values[each]), kwargs)
                fig.colorbar(im)
                plt.title(f'Modeshape at frequency {freq} Hz')
            else:
                try:
                    fig, ax = plt.subplots()
                    plt.set_cmap('jet')
                    im = ax.imshow(np.rot90(np.abs(self.mode_shapes[each])) /
                                   self.average_values[self.find_index_from_frequency(each)], kwargs)
                    fig.colorbar(im)
                    plt.title(f'Modeshape at frequency {each} Hz')
                except KeyError:
                    print('No such eigenfrequency. Try another one.')
                    print('Extracted eifgenfrequencies: ', list(self.mode_shapes.keys()))

    def find_index_from_frequency(self, frequency):
        return np.searchsorted(self.eigenfreqs, frequency)

    def associate_frequencies(self):
        eigenfreqs_indices = np.empty(0, dtype=np.int32)
        for each in self.eigenfreqs:
            ind = np.searchsorted(self.exp_freqs, each)
            if abs(self.exp_freqs[ind]-each) < abs(self.exp_freqs[ind-1]-each):
                eigenfreqs_indices = np.append(eigenfreqs_indices, ind)
            else:
                eigenfreqs_indices = np.append(eigenfreqs_indices, ind-1)
        return eigenfreqs_indices

    def get_raw_data(self):
        return self.raw_data

    def get_exp_freqs(self):
        return self.exp_freqs

    def get_eigenfreqs(self):
        return self.eigenfreqs
