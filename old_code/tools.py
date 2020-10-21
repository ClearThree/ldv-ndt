import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyuff
import skimage.io
from scipy.interpolate import griddata
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import SGD


def split_frequency(string):
    _, freq = string.split(':')
    freq, _ = freq.split('\n')
    return float(freq)


def displacements_from_exp_unv(file):
    unv_file = pyuff.UFF(os.path.join(os.getcwd(), file))
    types = unv_file.get_set_types()
    indexes = []
    for idx, each in enumerate(types):
        if each == 15 or each == 2411:
            coor_dict = unv_file.read_sets(idx)
            x = np.array(coor_dict['x'])
            y = np.array(coor_dict['y'])
            x = x - np.min(x)
            y = y - np.min(y)
        if each == 55:
            indexes.append(idx)
    x = x * 1000
    y = y * 1000
    hold = x
    x = y
    y = hold
    values = np.zeros((0, len(x)))
    nums = unv_file.read_sets(indexes[0])['node_nums']
    deect = {'Node_numbers': nums}
    freqs = np.empty(0, dtype=np.float32)
    for each in indexes:
        data = unv_file.read_sets(each)
        deect.update({data['id4']: data['r3']})
        _, test, _ = data['id4'].split(",")
        _, _, test = test.split(" ")
        test, _ = test.split("(")
        freqs = np.append(freqs, float(test))
    df = pd.DataFrame(deect).sort_values(by='Node_numbers')
    df = df.drop(['Node_numbers'], axis=1).reset_index().drop(['index'], axis=1)
    for each in df.columns:
        value = np.array(df[each]).imag
        values = np.append(values, [value], axis=0)
    return x, y, freqs, values


def displacements_from_sim_txt(file):
    """
    :param: file: A .txt file with nodes displacements on eigenfrequencies of simulated plate.
    :return: numpy arrays with original x and y coordinates of nodes, eigenfrequencies and displacements.
    """
    print("Started the parsing of txt file: ", file)
    print("Processing...")
    file = os.path.join(os.getcwd(), file)
    f = open(file, 'r')
    lines = f.readlines()
    x_coors = np.empty(0)
    y_coors = np.empty(0)
    freqs = np.empty(0, dtype=float)
    for idx, line in enumerate(lines):
        if 'F:' not in line:
            n, x, y, z = line.split(' ')
            x_coors = np.append(x_coors, float(x))
            y_coors = np.append(y_coors, float(y))
        else:
            xs = np.zeros((0, len(x_coors)))
            ys = np.zeros((0, len(y_coors)))
            zs = np.zeros((0, len(x_coors)))
            break
    freq = split_frequency(lines[idx])
    freqs = np.append(freqs, float(freq))
    lines = lines[idx + 1:]
    ux = np.empty(0, dtype=float)
    uy = np.empty(0, dtype=float)
    uz = np.empty(0, dtype=float)
    for idx, line in enumerate(lines):
        if 'F:' not in line:
            x, y, z = line.split(' ')
            z, _ = z.split('\n')
            ux = np.append(ux, float(x))
            uy = np.append(uy, float(y))
            uz = np.append(uz, float(z))
        elif 'F:' in line:
            xs = np.append(xs, [ux], axis=0)
            ys = np.append(ys, [uy], axis=0)
            zs = np.append(zs, [uz], axis=0)
            freq = split_frequency(lines[idx])
            freqs = np.append(freqs, float(freq))
            ux = np.empty(0, dtype=float)
            uy = np.empty(0, dtype=float)
            uz = np.empty(0, dtype=float)
    print("Txt file processed.")
    return x_coors, y_coors, freqs, xs, ys, zs


def calculate_mac_xyz(file, grid_density=1, use_cache=True, file_ref='None'):
    if grid_density == 1 and use_cache:
        x_coors_ref = np.load("x_coors_ref.npy")
        y_coors_ref = np.load("y_coors_ref.npy")
        # freqs_ref = np.load("freqs_ref.npy")
        phi_ref = np.load("phi_ref.npy")
    else:
        if file_ref == 'None':
            file_ref = "CFRP_fbh_circle_0r_0d_c(0_0_0).txt"
        x_coors_ref, y_coors_ref, freqs_ref, xs_ref, ys_ref, zs_ref = displacements_from_sim_txt(file_ref)

    x_coors, y_coors, freqs, xs, ys, zs = displacements_from_sim_txt(file)
    grid_x, grid_y = np.mgrid[min(x_coors_ref):max(x_coors_ref):grid_density,
                              min(y_coors_ref):max(y_coors_ref):grid_density]
    xs_interpolated, ys_interpolated, zs_interpolated = interpolate_vectors(x_coors, y_coors, xs, ys, zs, grid_x, grid_y)
    phi = stack_vectors(xs_interpolated, ys_interpolated, zs_interpolated, grid_x)
    if not use_cache:
        phi_ref = interpolate_and_stack_vectors(x_coors_ref, y_coors_ref, xs_ref, ys_ref, zs_ref, grid_x, grid_y)
    print('Mac calculation in progress')
    mac_values = MAC(phi, phi_ref)
    return mac_values.reshape((-1)), zs_interpolated, freqs, grid_x, grid_y


def calculate_mac_z(file, grid_density=1, use_cache=True, file_ref='None'):
    if grid_density == 1 and use_cache:
        x_coors_ref = np.load("x_coors_ref.npy")
        y_coors_ref = np.load("y_coors_ref.npy")
        phi_ref = np.load("phi_ref.npy")
    else:
        if file_ref == 'None':
            file_ref = "CFRP_fbh_circle_0r_0d_c(0_0_0).txt"
        x_coors_ref, y_coors_ref, freqs_ref, _, _, zs_ref = displacements_from_sim_txt(file_ref)

    x_coors, y_coors, freqs, values = displacements_from_exp_unv(file)
    grid_x, grid_y = np.mgrid[min(x_coors):max(x_coors):grid_density,
                              min(y_coors):max(y_coors):grid_density]
    zs_interpolated = interpolate_one_vector(x_coors, y_coors, values, grid_x, grid_y)
    mac_values = load_simc_mac(file)
    print(grid_x.shape)
    return mac_values.reshape((-1)), zs_interpolated, freqs, grid_x, grid_y


def load_simc_mac(file):
    if file:
        print("Loading exp-sim MAC file")
    return np.load('mac_exp_simc_values_170cut.npy')


def MAC(matrix1, matrix2):
    if matrix1.shape[0] < 170 or matrix2.shape[0] < 170:
        print("Warning! Shapes are less than 170!", matrix1.shape[0], matrix2.shape[0])
    if matrix1.shape[0] > matrix2.shape[0]:
        shape = matrix2.shape[0]
    else:
        shape = matrix1.shape[0]
    res = np.zeros((shape, shape))
    for i in range(shape):
        for j in range(shape):
            res[i][j] = (np.dot(matrix1[i].T, matrix2[j]) ** 2) / (
                    np.dot(matrix1[i].T, matrix1[i]) * np.dot(matrix2[j].T,
                                                              matrix2[j]))
    return res


def interpolate_vectors(x_coors, y_coors, xs, ys, zs, grid_x, grid_y, cut=170, method='nearest'):
    xs = xs[:cut]
    ys = ys[:cut]
    print("Started interpolation")
    xs_interpolated = np.zeros((0, grid_x.shape[0] * grid_x.shape[1]))
    ys_interpolated = np.zeros((0, grid_x.shape[0] * grid_x.shape[1]))
    zs_interpolated = np.zeros((0, grid_x.shape[0] * grid_x.shape[1]))
    for each in range(len(xs)):
        interpolation = griddata((x_coors, y_coors), zs[each], (grid_x, grid_y), method=method).reshape(-1)
        zs_interpolated = np.append(zs_interpolated, [interpolation], axis=0)
        interpolation = griddata((x_coors, y_coors), xs[each], (grid_x, grid_y), method=method).reshape(-1)
        xs_interpolated = np.append(xs_interpolated, [interpolation], axis=0)
        interpolation = griddata((x_coors, y_coors), ys[each], (grid_x, grid_y), method=method).reshape(-1)
        ys_interpolated = np.append(ys_interpolated, [interpolation], axis=0)
    for each in range(len(xs), len(zs)):
        interpolation = griddata((x_coors, y_coors), zs[each], (grid_x, grid_y), method=method).reshape(-1)
        zs_interpolated = np.append(zs_interpolated, [interpolation], axis=0)
    print("Interpolation complete")
    return xs_interpolated, ys_interpolated, zs_interpolated


def interpolate_one_vector(x_coors, y_coors, values, grid_x, grid_y, cut=170, method='nearest'):
    print("Started interpolation")
    zs_interpolated = np.zeros((0, grid_x.shape[0] * grid_x.shape[1]))
    for each in range(len(values)):

        #  mirroring option
        #interpolation = np.flip(griddata((x_coors, y_coors), values[each], (grid_x, grid_y), method=method), axis=(0,1)).reshape(-1)
        interpolation = np.flip(griddata((x_coors, y_coors), values[each], (grid_x, grid_y), method=method), axis=0).reshape(-1)
        zs_interpolated = np.append(zs_interpolated, [interpolation], axis=0)
    print("Interpolation complete")
    return zs_interpolated


def stack_vectors(xs_interpolated, ys_interpolated, zs_interpolated, grid_x, cut=170):
    zs_interpolated = zs_interpolated[:cut]
    phi = np.zeros((len(zs_interpolated), grid_x.shape[0] * grid_x.shape[1] * 3))
    print("Started stacking")
    for idx, each in enumerate(zs_interpolated):
        i = 0
        for every in range(len(each)):
            phi[idx][i] = zs_interpolated[idx][every]
            phi[idx][i + 1] = xs_interpolated[idx][every]
            phi[idx][i + 2] = ys_interpolated[idx][every]
            i += 3
    print("Stacking complete")
    return phi


def interpolate_and_stack_vectors(x_coors, y_coors, xs, ys, zs, grid_x, grid_y, cut=170, method='nearest'):
    xs = xs[:cut]
    ys = ys[:cut]
    zs = zs[:cut]
    print("Started interpolation and stacking")
    xs_interpolated = np.zeros((0, grid_x.shape[0] * grid_x.shape[1]))
    ys_interpolated = np.zeros((0, grid_x.shape[0] * grid_x.shape[1]))
    zs_interpolated = np.zeros((0, grid_x.shape[0] * grid_x.shape[1]))
    for each in range(len(zs)):
        interpolation = griddata((x_coors, y_coors), zs[each], (grid_x, grid_y), method=method).reshape(-1)
        zs_interpolated = np.append(zs_interpolated, [interpolation], axis=0)
        interpolation = griddata((x_coors, y_coors), xs[each], (grid_x, grid_y), method=method).reshape(-1)
        xs_interpolated = np.append(xs_interpolated, [interpolation], axis=0)
        interpolation = griddata((x_coors, y_coors), ys[each], (grid_x, grid_y), method=method).reshape(-1)
        ys_interpolated = np.append(ys_interpolated, [interpolation], axis=0)
    print("Interpolation complete")
    phi = np.zeros((len(zs_interpolated), grid_x.shape[0] * grid_x.shape[1] * 3))

    for idx, each in enumerate(zs_interpolated):
        i = 0
        for every in range(len(each)):
            phi[idx][i] = zs_interpolated[idx][every]
            phi[idx][i + 1] = xs_interpolated[idx][every]
            phi[idx][i + 2] = ys_interpolated[idx][every]
            i += 3
    print("Stacking complete")
    return phi


def mlp_predict(mac_vector):
    if mac_vector.shape != (28900,):
        print("MAC vector is not of shape equal to MLP input dimensions")
        return ValueError
    json_file = open(os.path.join(os.getcwd(), 'saved_models/MLPv4_all_dropouts11052020_193914.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(os.path.join(os.getcwd(), 'saved_models/MLPv4_all_dropouts11052020_193914.h5'))
    sgd1 = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    loaded_model.compile(loss='categorical_crossentropy',
                         optimizer=sgd1,
                         metrics=['accuracy'])
    classes = np.load('CFRP_Dataset/delam/class_names_all.npy')
    mac_vector = mac_vector.reshape(1, 28900)
    prediction = loaded_model.predict(mac_vector)
    defects = {}
    for idx, each in enumerate(prediction[0]):
        if each > 0.35:
            name = classes[idx].split('r')
            defects[name[0]+'r_'+name[1]] = each
    if not defects:
        for index, each in enumerate(prediction[0]):
            if each == max(prediction[0]):
                break
        name = classes[index].split('r')
        defects[name[0] + 'r_' + name[1]] = each
        prediction[0][index] = 0
        for index, each in enumerate(prediction[0]):
            if each == max(prediction[0]):
                break
        print(classes[index], each)
    return defects


def read_dbr(file):
    # print("Started the parsing of DBR file: ", file)
    # print("Processing...")
    f = open(file, 'r')
    lines = f.readlines()
    dbrs = np.empty(0, dtype=float)
    freqs = np.empty(0, dtype=float)
    i = 1
    line = lines[i]
    while 'F:' not in line:
        dbrs = np.append(dbrs, float(line))
        i += 1
        line = lines[i]
    for line in lines[i+1:]:
        freqs = np.append(freqs, float(line))
    freqs = freqs*1000
    # print("DBR file processed.")
    return dbrs, freqs


def plot_LDRs(defects, zs, freqs, grid_x, custom=None):
    ldr_pictures = []
    detected_defects = []
    if not os.path.isdir('temp_images'):
        os.mkdir('temp_images')
    for defect_index, defect in enumerate(defects.keys()):
        detected_defects.append(defect)
        if not os.path.isdir(f'temp_images/{defect}'):
            os.mkdir(f'temp_images/{defect}')
    if custom:
        indexes = np.empty((0), dtype=int)
        custom_freqs = np.load(custom)
        for each in custom_freqs:
            if each >= max(freqs):
                i = len(freqs)-1
            else:
                i = np.searchsorted(freqs, each)
            if freqs[i] - each < each - freqs[i - 1]:
                if i <= len(zs):
                    indexes = np.append(indexes, i)
                elif i - 1 <= len(zs) < i:
                    indexes = np.append(indexes, i - 1)
            else:
                if i - 1 <= len(zs):
                    indexes = np.append(indexes, i - 1)
        indexes = np.unique(indexes)
    else:
            dbrs, freqs_d = read_dbr(f'CFRP_dataset/dbrs/DBR_CFRP_T300_fbh_circle_{defect}_c(50_75_0.0).txt')
            for idx, each in enumerate(freqs_d):
                if each < max(freqs):
                    pass
                else:
                    break
            freqs_d = freqs_d[:idx]
            dbrs = dbrs[:idx]
            indexes = np.empty((0), dtype=int)
            threshold = 0.5 * max(dbrs)
            for idx, each in enumerate(dbrs):
                if each > threshold:
                    i = np.searchsorted(freqs, freqs_d[idx])
                    if freqs[i] - freqs_d[idx] < freqs_d[idx] - freqs[i - 1]:
                        if i <= len(zs):
                            indexes = np.append(indexes, i)
                        elif i-1 <= len(zs) < i:
                            indexes = np.append(indexes, i - 1)
                    else:
                        if i-1 <= len(zs):
                            indexes = np.append(indexes, i - 1)
            indexes = np.unique(indexes)
    fig = plt.figure(frameon=False)
    plt.subplots_adjust(0, 0, 1, 1, 0, 0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    for index in indexes:
        plt.imshow(zs[index].reshape(grid_x.shape), cmap="binary")
        name = f'temp_images/{defect}/'+str(defect_index)+'_'+str(int(freqs[index])) + ".jpg"
        fig.savefig(name, pad_inches=0, bbox_inches='tight', dpi=100)
        ldr_pictures.append(name)
    zs_export = np.empty((0, grid_x.shape[0], grid_x.shape[1]))
    for zs in zs[indexes]:
        zs_export = np.append(zs_export, [zs.reshape(grid_x.shape)], axis=0)
    return ldr_pictures, zs_export


def mrcnn_predict(ldr_pictures):
    predicted_images = []
    ROOT_DIR = r"D:\Thesis"
    import mrcnn.model as modellib
    from mrcnn import visualize
    import fbh
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    FBH_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_fbh_0019.h5")

    class InferenceConfig(fbh.FBHConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        DETECTION_MIN_CONFIDENCE = 0.8
    config = InferenceConfig()
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    model.load_weights(FBH_MODEL_PATH, by_name=True)
    class_names = ['BG', 'LDR']
    file_names = ldr_pictures
    results_cache = []
    for file_name in file_names:
        image = skimage.io.imread(file_name)
        results = model.detect([image], verbose=1)
        r = results[0]
        results_cache.append(r)
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                    class_names, r['scores'], ax=plt.gca(), file_name=file_name)
        predicted_images.append('temp_images/prediction_results/' + "predicted_" + file_name.rsplit('/', 1)[1])
    return predicted_images, results_cache


def mrcnn_prediction_procedure(defects, zs, freqs, grid_x, custom=None, all_pics=False):
    if custom:
        defects = custom
    if all_pics:
        ldr_pictures, zs_ldr = plot_all_modeshapes(defects, zs, freqs, grid_x)
    else:
        ldr_pictures, zs_ldr = plot_LDRs(defects, zs, freqs, grid_x)
        #ldr_pictures, zs_ldr = plot_LDRs(defects, zs, freqs, grid_x, custom='LDRS_paper.npy')  # Custom LDR freqs option
    predicted_images, results_cache = mrcnn_predict(ldr_pictures)
    number = 0
    defects = []
    masks = []
    while True:
        number += 1
        print('Iter', number)
        iter_images = []
        idx_to_delete = []
        for idx, result in enumerate(results_cache):
            if len(result['rois']) != 0:
                cut_image_name, zs_ldr[idx] = cut_polygon(zs_ldr[idx], ldr_pictures[idx], result['rois'], number)
                iter_images.append(cut_image_name)
                defects.append(result['rois'])
                masks.append(result['masks'])
            else:
                idx_to_delete.append(idx)
        zs_ldr = np.delete(zs_ldr, idx_to_delete, 0)
        for each in reversed(idx_to_delete):  # it was commented
            del ldr_pictures[each]  # it also
        predicted_images, results_cache = mrcnn_predict(iter_images)
        if len(results_cache) == 0:
            break
    heatmap, heatmap_b, heatmask, heatmask_b = infer_mrcnn_predictions(defects, masks, grid_x.shape)
    return defects, masks, heatmap, heatmask, heatmap_b, heatmask_b


def cut_polygon(zs, image, polygons, number):
    image_orig = skimage.io.imread(image)
    color = np.mean(zs)
    zs_new = zs
    xscale = image_orig.shape[0] / zs_new.shape[0]
    yscale = image_orig.shape[1] / zs_new.shape[1]
    for polygon in polygons:
        y1, x1, y2, x2 = polygon
        y1 = int(y1 / yscale)
        y2 = int(y2 / yscale)
        x1 = int(x1 / xscale)
        x2 = int(x2 / xscale)
        zs_new[y1:y2, x1:x2] = color
    fig = plt.gcf()
    ax = plt.gca()
    ax.axis('off')
    name = 'temp_images/prediction_results/iter_'+str(number)
    if not os.path.isdir(name):
        os.mkdir(name)
    name = name+'/'+image.rsplit('.', 1)[0].rsplit('/', 1)[1]+'_'+str(number)+'cut.jpg'
    plt.imshow(zs_new, cmap='binary')
    fig.savefig(name, pad_inches=0, bbox_inches='tight', dpi=100)
    return name, zs_new


def infer_mrcnn_predictions(defects, masks, original_shape):
    pic_path = r'D:\Thesis\temp_images\prediction_results'
    files = os.listdir(r'D:\Thesis\temp_images\prediction_results')
    i = 0
    while True:
        if '.jpg' in files[i]:
            break
        i += 1
    image = skimage.io.imread(os.path.join(pic_path, files[i]))
    flat_defects = []
    for each in defects:
        for every in each:
            flat_defects.append(every)

    heatmap = np.zeros(original_shape)
    xscale = image.shape[0] / original_shape[0]
    yscale = image.shape[1] / original_shape[1]
    for each in flat_defects:
        y1, x1, y2, x2 = each
        y1 = int(y1 / yscale)
        y2 = int(y2 / yscale)
        x1 = int(x1 / xscale)
        x2 = int(x2 / xscale)
        heatmap[y1:y2, x1:x2] += 1
    plt.close()
    fig = plt.figure(figsize=(8, 5), frameon=False)
    plt.imshow((heatmap/len(flat_defects))*100, cmap='jet')
    cbar = plt.colorbar()
    cbar.set_label("Percentage of region detections", fontsize=18)
    plt.xlabel('X coordinates of the specimen', fontsize=18)
    plt.ylabel('Y coordinates of the specimen', fontsize=18)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.title("Region-based defect heatmap", fontsize=20)
    plt.savefig(r'D:\Thesis\temp_images\prediction_results\heatmap.jpg')
    maximum = np.max(heatmap)
    heatmap_b = np.zeros_like(heatmap)
    for i, each in enumerate(heatmap):
        for j, every in enumerate(each):
            if every >= maximum * 0.06:
                heatmap_b[i][j] = 1
            else:
                heatmap_b[i][j] = 0
    plt.clf()
    plt.imshow(heatmap_b, cmap='binary')
    cbar = plt.colorbar()
    cbar.set_label("Defect presence", fontsize=18)
    plt.xlabel('X coordinates of the specimen', fontsize=18)
    plt.ylabel('Y coordinates of the specimen', fontsize=18)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.title("Region-based binarized defect heatmap", fontsize=20)
    plt.savefig(r'D:\Thesis\temp_images\prediction_results\heatmap_binary.jpg')
    heatmask = np.zeros(masks[0].shape[:2])
    for each in masks:
        for x in range(each.shape[0]):
            for y in range(each.shape[1]):
                for z in range(each.shape[2]):
                    if each[x][y][z]:
                        heatmask[x][y] += int(each[x][y][z])
    plt.clf()
    plt.imshow((heatmask/len(flat_defects))*100, cmap='jet')
    cbar = plt.colorbar()
    cbar.set_label("Percentage of region detections", fontsize=18)
    plt.xlabel('X coordinates of the picture', fontsize=18)
    plt.ylabel('Y coordinates of the picture', fontsize=18)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.title("Mask-based defect heatmap", fontsize=20)
    plt.savefig(r'D:\Thesis\temp_images\prediction_results\heatmask.jpg')

    maximum = np.max(heatmask)
    heatmask_b = np.zeros(original_shape)
    print(original_shape)
    for i, each in enumerate(heatmask):
        for j, every in enumerate(each):
            if every >= maximum * 0.06:
                heatmask_b[int(i / xscale)][int(j / yscale)] = 1
            else:
                try:
                    heatmask_b[int(i / xscale)][int(j / yscale)] = 0
                except:
                    print(i, j, i/xscale, j/yscale, int(i/xscale), int(j/yscale))
    plt.clf()
    plt.imshow(heatmask_b, cmap='binary')
    cbar = plt.colorbar()
    cbar.set_label("Defect presence", fontsize=18)
    plt.xlabel('X coordinates of the specimen', fontsize=18)
    plt.ylabel('Y coordinates of the specimen', fontsize=18)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.title("Mask-based binarized defect heatmap", fontsize=20)
    plt.savefig(r'D:\Thesis\temp_images\prediction_results\heatmask_binary.jpg')
    plt.close()
    return heatmap, heatmap_b, heatmask, heatmask_b


def plot_all_modeshapes(defects, zs, freqs, grid_x):
    ldr_pictures = []
    if not os.path.isdir('temp_images'):
        os.mkdir('temp_images')
    indexes = range(len(zs))
    fig = plt.figure(frameon=False)
    plt.subplots_adjust(0, 0, 1, 1, 0, 0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    for index in indexes:
        plt.imshow(zs[index].reshape(grid_x.shape), cmap="binary")
        name = 'temp_images/all/'+str(int(freqs[index])) + ".jpg"
        fig.savefig(name, pad_inches=0, bbox_inches='tight', dpi=100)
        ldr_pictures.append(name)
    zs_export = np.empty((0, grid_x.shape[0], grid_x.shape[1]))
    for zs in zs[indexes]:
        zs_export = np.append(zs_export, [zs.reshape(grid_x.shape)], axis=0)
    return ldr_pictures, zs_export
