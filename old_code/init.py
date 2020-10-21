import tools
file = 'CFRP_v1_fbh_circle_4r_80%d_c(20_50_0).txt'
exp_file = 'Modeset_new_sp14C.unv'
UGent_file = 'Mode shape .UNV files/all.unv'

def simulation_workflow(file):
    #mac_vector, zs, freqs, grid_x, grid_y = tools.calculate_mac_xyz(file)
    mac_vector, zs, freqs, grid_x, grid_y = tools.calculate_mac_xyz(file, use_cache=False, file_ref='Carbon_intact_100x150.txt')
    defects = tools.mlp_predict(mac_vector)
    print("Defects = ", defects)
    #custom = {'3r_80%d': 0.6572611}
    defects_mrcnn, masks_mrcnn, heatmap, heatmask, heatmap_b, heatmask_b = tools.mrcnn_prediction_procedure(
        defects, zs, freqs, grid_x)
    return defects, defects_mrcnn, masks_mrcnn, heatmap, heatmask, heatmap_b, heatmask_b


def test_workflow(file):
    mac_vector, zs, freqs, grid_x, grid_y = tools.calculate_mac_z(file)
    defects = tools.mlp_predict(mac_vector)
    print("Defects = ", defects)
    #custom = {'10r_80%d': 0.6572611}
    defects_mrcnn, masks_mrcnn, heatmap, heatmask, heatmap_b, heatmask_b = tools.mrcnn_prediction_procedure(
                                                                                    defects, zs, freqs, grid_x) #, custom=custom)
    return defects, defects_mrcnn, masks_mrcnn, heatmap, heatmask, heatmap_b, heatmask_b


def UGent(file):
    mac_vector, zs, freqs, grid_x, grid_y = tools.calculate_mac_z(file)
    custom = {'UGent': 0.99}
    defects = custom
    defects_mrcnn, masks_mrcnn, heatmap, heatmask, heatmap_b, heatmask_b = tools.mrcnn_prediction_procedure(
        defects, zs, freqs, grid_x, custom=custom)
    return defects, defects_mrcnn, masks_mrcnn, heatmap, heatmask, heatmap_b, heatmask_b


if __name__ == '__main__':
    # defects, ldr_pictures = simulation_workflow(file)
    defects, defects_mrcnn, masks_mrcnn, heatmap, heatmask, heatmap_b, heatmask_b = test_workflow(exp_file)
    print(defects, defects_mrcnn)
    print(masks_mrcnn)
