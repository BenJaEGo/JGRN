name = "freiburg"

raw_data_dir = r"/home/benjaego/datasets_reorganized/Freiburg"

raw_out_dir = r"/home/benjaego/seizure_prediction/"

selection = [0, 1, 2, 3, 4, 5]
bands = "combination" + "".join(str(id) for id in selection)

feature_length_sec = 10
feature_slide_sec = 1
feature_prefix = "sl{}_ss{}".format(feature_length_sec, feature_slide_sec)

psd_feature = "power_spectral_density"
psd_feature_dir = r"{}/FEATURES/{}/{}/{}/{}".format(raw_out_dir, name, bands, psd_feature, feature_prefix)

fs = 256
n_channel = 6
n_frequency = 114
n_class = 2
n_fold = 5
