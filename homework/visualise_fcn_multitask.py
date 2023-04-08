from .utils import load_dense_data, DenseVisualization

test_data_vis = load_dense_data('drive-download-20230401T115945Z-001/test', 2, batch_size=6)
inputs, labels, depth = list(test_data_vis)[0]

DenseVisualization(inputs, depth, labels).__visualizeitem__()