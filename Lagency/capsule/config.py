epochs = 50
learning_rate = 0.001
batch_size = 100
log_step = 20
save_step = 10
output_dir = './outputs_datasetx'

route_iterations = 3
sigma = 2

device_ids = [0, 1, 2, 3]
device = 'cuda:{}'.format(device_ids[0])
