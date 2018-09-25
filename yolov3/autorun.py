import os
import re
import time
import multiprocessing

f = re.compile('\d.*\d')

for lin in [
			'python train.py --logtostderr --train_dir=./outputs/train-v7/train --pipeline_config_path=./config/train-v7.config',\
			'python train.py --logtostderr --train_dir=./outputs/train-v8/train --pipeline_config_path=./config/train-v8.config'
		]:
	
	while True:
		
		res = os.popen('nvidia-smi --query-gpu=memory.free --format=csv').readlines()
		mem = int(f.findall(res[-1])[0])

		if mem > 7000:
			print('large than 7000.', mem)
			os.system('{}'.format(lin))
			break
		else:
			pass
			print(mem)
			
		time.sleep(600*6*1)

