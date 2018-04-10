import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from model import model_dcgan 
import torch.utils.data as data 
import numpy as np
from data.dataset import MNIST
import torchvision.utils as vutils
from collections import OrderedDict
import logging
from util.cfgs import opts

import os
from multiprocessing import Queue 

import copy
import pickle

try:
	from tensorboardX import SummaryWriter
except:
	raise RuntimeError('no tensorboardX')


class Solver(object):


	def __init__(self, batch_size=10, use_cuda=True, restore=False):

		## params
		self.use_cuda = use_cuda
		self.batch_size = batch_size


		self.state = copy.deepcopy(opts)

		# self.state = OrderedDict()
		# self.state['step'] = self.step
		# self.state['lr'] = self.lr


		## modules
		self.g = model_dcgan.generator()
		self.d = model_dcgan.discriminator()


		if restore:

			self.restore()


		if use_cuda:
			self.g = self.g.cuda()
			self.d = self.d.cuda()


		## optimizer
		self.g_optim = optim.Adam(self.g.parameters(), lr=self.state.lr, betas=(0.5, 0.99))
		self.d_optim = optim.Adam(self.d.parameters(), lr=self.state.lr, betas=(0.5, 0.99))


		## criteria
		self.crit = nn.BCELoss()
		self.l1loss = nn.L1Loss()


		## scheduler
		self.scheduler_d = optim.lr_scheduler.StepLR(self.d_optim, step_size=10, gamma=0.1)
		self.scheduler_g = optim.lr_scheduler.StepLR(self.g_optim, step_size=10, gamma=0.1)
		self.schedulers = [self.scheduler_d, self.scheduler_g]

		print(self.g)
		print(self.d)


		## buffer
		# self.buffer = Queue(20)
		

		## logging
		self.set_logger()



		## tensorboardX

		# self.set_summary()



	def set_summary(self, ):
		
		writer = SummaryWriter()
		try:
			dummy_data = Variable(torch.randn(1, 3, 224, 224))
			torch.onnx.export(self.model, dummy_data, 'model.proto', verbose=True)
			writer.add_graph_onnx('model.proto')

		except ImportError:
			pass

		self.writer = writer


	def set_logger(self, ):
		
		file_handler = logging.FileHandler( opts.log_file )
		fmat = logging.Formatter('%(name)s %(asctime)s %(levelname)-4s: %(message)s')
		file_handler.setFormatter(fmat)
		self.logger = logging.getLogger('solver')
		self.logger.addHandler(file_handler)
		self.logger.setLevel(logging.INFO)


	def set_input(self, x, y, z):

		self.x = Variable(x)
		self.y = Variable(y)
		self.z = Variable(z)
		self.y_exp = Variable( torch.zeros(self.batch_size, 10, 28,28) ) + self.y

		if self.state.use_cuda:

			self.x = Variable(x.cuda())
			self.y = Variable(y.cuda())
			self.z = Variable(z.cuda())
			self.y_exp = Variable( torch.zeros(self.batch_size, 10, 28,28).cuda() ) + self.y

		# print('set input done...')



	def forward(self):

		self.fake = self.g(self.z, self.y)
		# self.buffer.put([self.fake, self.y_exp])


	def backward_d(self):
		
		d_real = self.d(self.x, self.y_exp)
		d_fake = self.d(self.fake.detach(), self.y_exp)

		self.lossd = self.crit( d_real, Variable( d_real.data.new(*d_real.size()).fill_(1.).cuda() ) ) + self.crit( d_fake, Variable( d_real.data.new(*d_real.size()).fill_(0.).cuda() ) )

		self.lossd.backward()


	def backward_g(self):
		
		d_fake = self.d(self.fake, self.y_exp)
		# self.fake, self.y_exp = self.buffer.get()

		self.lossg = self.crit( d_fake, Variable( d_fake.data.new(*d_fake.size()).fill_(1.).cuda() ) ) + self.l1loss(self.fake, self.x)

		self.lossg.backward()


	def optimizer_step(self):

		self.g.train()
		self.d.train()

		self.forward()

		self.d_optim.zero_grad()
		self.backward_d()
		self.d_optim.step()

		self.g_optim.zero_grad()
		self.backward_g()
		self.g_optim.step()

		self.state.step += 1



	def get_current_errors(self):

		errors = OrderedDict([('d_loss', self.lossd.cpu().data.numpy()[0]), ('g_loss', self.lossg.cpu().data.numpy()[0])])
		
		self.logger.info(errors)

		return errors




	def test(self, z=None, y=None):
		
		def set_input(z, y):
			pass
			return z, y

		if (z is None) or (y is None):
						
			z = Variable( ((torch.randn(100, 100, 1, 1)-0.5)/0.5).cuda() )
			y = Variable( torch.cat([torch.eye(10,10)]*10, dim=0).view(100, 10, 1, 1).cuda())

		else:

			z, y = set_input(z, y)

		self.g.eval()
		
		return self.g(z, y)



	def update_lr(self):
		
		self.scheduler_d.step()
		self.scheduler_g.step()

		self.d_lr = self.d_optim.param_groups[0]['lr']
		self.g_lr = self.g_optim.param_groups[0]['lr']
		
		# self.lr = [self.d_lr, self.g_lr] 
		self.state.lr = self.d_lr

		print('d net lr: {}'.format(self.d_optim.param_groups[0]['lr']))
		print('g net lr: {}'.format(self.g_optim.param_groups[0]['lr']))



	def save(self, output_dir = '.', name='x'):
		
		# in case of OUT OF MEM
		torch.save( self.g.cpu().state_dict(), output_dir+'/g_{}.pt'.format(name) )
		self.g.cuda()

		with open(os.path.join(output_dir, 'state.pkl'), 'wb') as f:
			pickle.dump(self.state, f)

		print('save done...')



	def restore(self, dir_name='.'):

		## state
		with open(dir_name, 'rb') as f:

			self.state = pickle.load( f ) 

		## parameter
		torch.load('')



if __name__ == '__main__':

	batch_size = 50

	mnist = MNIST()
	dataloader = data.DataLoader(mnist, batch_size=batch_size, shuffle=True)

	solver = Solver(batch_size)


	for e in range(100):
		

		for i, (imgs, labs) in enumerate(dataloader):

			x = imgs.float().view(batch_size, 1, 28,28)
			y = labs.float().view(batch_size, 10, 1, 1)
			z = (torch.rand(batch_size, 100, 1, 1)-0.5)/0.5 

			solver.set_input(x, y, z)
			solver.optimizer_step()


			if i%20 == 0:

				print( solver.get_current_errors() )
				vutils.save_image( solver.fake.cpu().data , './output/gn_dcgan_{:0>3}.jpg'.format(e), nrow=10)
			
				vutils.save_image( solver.test().cpu().data , './output/gn_dcgan_{:0>3}_test.jpg'.format(e), nrow=10)

		solver.update_lr()

		solver.save()
		