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




class Solver(object):


	def __init__(self, batch_size=10, use_cuda=True):

		## params
		self.use_cuda = use_cuda
		self.batch_size = batch_size


		## modules
		self.g = model_dcgan.generator()
		self.d = model_dcgan.discriminator()

		if use_cuda:
			self.g = self.g.cuda()
			self.d = self.d.cuda()


		## optimizer
		self.g_optim = optim.Adam(self.g.parameters(), lr=0.0002, betas=(0.5, 0.99))
		self.d_optim = optim.Adam(self.d.parameters(), lr=0.0002, betas=(0.5, 0.99))


		## criteria
		self.crit = nn.BCELoss()
		self.l1loss = nn.L1Loss()


		## scheduler
		self.scheduler_d = optim.lr_scheduler.StepLR(self.d_optim, step_size=10, gamma=0.1)
		self.scheduler_g = optim.lr_scheduler.StepLR(self.g_optim, step_size=10, gamma=0.1)
		self.schedulers = [self.scheduler_d, self.scheduler_g]

		print(self.g)
		print(self.d)


	def set_input(self, x, y, z):

		self.x = Variable(x)
		self.y = Variable(y)
		self.z = Variable(z)
		self.y_exp = Variable( torch.zeros(self.batch_size, 10, 28,28) ) + self.y

		if self.use_cuda:

			self.x = Variable(x.cuda())
			self.y = Variable(y.cuda())
			self.z = Variable(z.cuda())
			self.y_exp = Variable( torch.zeros(self.batch_size, 10, 28,28).cuda() ) + self.y

		# print('set input done...')



	def forward(self):

		self.fake = self.g(self.z, self.y)
		

	def backward_d(self):
		
		d_real = self.d(self.x, self.y_exp)
		d_fake = self.d(self.fake.detach(), self.y_exp)

		self.lossd = self.crit( d_real, Variable( d_real.data.new(*d_real.size()).fill_(1.).cuda() ) ) + self.crit( d_fake, Variable( d_real.data.new(*d_real.size()).fill_(0.).cuda() ) )

		self.lossd.backward()


	def backward_g(self):
		
		d_fake = self.d(self.fake, self.y_exp)

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


	def get_current_errors(self):

		errors = OrderedDict([('d_loss', self.lossd.cpu().data.numpy()[0]), ('g_loss', self.lossg.cpu().data.numpy()[0])])
		
		return errors


	def test(self, z, y):
		
		self.g.eval()
		
		return self.g(z, y)


	def save(self, output_dir = ''):
		pass


	def update_lr(self):
		
		self.scheduler_d.step()
		self.scheduler_g.step()

		print('d net lr: {}'.format(self.d_optim.param_groups[0]['lr']))
		print('g net lr: {}'.format(self.g_optim.param_groups[0]['lr']))




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
				


		solver.update_lr()


		