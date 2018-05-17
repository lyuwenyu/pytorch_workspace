import torch
import utils
import copy 

class Solver(object):


    def run(self, train_data_loader=None, test_data_loader=None):

        swa_model = None
        swa_n = 0
        for i in range(0, epoches):

            scheduler.step()
            train(model, train_data_loader)
            test(model, test_data_loader)


            if (i+1) >= swa_start and (i+1-swa_start) % swa_c_epochs == 0:
                
                if swa_params is None:
                    swa_model = copy.deepcopy(model)
                    continue

                utils.moving_average(swa_model, model, 1./(swa_n+1))
                test(swa_model, test_data_loader)
                swa_n += 1
