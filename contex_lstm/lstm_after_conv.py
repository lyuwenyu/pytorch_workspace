import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models



class lstm_layer(nn.Module):
    

    def __init__(self, c):

        super(lstm_layer, self).__init__()
        
        self.lstm1 = nn.LSTM(c, c//4, 2, bidirectional=True)
        
        self.lstm2 = nn.LSTM(c, c//4, 2, bidirectional=True)
        
        
    def forward(self, x):
            
        n, c, h, w = x.size() 
        x1 = x.permute(3, 0, 2, 1).contiguous()
        x1, _ = self.lstm1(x1.view(w, -1, c))
        x1 = x1.view(w, n, h, -1).permute(1, 3, 2, 0)
        
        x2 = x.permute(2, 0, 3, 1).contiguous()
        x2, _ = self.lstm2(x2.view(h, -1, c))
        x2 = x2.view(h, n, w, -1).permute(1, 3, 0, 2)
        
        x = torch.cat([x1, x2], dim=1)

        return x



class resnet_lstm_contex(nn.Module):

    
    def __init__(self, num_classes, pretrained=True):
        
        super(resnet_lstm_contex, self).__init__()
        
        self.model = models.resnet101() 
        if pretrained:
            self.model.load_state_dict( torch.load('/home/lvwenyu/.torch/models/resnet101-5d3b4d8f.pth'), strict=False)
        
    
        # self.module_list = nn.ModuleList( list(model.children())[:-1] )
        # self.module_list.append( nn.Linear(2048, num_classes) )
        
        self.fc = nn.Linear(2048, num_classes)
        
        self.lstm_layer = nn.ModuleList( [lstm_layer(c) for c in [256, 512, 1024, 2048] ] )
        
        
        
    def forward(self, x):
        
        for i, (n, m) in enumerate( self.model.named_children() ):
            
            if 'layer' in n:
                
                x = m(x)
                x = self.lstm_layer[i-4](x)
                
            elif 'fc' == n :
                
                x = m( x.view(-1, 2048))
                
            else:
                
                x = m(x)
                
        return x
        
    
    
   
if __name__ == '__main__':

	mm = resnet_lstm_contex(num_classes=10, pretrained=False) 

	nn = lstm_layer(256)

	x = Variable( torch.randn(10, 3, 224, 224) )
	xx = Variable( torch.randn(10, 256, 56, 56) )

	print(nn(xx).size())


	print(mm)
	print(mm(x).size())
	
	
