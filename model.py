import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
import torch.nn.functional as F
#from cc_attention import CrissCrossAttention #for Criss-Cross attention

######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

def weights_init_new(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)
    
# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, dropout=True, relu=True, num_bottleneck=512):
        super(ClassBlock, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)] 
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
            #add_block += [nn.PReLU()]
        if dropout:
            add_block += [nn.Dropout(p=0.5)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        self.add_block = add_block
        
        
        ##### original code, if use cosmargin, comment the below lines ####
        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)
        self.classifier = classifier
        
    def forward(self, x):
        x = self.add_block(x)
        ##### original code, if use cosmargin, comment the below lines ####
        x = self.classifier(x)
        
        return x

# Define the ResNet50-based Model
class ft_net(nn.Module):

    def __init__(self, class_num ):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.classifier = ClassBlock(2048, class_num, dropout = False)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Define the DenseNet121-based Model
class ft_net_dense(nn.Module):

    def __init__(self, class_num ):
        super().__init__()
        model_ft = models.densenet121(pretrained=True)
        model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.fc = nn.Sequential()
        self.model = model_ft
        # For DenseNet, the feature dim is 1024 
        self.classifier = ClassBlock(1024, class_num)

    def forward(self, x):
        x = self.model.features(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x
    
# Define the ResNet50-based Model (Middle-Concat)
# In the spirit of "The Devil is in the Middle: Exploiting Mid-level Representations for Cross-Domain Instance Matching." Yu, Qian, et al. arXiv:1711.08106 (2017).
class ft_net_middle(nn.Module):

    def __init__(self, class_num ):
        super(ft_net_middle, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.classifier = ClassBlock(2048+1024, class_num)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        # x0  n*1024*1*1
        x0 = self.model.avgpool(x)
        x = self.model.layer4(x)
        # x1  n*2048*1*1
        x1 = self.model.avgpool(x)
        x = torch.cat((x0,x1),1)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x

# Part Model proposed in Yifan Sun etal. (2018)
class PCB(nn.Module):
    def __init__(self, class_num, part = 6):
        super(PCB, self).__init__()

        self.part = part # We cut the pool5 to 6 parts
        model_ft = models.resnet50(pretrained=True)
        self.model = model_ft
        self.avgpool = nn.AdaptiveAvgPool2d((self.part,1))
        self.dropout = nn.Dropout(p=0)
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)
        # define 6 classifiers
        for i in range(self.part):
            name = 'classifier'+str(i)
            setattr(self, name, ClassBlock(2048, class_num, False, False, 256))

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        part = {}
        predict = {}
        # get six part feature batchsize*2048*6
        for i in range(self.part):
            part[i] = torch.squeeze(x[:,:,i])
            name = 'classifier'+str(i)
            c = getattr(self,name)
            predict[i] = c(part[i])

        # sum prediction
        #y = predict[0]
        #for i in range(self.part-1):
        #    y += predict[i+1]
        y = []
        for i in range(self.part):
            y.append(predict[i])
        return y

class PCB_test(nn.Module):
    def __init__(self,model,part=6):
        super(PCB_test,self).__init__()
        self.part = part
        self.model = model.model
        self.avgpool = nn.AdaptiveAvgPool2d((self.part,1))
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        y = x.view(x.size(0),x.size(1),-1)
        return y



##### confusion net ###########
class confusion_net(nn.Module):

    def __init__(self, class_num):
        super(confusion_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        
        fc = nn.Sequential(nn.Linear(2048, 512))
        fc.apply(weights_init_kaiming)
        self.fc = fc
        
        bn = nn.Sequential(nn.BatchNorm1d(512))
        bn.apply(weights_init_kaiming)
        self.bn = bn
        
        classifier = nn.Sequential(nn.Linear(512, class_num))
        classifier.apply(weights_init_classifier)
        self.classifier = classifier

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)
        y = self.fc(x)
        y = self.bn(y)
        y = self.classifier(y)
        y_confuse = self.fc(x.detach())
        return y, y_confuse
        
##### BIER net ###########
class BIER_net(nn.Module):

    def __init__(self, class_num):
        super(BIER_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        
        fc = nn.Sequential(nn.Linear(2048, 512))
        fc.apply(weights_init_kaiming)
        self.fc = fc
        
        bn = nn.Sequential(nn.BatchNorm1d(512))
        bn.apply(weights_init_kaiming)
        self.bn = bn


    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)
        y = self.fc(x)
        y = self.bn(y)
        #y_confuse = self.fc(x.detach())
        return y
        
##### Recurrent Cross-Criss model #############
#class RCCAModule(nn.Module):
#    def __init__(self, in_channels, out_channels):
#        super(RCCAModule, self).__init__()
#        inter_channels = in_channels // 4
#        self.conva = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
#                                   nn.BatchNorm2d(inter_channels),
#                                   nn.ReLU(inplace=True))
#        self.cca = CrissCrossAttention(inter_channels)
#        self.convb = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
#                                   nn.BatchNorm2d(inter_channels),
#                                   nn.ReLU(inplace=True))
#        self.bottleneck = nn.Sequential(
#            nn.Conv2d(in_channels+inter_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False),
#            nn.BatchNorm2d(out_channels),
#            nn.ReLU(inplace=True)
#            )
#
#    def forward(self, x, recurrence=1):
#        output = self.conva(x)
#        for i in range(recurrence):
#            output = self.cca(output)
#        output = self.convb(output)
#
#        output = self.bottleneck(torch.cat([x, output], 1))
#        return output
#        
#class RCCA_net(nn.Module):
#
#    def __init__(self, class_num):
#        super(RCCA_net, self).__init__()
#        model_ft = models.resnet50(pretrained=True)
#        # avg pooling to global pooling
#        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
#        self.model = model_ft
#        
#        self.head = RCCAModule(2048,2048)
#        
#        self.classifier = ClassBlock(2048, class_num)
#
#
#    def forward(self, x, recurrence = 1):
#        x = self.model.conv1(x)
#        x = self.model.bn1(x)
#        x = self.model.relu(x)
#        x = self.model.maxpool(x)
#        x = self.model.layer1(x)
#        x = self.model.layer2(x)
#        x = self.model.layer3(x)
#        x = self.model.layer4(x)
#        
#        x = self.head(x, recurrence)
#        
#        x = self.model.avgpool(x)
#
#        x = x.view(x.size(0), -1)
#        x = self.classifier(x)
#        return x
########## non-local ###################
class NonLocalModule(nn.Module):
    def __init__(self, in_channels, inter_channels, recurrence = 1):
        super(NonLocalModule, self).__init__()
        self.recurrence = recurrence
        for i in range(self.recurrence):
            name = 'conva_' + str(i)
            setattr(self, name, nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, padding=0, bias=False),
                                   )
                   )
        for i in range(self.recurrence):
            name = 'convb_' + str(i)
            setattr(self, name, nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, padding=0, bias=False),
                                   )
                   )
        for i in range(self.recurrence):
            name = 'convc_' + str(i)
            setattr(self, name, nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, padding=0, bias=False),
                                   )
                   )
        for i in range(self.recurrence):
            name = 'bottleneck_' + str(i)
            setattr(self, name, nn.Sequential(nn.Conv2d(inter_channels, in_channels, 1, bias=False),
                                        nn.BatchNorm2d(in_channels),
                                        nn.ReLU(inplace=True)
                                   )
                   )
    def forward(self, x):
        
        for i in range(self.recurrence):
            name = 'conva_' + str(i)
            layer_a = getattr(self, name)
            name = 'convb_' + str(i)
            layer_b = getattr(self, name)
            name = 'convc_' + str(i)
            layer_c = getattr(self, name)
            name = 'bottleneck_' + str(i)
            bottleneck = getattr(self, name)
            
            output_a = layer_a(x)
            output_b = layer_b(x)
            output_c = layer_c(x)
            
            out_a_b = torch.matmul(output_a.view(output_a.size(0), output_a.size(1), -1).transpose(1,2), output_b.view(output_b.size(0), output_b.size(1), -1))
            out = torch.matmul(output_c.view(output_c.size(0), output_c.size(1), -1), out_a_b)
            x = x + bottleneck(out.view(out.size(0), out.size(1) , x.size(2), x.size(3)))
        return x
        
class NonLocal_net(nn.Module):

    def __init__(self, class_num, recurrence = 1):
        super(NonLocal_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        self.recurrence = recurrence;
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)
        
        self.head1 = NonLocalModule(256, 96, self.recurrence)
        self.head2 = NonLocalModule(512, 192, self.recurrence)
        self.head3 = NonLocalModule(1024, 384, self.recurrence)
        self.head4 = NonLocalModule(2048, 768, self.recurrence)
        self.fc = nn.Sequential(nn.Linear(2048, 512), nn.BatchNorm1d(512), nn.LeakyReLU(0.1))
        self.fc.apply(weights_init_kaiming)
        self.classifier = nn.Sequential(nn.Linear(512, class_num))
        self.classifier.apply(weights_init_classifier)
        

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.head1(x)
        x = self.model.layer2(x)
        x = self.head2(x)
        x = self.model.layer3(x)
        x = self.head3(x)
        x = self.model.layer4(x)
        x = self.head4(x)
        
        x = self.model.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        y = self.classifier(x)
        #z = nn.functional.normalize(x)
        return y, x
##### diverse attention #######
##### IDE high div #####
class HighDivModule(nn.Module):
    def __init__(self, in_channels, order=1):
        super(HighDivModule, self).__init__()
        self.order = order
        self.inter_channels = in_channels // 8 * 2
        for j in range(self.order):
            for i in range(j+1):
                name = 'order' + str(self.order) + '_' + str(j+1) + '_' + str(i+1)
                setattr(self, name, nn.Sequential(nn.Conv2d(in_channels, self.inter_channels, 1, padding=0, bias=False))
            )
        for i in range(self.order):
            name = 'convb' + str(self.order) + '_' + str(i+1)
            setattr(self, name, nn.Sequential(nn.Conv2d(self.inter_channels, in_channels, 1, padding=0, bias=False),
                                              nn.Sigmoid()
                                   )
                                   )

    def forward(self, x):
        y=[]
        for j in range(self.order):
            for i in range(j+1):
                name = 'order' + str(self.order) + '_' + str(j+1) + '_' + str(i+1)
                layer = getattr(self, name)
                y.append(layer(x))
        y_ = []
        cnt=0
        for j in range(self.order):
            y_temp = 1
            for i in range(j+1):
                y_temp = y_temp * y[cnt]
                cnt += 1
            y_.append(F.relu(y_temp))
        
        #y_ = F.relu(y_)
        y__ = 0
        for i in range(self.order):
            name = 'convb' + str(self.order) + '_' + str(i+1)
            layer = getattr(self, name)
            y__ += layer(y_[i])
        out = x * y__ / self.order
        return out#, y__/ self.order
       
class MHN_IDE(nn.Module):

    def __init__(self, class_num, parts=4):
        super(MHN_IDE, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.parts = parts
        
        for i in range(self.parts):
            name = 'HIGH' + str(i)
            setattr(self, name, HighDivModule(512, i+1))
        
        self.fc = nn.Sequential(nn.Linear(2048, 256), nn.BatchNorm1d(256))
        self.fc.apply(weights_init_kaiming)
        
        for i in range(self.parts):
            name = 'classifier' + str(i)
            setattr(self, name, nn.Sequential(nn.Linear(256, class_num)))
        
        for i in range(self.parts):
            name = 'classifier' + str(i)
            layer = getattr(self, name)
            layer.apply(weights_init_classifier)


    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        
        x_=[]
        for i in range(self.parts):
            name = 'HIGH' + str(i)
            layer = getattr(self, name)
            x_.append(layer(x))
            
        x = torch.cat(x_ , 0)
        
        x = self.model.layer3(x)

        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)
        
        x = self.fc(x)
        y = []
        num = int(x.size(0)/self.parts)
        for i in range(self.parts):
            name = 'classifier' + str(i)
            layer = getattr(self, name)
            y.append(layer(x[i*num:(i+1)*num,:]))
        
        return y, x    
##### PCB high ######
class PCBHighDivModule(nn.Module):
    def __init__(self, in_channels, order=1):
        super(PCBHighDivModule, self).__init__()
        self.order = order
        self.inter_channels = in_channels // 8 * 2
        for j in range(self.order):
            for i in range(j+1):
                name = 'order' + str(self.order) + '_' + str(j+1) + '_' + str(i+1)
                setattr(self, name, nn.Sequential(nn.Conv2d(in_channels, self.inter_channels, 1, padding=0, bias=False))
            )
        for i in range(self.order):
            name = 'convb' + str(self.order) + '_' + str(i+1)
            setattr(self, name, nn.Sequential(nn.Conv2d(self.inter_channels, in_channels, 1, padding=0, bias=False),
                                              nn.Sigmoid()
                                   )
                                   )

    def forward(self, x):
        y=[]
        for j in range(self.order):
            for i in range(j+1):
                name = 'order' + str(self.order) + '_' + str(j+1) + '_' + str(i+1)
                layer = getattr(self, name)
                y.append(layer(x))
        y_ = []
        cnt=0
        for j in range(self.order):
            y_temp = 1
            for i in range(j+1):
                y_temp = y_temp * y[cnt]
                cnt += 1
            y_.append(F.relu(y_temp))
        
        #y_ = F.relu(y_)
        y__ = 0
        for i in range(self.order):
            name = 'convb' + str(self.order) + '_' + str(i+1)
            layer = getattr(self, name)
            y__ += layer(y_[i])
        out = x * y__ / self.order
        return out
class MHN_smallPCB(nn.Module):
    def __init__(self, class_num, parts=4, part = 6):
        super(MHN_smallPCB, self).__init__()

        self.part = part # We cut the pool5 to 6 part
        model_ft = models.resnet50(pretrained=True)
        self.model = model_ft
        self.avgpool = nn.AdaptiveAvgPool2d((self.part,1))
        self.dropout = nn.Dropout(p=0)
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)
        self.parts = parts
        
        for i in range(self.parts):
            name = 'HIGH' + str(i)
            setattr(self, name, PCBHighDivModule(512, i+1))
        
        # define 6 fc
        for i in range(self.part):
            name = 'fc' + str(i)
            setattr(self, name, nn.Sequential(nn.Linear(2048, 256), nn.BatchNorm1d(256)))
        for i in range(self.part):
            name = 'fc' + str(i)
            layer = getattr(self, name)
            layer.apply(weights_init_kaiming)
            
        # define 6*parts classifiers
        for i in range(self.part):
            for j in range(self.parts):
                name = 'classifier' + str(i) + '_' + str(j)
                setattr(self, name, nn.Sequential(nn.Linear(256, class_num)))
        for i in range(self.part):
            for j in range(self.parts):
                name = 'classifier' + str(i) + '_' + str(j)
                layer = getattr(self, name)
                layer.apply(weights_init_classifier)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)    
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        
        x_=[]
        for i in range(self.parts):
            name = 'HIGH' + str(i)
            layer = getattr(self, name)
            x_.append(layer(x))
            
        x = torch.cat(x_ , 0)
        
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        x = self.dropout(x)

        fea = []
        # get six part feature parts*batchsize,256,part
        for i in range(self.part):
            name = 'fc' + str(i)
            c = getattr(self,name)
            fea.append(F.normalize(c(torch.squeeze(x[:,:,i]))) * 20)#normalize features and scale to 20
            
        # get part*parts predict
        y=[]
        num = int(fea[0].size(0)/self.parts)
        for i in range(self.part):
            for j in range(self.parts):
                name = 'classifier' + str(i) + '_' + str(j)
                c = getattr(self, name)
                y.append(c(fea[i][j*num:(j+1)*num,:]))
        return y, fea
## debug model structure
#net = ft_net(751)
##net = ft_net_dense(751)
##print(net)
#input = Variable(torch.FloatTensor(8, 3, 224, 224))
#output = net(input)
#print('net output size:')
#print(output.shape)
