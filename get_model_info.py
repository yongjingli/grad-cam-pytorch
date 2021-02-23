import torch
pthfile = '/home/liyongjing/Egolee/programs/grad-cam-pytorch-master/models/epoch_100_export.pth'
net = torch.load(pthfile)
print(net.export)
with torch.no_grad():
    img = torch.ones((1, 3, 224, 224))
    y = net(img)
    print(y[0].shape)

# exit(1)
for name, paras, in net.named_modules():
    print(name) 
