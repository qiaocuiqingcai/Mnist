
#1.导入库
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms
#测试玩玩
# x=torch.rand(3)
# print(x)
# print(torch.is_tensor(x))
#测试玩玩
#测试玩玩
#测试玩玩
#=测试CPU速度
# a=torch.randn(10000,1000)
# b=torch.randn(1000,2000)
#
# t0=time.time()
# c=torch.matmul(a,b)
# t1=time.time()
# print(a.device,t1-t0,c.norm(2))
#自动求导
#梯度算法求二元一次方程的系数
#2.定义超参数，优化和调整的参数
BATCH_SIZE=64 #每次处理的数据数量
DEVICE=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

EPOCHS=10 #训练数据集的轮次
##3.构建pipline，对图像做处理
pipeline =transforms.Compose([
    transforms.ToTensor(),#将图片转转成tensor
     transforms.Normalize((0.1307,),(0.3081,))#正则化，预防过拟合，降低模型复杂度0.2原来是0.3
])
#4.下载/加载数据
from torch.utils.data import DataLoader
train_set=datasets.MNIST("data",train=True,download=True,transform=pipeline,)
test_set=datasets.MNIST("data",train=False,download=True,transform=pipeline,)
#加载数据
train_loader=DataLoader(train_set,batch_size=BATCH_SIZE,shuffle=True)
test_loader=DataLoader(test_set,batch_size=BATCH_SIZE,shuffle=True)
#查看下图片-----可不运行
#5.构建网络模型
class Digit(nn.Module):
    def __init__(self):
        super().__init__()
        #卷积层图片大小28*28
        self.conv1 = nn.Conv2d(1,64,1)#1:输入通道/每个卷积核通道数 10：输出通道/卷积核个数 5：卷积核尺寸是5*5
        self.conv2 = nn.Conv2d(64,128,1)#10:每个卷积核通道数 20：卷积核个数 20：卷积核kernel是3*3
        self.conv3 = nn.Conv2d(128, 128, 1)
        self.conv4 = nn.Conv2d(128, 256, 7)
        self.conv5 = nn.Conv2d(256,512, 5)
        self.conv6 = nn.Conv2d(512, 1024, 3)
        self.conv7 = nn.Conv2d(1024, 2000, 2)
        #全连接层
        self.fc1 = nn.Linear(2000,400)#20*10*10输入通道，原500：输出通道
        self.fc2 = nn.Linear(400,100)
         # self.fc3 = nn.Linear(100,10)
    def forward(self,x):
        input_size=x.size(0)#batch_size*1*28*28,只取batch——size
        #卷积层1
        x=self.conv1(x) #输入：batch_size*1*28*28，#输出batch_size*10*24*24(28-5+1=24)
        #激活函数
        x=F.relu(x)#保持shape不变，输出batch*10*24*24
        #池化层：（最大池化）是不会被反向传播参数的。通常在卷积后，步长通常是池化层大小

        # 卷积层2
        x = self.conv2(x)  # 输入：batch_size*10*12*12，#输出batch_size*20*10*10(12-3+1=10)
        # 激活函数
        x = F.relu(x)  # 保持shape不变
        x = F.max_pool2d(x, 2, 2)  # 输出batch*10*12*12压缩可能可以减少过拟合），选取最有特征的信息。
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = F.relu(x)
        x = self.conv7(x)
        x = F.relu(x)

        #降维层
        x=x.view(input_size,-1)#拉平，-1 自动计算维度，20*10*10=2000.输出batch*2000
        # 全连接层1
        x=self.fc1(x)#输入：batch*2000 输出batch*500
        # 激活函数
        x = F.relu(x)  # 保持shape不变
        # 全连接层2
        x = self.fc2(x)  # 输入：batch*500 输出batch*10
        # 激活函数
        # x = F.relu(x)  # 保持shape不变
        # 全连接层3
        # x = self.fc3(x)  # 输入：batch*500 输出batch*10
        output = F.log_softmax(x,dim=1)#计算分类后，每个数字的概率值

        return output
#6.定义优化器
model=Digit().to(DEVICE)#学习率太大，极度影响准确性，太小也会影响达到最大准确率所需的EPOCH
optimizer=optim.Adam(model.parameters(),)#优化器可能是原因,为不同的参数计算不同的自适应学习率可能0.001
#7.定义训练方法
def train_model(model,device,train_loader,optimizer,epoch):
    model.train()
    for batch_index,(data,target) in enumerate (train_loader):
        #部署到DEVICE
        data,target=data.to(device),target.to(device)
        #梯度初始化为0
        optimizer.zero_grad()
        #训练后的结果
        output=model(data)
        #计算损失
        loss = F.cross_entropy(output,target)#交叉商损失，使用多分类损失函数。二分类一般用sigmoid
        #找到概率值最大的下标
        pred=output.argmax(dim=1)#0是值，1是索引。等价pred=output.argmax(dim=1)
        #反向传播！！！！必须有！
        loss.backward()
        #参数优化
        optimizer.step()
        if batch_index % 80==0:
            print("Train Epoch :{}\t Loss:{:.6f}".format(epoch,loss.item()))
#定义测试方法
def test_model(model,device,test_loader):
    #模型验证
    model.eval()
    #正确率
    correct=0.0
    #测试损失
    test_loss=0.0
    with torch.no_grad():#不会计算梯度和反向传播
        for data ,target in test_loader:
            # 部署到DEVICE
            data, target = data.to(device), target.to(device)
            #测试数据
            output = model(data)
            # 计算损失
            test_loss += F.cross_entropy(output, target)
            # 找到概率值最大的下标
            pred=output.argmax(dim=1)
            #累计正确率
            correct+=pred.eq(target.view_as(pred)).sum().item()
        test_loss/=len(test_loader.dataset)
        print("Test---Average loss:{:.6f},Accuracy:{:.6f}\n".format(test_loss,100.0*correct/len(test_loader.dataset)))
# #调用方法
for epoch in range(1,EPOCHS+1):
    train_model(model,DEVICE,train_loader,optimizer,epoch)
    test_model(model,DEVICE,test_loader)

