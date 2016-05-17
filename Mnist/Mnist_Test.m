%Get Datas
load Mnist

% Mnist.Train_X = 60000*784, Train_X = 28 * 28 * 60000
Train_X = reshape(Mnist.Train_X',28,28,60000);
% Mnist.Train_Y = 60000*10, Train_Y = 10 * 60000
Train_Y = Mnist.Train_Y';

% Mnist.Train_X = 10000*784, Train_X = 28 * 28 * 10000
Test_X = reshape(Mnist.Test_X',28,28,10000);
% Mnist.Train_Y = 10000*10, Train_Y = 10 * 10000
Test_Y = Mnist.Test_Y';

Train_X_Size = size(Train_X);
Train_Y_Size = size(Train_Y);

opts.alpha = 1;
opts.batchsize = 50;
opts.numepochs = 1;

Net = Mnist_CNN_Init(Train_X_Size,Train_Y_Size);
Net = CNN_Train(Net, Train_X, Train_Y, opts);

[bar,er] = CNN_Test(Net,Test_X, Test_Y);

% save Net;
% save bad;
% save er;


