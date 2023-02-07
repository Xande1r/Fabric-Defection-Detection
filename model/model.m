%%
layers = [
    %3层卷积层
    imageInputLayer([100 100 3])
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    
    dropoutLayer(0.2)%丢弃层
    
    fullyConnectedLayer(17)%分类数量
    softmaxLayer
    classificationLayer];
options = trainingOptions('sgdm',...
    'InitialLearnRate',0.005,...%学习率
    'MaxEpochs',6,... %训练轮数
    'ValidationData',validds,...
    'ValidationFrequency',30,...
    'Verbose',false,...
    'Plots','training-progress');
net = trainNetwork(trainds,layers,options);
%%