%% 数据预处理
%加载数据集
wenjian='data';
imds=imageDatastore(wenjian,...
    'includesubfolders',true,...
    'labelsource','foldernames');
%显示类别数量
tbl=countEachLabel(imds);
%图像的维度大小
tuxiang1=imread(imds.Files{1});
%为了使各类样本数量平衡选取数量最少的基准抽取样本
minSetCount = min(tbl{:,2});
imds = splitEachLabel(imds, minSetCount, 'randomize');

%图像预处理，将图像转换
imds.ReadFcn = @(filename)readAndPreprocessImage(filename);
%样本分割,随机抽取样本分割为7:3的训练集和验证集
[trainds,validds]=splitEachLabel(imds, 0.7, 'randomize');
%% 使用搭建的神经网络
layers = [
    %3层卷积层
    imageInputLayer([128 128 3])
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
%% 配置训练参数
options = trainingOptions('sgdm',...
    'InitialLearnRate',0.005,...%学习率
    'MaxEpochs',16,... %训练轮数
    'ValidationData',validds,...
    'ValidationFrequency',30,...
    'Verbose',false,...
    'Plots','training-progress');
%% 训练神经网络
net = trainNetwork(trainds,layers,options);
yp= classify(net,validds);
y=validds.Labels;
accurary= sum(yp==y)/numel(y);
%% 可视化验证 从验证集中随机抽9张图片与缺陷验证
perm=randperm(numel(y),9);
for i=1:9
    subplot(3,3,i);
    imshow(validds.Files{perm(i)});
    title(yp(perm(i)));
end
%% 预处理函数
function Iout = readAndPreprocessImage(filename)

I = imread(filename);

% 把灰度图像转换为RGB图像
if ismatrix(I)
    I = cat(3,I,I,I);
end

% 拉伸到100*100
Iout = imresize(I, [128 128]);
end 