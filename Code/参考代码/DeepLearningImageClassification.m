%% 检查系统配置
deviceInfo = gpuDevice;

computeCapability = str2double(deviceInfo.ComputeCapability);
assert(computeCapability > 3.0, ...
    'This example requires a GPU device with compute capability 3.0 or higher.')

%% 数据集
datasetsFolder = '../../datasets/caltech101'; % define output folder
url = 'http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz';
if ~exist(datasetsFolder, 'dir') % download only once
    disp('Downloading 126MB Caltech101 data set...');
    untar(url, datasetsFolder);
end
% 加载数据集
rootFolder = fullfile(datasetsFolder, '101_ObjectCategories');
categoriesFolders=dir(rootFolder);
categoriesFolders(1:3)=[];
categories= {categoriesFolders(:).name}';
% categories=categories(randperm(length(categories),20));
% categories=categories(1:10);

imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');

%%
% 显示类别和数量
tbl = countEachLabel(imds)
%%
% 为了使各类样本数量平衡，选取数量最少的为基准抽取样本

minSetCount = min(tbl{:,2});

imds = splitEachLabel(imds, minSetCount, 'randomize');

countEachLabel(imds)
%% 加载AlexNet CNN网络
cnnURL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-caffe-alex.mat';
cnnMatFile = fullfile('../../alexnet', 'imagenet-caffe-alex.mat');
if ~exist(cnnMatFile, 'file') % download only once     
    disp('Downloading pre-trained CNN model...');     
    websave(cnnMatFile, cnnURL);
end
convnet = helperImportMatConvNet(cnnMatFile)

%% 展示CNN结构
convnet.Layers
% 展示第一层结构
convnet.Layers(1)
% 展示最后一层结构
convnet.Layers(end)
% 原始CNN网络的输出类别数
numel(convnet.Layers(end).ClassNames)

%% 图像预处理
% AlexNet CNN以227 227 3的RGB图像作为输入
% 这里把将样本拉伸到227*227并转换为RGB图像的函数
% 作为imageDatastore的读取时调用的函数
imds.ReadFcn = @(filename)readAndPreprocessImage(filename);

%% 分割样本
% 将样本随机分为训练集和测试集
[trainingSet, testSet] = splitEachLabel(imds, 0.3, 'randomize');

%% 选取CNN的fc7层输出作为特征向量
featureLayer = 'fc7';
trainingFeatures = activations(convnet, trainingSet, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');

%% 训练多类别SVM
trainingLabels = trainingSet.Labels;

% 选择线性svm
classifier = fitcecoc(trainingFeatures, trainingLabels, ...
    'Learners', 'svm', 'Coding', 'onevsall', 'ObservationsIn', 'columns');

%% 测试分类器
% 测试训练集
predictedLabels = predict(classifier, trainingFeatures');

% 获取训练集混淆矩阵
confMat = confusionmat(trainingLabels, predictedLabels);

% 转换为百分比
confMat = bsxfun(@rdivide,confMat,sum(confMat,2))

% 显示精度
mean(diag(confMat))


% 获取测试集特征向量
testFeatures = activations(convnet, testSet, featureLayer, 'MiniBatchSize',32);
predictedLabels = predict(classifier, testFeatures);

testLabels = testSet.Labels;

confMat = confusionmat(testLabels, predictedLabels);

confMat = bsxfun(@rdivide,confMat,sum(confMat,2))

mean(diag(confMat))

%% 保存SVM分类器
save('classifier','classifier');


%% 预处理函数
function Iout = readAndPreprocessImage(filename)

I = imread(filename);

% 把灰度图像转换为RGB图像
if ismatrix(I)
    I = cat(3,I,I,I);
end

% 拉伸到277*277
Iout = imresize(I, [227 227]);
end
