function [categories,scores]=predictCategory(im,categoryClassifier,convnet)
if nargin==1
    categoryClassifier=load('classifier.mat');
    categoryClassifier=categoryClassifier.classifier;
    convnet = helperImportMatConvNet('imagenet-caffe-alex.mat');
elseif nargin==2
    convnet = helperImportMatConvNet('imagenet-caffe-alex.mat');
end
img = readAndPreprocessImage(im);
featureLayer = 'fc7';
imageFeatures = activations(convnet, img, featureLayer);
[categories, scores] = predict(categoryClassifier, imageFeatures);
end

%%
% Note that other CNN models will have different input size constraints,
% and may require other pre-processing steps.
function Iout = readAndPreprocessImage(I)
    % Some images may be grayscale. Replicate the image 3 times to
    % create an RGB image. 
    if ismatrix(I)
        I = cat(3,I,I,I);
    end

    % Resize the image as required for the CNN. 
    Iout = imresize(I, [227 227]);  

    % Note that the aspect ratio is not preserved. In Caltech 101, the
    % object of interest is centered in the image and occupies a
    % majority of the image scene. Therefore, preserving the aspect
    % ratio is not critical. However, for other data sets, it may prove
    % beneficial to preserve the aspect ratio of the original image
    % when resizing.
end