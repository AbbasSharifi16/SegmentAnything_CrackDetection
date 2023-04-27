clc
clear
close all
%% Train
imageDir = fullfile('SemanticSegmentationDefects/ImageDatastore');
labelDir = fullfile('SemanticSegmentationDefects/PixelLabelDatastore');

imds = imageDatastore(imageDir);

% (17, 141, 215): Water
% (225, 227, 155): Grassland
% (127, 173, 123): Forest
% (185, 122, 87): Hills
% (230, 200, 181): Desert
% (150, 150, 150): Mountain
% (193, 190, 175): Tundra

classNames = ["C1" "C2" "C3"];
labelIDs = [1  2 3];
imageSize = [256 256 3];

pxds = pixelLabelDatastore(labelDir, classNames, labelIDs);
ds = pixelLabelImageDatastore(imds,pxds);
  tbl = countEachLabel(pxds);
  totalNumberOfPixels = sum(tbl.PixelCount);
  frequency = tbl.PixelCount / totalNumberOfPixels;
  inverseFrequency = 1./frequency;
options = trainingOptions('sgdm', ...
    'InitialLearnRate',1e-2, ...
    'MaxEpochs',1000, ...
    'LearnRateDropFactor',1e-1, ...
    'LearnRateDropPeriod',50, ...
    'LearnRateSchedule','piecewise', ...
     'Plots','training-progress',...
    'ExecutionEnvironment','multi-gpu', ...
    'MiniBatchSize',32);
layers = [
    imageInputLayer([256 256 3])
    convolution2dLayer(3,32,'Padding',1)
    reluLayer
    dropoutLayer(0.5)
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,64,'Padding',1)
    reluLayer
    transposedConv2dLayer(4,32,'Stride',2,'Cropping',1)
    convolution2dLayer(1,3)
    softmaxLayer
    pixelClassificationLayer('Classes',tbl.Name,'ClassWeights',inverseFrequency)];
% a=gpuArray(7);

layers2 = unetLayers([256 256 3], 3);

% Apply transformations (using randomly picked values) and build augmented
% data store
%imageAugmenter = imageDataAugmenter( ...
%    'RandRotation',[-20,20], ...
%    'RandXTranslation',[-5 5], ...
 %   'RandYTranslation',[-5 5]);
%augImds = augmentedImageDatastore(imageSize,ds,'DataAugmentation',imageAugmenter);

% (OPTIONAL) Preview augmentation results 
%batchedData = preview(augImds);
%figure, imshow(imtile(batchedData.input))
    
%% Train the network. 
%ds=augImds;
%pool = parpool
%gpuDevice(2)

net = trainNetwork(ds,layers2,options);
save('netColor.mat','net');
%% Test
I = imread('SemanticSegmentationDefects/ImageDatastore/4.jpg');
GT1=imread('SemanticSegmentationDefects/PixelLabelDatastore/4.png');

[C,scores] = semanticseg(I,net);
% B = labeloverlay(I,C); 
C1=(C=='C1');
C2=(C=='C2');
C3=(C=='C3');


figure
imshow(imtile({I,GT,C1,C2,C3}))


