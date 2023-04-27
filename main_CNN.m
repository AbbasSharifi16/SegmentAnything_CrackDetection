clc
clear
close all
%% Train
imageDir = fullfile('SemanticSegmentationDefects/ImageDatastore');
labelDir = fullfile('SemanticSegmentationDefects/PixelLabelDatastore');

imds = imageDatastore(imageDir);

classNames = ["C1" "C2" "C3"];
labelIDs = [1  2 3];
imageSize = [256 256 3];

pxds = pixelLabelDatastore(labelDir, classNames, labelIDs);
ds = pixelLabelImageDatastore(imds,pxds);
  tbl = countEachLabel(pxds);
  totalNumberOfPixels = sum(tbl.PixelCount);
  frequency = tbl.PixelCount / totalNumberOfPixels;
  inverseFrequency = 1./frequency;

% a=gpuArray(7);

layers2 = unetLayers([256 256 3], 3);


options2 = trainingOptions('adam', ...
    'MaxEpochs', 300, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationFrequency', 10, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'multi-gpu');

net = trainNetwork(ds,layers2,options2);
save('netColor.mat','net');
%% Test
I = imread('SemanticSegmentationDefects/ImageDatastore/176.jpg');
GT1=imread('SemanticSegmentationDefects/PixelLabelDatastore/176.png');

[C,scores] = semanticseg(I,net);
%297,176,68, 41, 10, 56, 710

C1=(C=='C1');
C2=(C=='C2');
C3=(C=='C3');

B = labeloverlay(I,C1); 

figure(2)
imshow(imtile({I,GT1,C1,C2,C3}))

imshow(B)

% Define the three colors to use for each label
colors = [1 1 1; 1 0 0; 1 1 1];

% Display the mask with each label colored differently
imshow(ind2rgb(C, colors));


