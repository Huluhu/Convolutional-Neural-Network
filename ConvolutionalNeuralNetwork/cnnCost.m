function [cost, grad, preds] = cnnCost(theta,trainimages,trainlabels,numClasses,...
                                filterDimesion,numofFilters,poolDimesion,pred)
% Calcualte cost and gradient for a single layer convolutional
% neural network followed by a softmax layer with cross entropy
% objective.
%                            
% Parameters:
%  theta      -  unrolled parameter vector
%  images     -  stores images in imageDim x imageDim x numImges
%                array
%  numClasses -  number of classes to predict
%  filterDim  -  dimension of convolutional filter                            
%  numFilters -  number of convolutional filters
%  poolDim    -  dimension of pooling area
%  pred       -  boolean only forward propagate and return
%                predictions
%
%
% Returns:
%  cost       -  cross entropy cost
%  grad       -  gradient with respect to theta (if pred==False)
%  preds      -  list of predictions for each example (if pred==True)


if ~exist('pred','var')
    pred = false;
end;


imageDim = size(trainimages,1); % height/width of image
numImages = size(trainimages,3); % number of images

%% Reshape parameters and setup gradient matrices

% Wc is filterDim x filterDim x numFilters parameter matrix
% bc is the corresponding bias

% Wd is numClasses x hiddenSize parameter matrix where hiddenSize
% is the number of output units from the convolutional layer
% bd is corresponding bias
[Wc, Wd, bc, bd] = cnnParamsToStack(theta,imageDim,filterDimesion,numofFilters,...
                        poolDimesion,numClasses);

% Same sizes as Wc,Wd,bc,bd. Used to hold gradient w.r.t above params.
Wc_gradient = zeros(size(Wc));
Wd_gradient = zeros(size(Wd));
bc_gradient = zeros(size(bc));
bd_gradient = zeros(size(bd));

%%======================================================================
%% STEP 1a: Forward Propagation
%  In this step you will forward propagate the input through the
%  convolutional and subsampling (mean pooling) layers.  You will then use
%  the responses from the convolution and pooling layer as the input to a
%  standard softmax layer.

%% Convolutional Layer
%  For each image and each filter, convolve the image with the filter, add
%  the bias and apply the sigmoid nonlinearity.  Then subsample the 
%  convolved activations with mean pooling.  Store the results of the
%  convolution in activations and the results of the pooling in
%  activationsPooled.  You will need to save the convolved activations for
%  backpropagation.
convDim = imageDim-filterDimesion+1; % dimension of convolved output
outputDim = (convDim)/poolDimesion; % dimension of subsampled output

% convDim x convDim x numFilters x numImages tensor for storing activations
activations = zeros(convDim,convDim,numofFilters,numImages);

% outputDim x outputDim x numFilters x numImages tensor for storing
% subsampled activations
activationsPooled = zeros(outputDim,outputDim,numofFilters,numImages);

%%% YOUR CODE HERE %%%
poolFilter = ones(poolDimesion)/(poolDimesion.^2);
for imageNumber = 1:numImages
  for filterNum = 1:numofFilters
      % get the filters 
      filter = Wc(:,:,filterNum);
      filterb = bc(filterNum);
      % prepare for convolution
      filter = rot90(squeeze(filter),2);
      image = squeeze(trainimages(:, :, imageNumber));
      % sigmoid the convolution result plus bias filter
      convolvedImage = sigmoid(conv2(image, filter, 'valid') + filterb);
      activations(:,:,filterNum,imageNumber) = convolvedImage;
      pooledImage = conv2(convolvedImage, poolFilter, 'valid');
      activationsPooled(:,:,filterNum,imageNumber) = pooledImage(1:poolDimesion:end,1:poolDimesion:end);
  end
end
% Reshape activations into 2-d matrix, hiddenSize x numImages,
% for Softmax layer
activationsPooled = reshape(activationsPooled,[],numImages);
%% Softmax Layer
%  Forward propagate the pooled activations calculated above into a
%  standard softmax layer. For your convenience we have reshaped
%  activationPooled into a hiddenSize x numImages matrix.  Store the
%  results in probs.

% numClasses x numImages for storing probability that each image belongs to
% each class.

probs = zeros(numClasses,numImages);
%%% YOUR CODE HERE %%%

probs = bsxfun(@plus, Wd * activationsPooled, bd);
%%======================================================================
%% STEP 1b: Calculate Cost
%  In this step you will use the labels given as input and the probs
%  calculate above to evaluate the cross entropy objective.  Store your
%  results in cost.

cost = 0; % save objective into cost

%%% YOUR CODE HERE %%%
probs = exp(probs);
probs = bsxfun(@rdivide, probs, sum(probs));
groundTruth = full(sparse(trainlabels, 1:numImages, 1));
cost = - 1/numImages * groundTruth(:)' * log(probs(:));

% Makes predictions given probs and returns without backproagating errors.
if pred
    [~,preds] = max(sigmoid(probs),[],1);
    preds = preds';
    grad = 0;
    return;
end;

%%======================================================================
%% STEP 1c: Backpropagation
%  Backpropagate errors through the softmax and convolutional/subsampling
%  layers.  Store the errors for the next step to calculate the gradient.
%  Backpropagating the error w.r.t the softmax layer is as usual.  To
%  backpropagate through the pooling layer, you will need to upsample the
%  error with respect to the pooling layer for each filter and each image.  
%  Use the kron function and a matrix of ones to do this upsampling 
%  quickly.

%%% YOUR CODE HERE %%%
deltad = - (groundTruth - probs); % 10 * numImg, delta of top layer d
deltap = (Wd' * deltad);% * Pooled activations * (1 - activation's Pooled);
deltap = reshape(deltap, outputDim,outputDim,numofFilters,numImages);
deltac = zeros(convDim,convDim,numofFilters,numImages);

for filterNum = 1:numofFilters
  for imageNumber = 1:numImages
    deltac(:,:,filterNum,imageNumber) = (1/poolDimesion^2) *...
        kron(deltap(:,:,filterNum,imageNumber), ones(poolDimesion)).*...
        activations(:,:,filterNum,imageNumber) .*...
        (1 - activations(:,:,filterNum,imageNumber));
  end
end
%%======================================================================
%% STEP 1d: Gradient Calculation
%  After backpropagating the errors above, we can use them to calculate the
%  gradient with respect to all the parameters.  The gradient w.r.t the
%  softmax layer is calculated as usual.  To calculate the gradient w.r.t.
%  a filter in the convolutional layer, convolve the backpropagated error
%  for that filter with each image and aggregate over images.

%%% YOUR CODE HERE %%%2
Wd_gradient = deltad * activationsPooled'/numImages;
bd_gradient = sum(deltad, 2)/numImages;

for imageNumber = 1:numImages
  for filterNum = 1:numofFilters
      % prepare the image (a^(l)_i) and rotated 180's error(delta)
      image = squeeze(trainimages(:, :, imageNumber));
      delta = rot90(squeeze(deltac(:, :, filterNum,imageNumber)), 2);
      
      Wc_gradient(:,:,filterNum) = Wc_gradient(:,:,filterNum) + conv2(image, delta, 'valid');
      bc_gradient(filterNum) = bc_gradient(filterNum) + sum(delta(:));
  end
end
Wc_gradient = Wc_gradient/numImages;
bc_gradient = bc_gradient/numImages;

%% Unroll gradient into grad vector for minFunc
grad = [Wc_gradient(:) ; Wd_gradient(:) ; bc_gradient(:) ; bd_gradient(:)];

end
