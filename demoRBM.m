%Demo script to show restrictedBoltzmannMachine class
load('mnistSmall.mat');

%Initialise number of visible and hidden units, and epochs for training
numVisible=784;
numHidden=100;
numEpochs=5000;

%Initialise object of restrictedBoltzmannMachine class
rbm=restrictedBoltzmannMachine(numVisible,numHidden,numEpochs);

%Train object of restrictedBoltzmannMachine class using MNIST data
rbm=rbm.train(trainData);