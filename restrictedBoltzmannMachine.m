classdef restrictedBoltzmannMachine
    %restrictedBoltzmannMachine Class to implement restricted Boltzmann
    %machines
    %   Currently simple implementation of binary RBMs which are to be
    %   trained using CD-1. To be later generalised/extended to CRBMs and
    %   CD-k. If possible in future add GPU support.
    
    properties
        class = 'rbm';          %general name
        numSamples;             %number of samples
        numVisible;             %number of visible units
        numHidden;              %number of hidden units
        data;                   %data matrix
        weights;                %connection weight matrix
        dWeights;               %change in weight matrix
        visibleBiases;          %visible layer bias
        dVisibleBiases;         %change in visible layer bias
        hiddenBiases;           %hidden layer bias
        dHiddenBiases;          %change in hidden layer bias
        learningRate = 0.0001;  %learning rate
        numEpochs;              %number of epochs
        cdk = 1;                %contrastive divergence-k
        phase=false;            %flag for negative/positive phase
        visibleActivations;     %visible layer activations
        hiddenActivations;      %hidden layer activations
        visibleProbs;           %visible layer probabilities
        hiddenProbs;            %hidden layer probabilities
        hiddenProbsPos;         %hidden layer probabilities in positive phase
        hiddenProbsNeg;         %hidden layer probabilities in negative phase
        positiveAssociations;   %positive phase associations
        negativeAssociations;   %negative phase associations
        visibleStates;          %visible layer states
        hiddenStates;           %hidden layer states
        error;                  %error
    end
    
    methods
        
        function self = restrictedBoltzmannMachine(numVisible,numHidden,numEpochs)
        %Class constructor for restrictedBoltzmannMachine class
        %INPUT:
        %   numVisible - number of visible units
        %   numHidden - number of hidden units
        %   numEpochs - number of epochs for training
        %OUTPUT:
        %   self - restrictedBoltzmannMachine class object

            self.numVisible = numVisible;
            self.numHidden = numHidden;
            self.numEpochs = numEpochs;

            %Initialise weights and biases randomly
            self.weights = 0.1*randn(self.numVisible,self.numHidden);
            self.dWeights = zeros(self.numVisible,self.numHidden);
            self.visibleBiases = 0.01*randn(1,numVisible);
            self.dVisibleBiases = zeros(1,numVisible);
            self.hiddenBiases = 0.01*randn(1,numHidden);
            self.dHiddenBiases = zeros(1,numHidden);
        end
        
        function probabilities = sigmoid(self,activations)
        %sigmoid activation function
        %INPUT:
        %   activations - data to be activated
        %OUTPUT:
        %   probabilities - activation probabilities according to sigmoid function

            probabilities = 1./(1+exp(-activations));
        end
        
        function self = train(self,data)
        %training function for RBM
        %INPUT:
        %   data - daa matrix
        %OUtPUT:
        %   self - trained restrictedBoltzmannMachine class object
        
            %Initislise class members and preallocate space
            self.data = data;
            self.numSamples = size(self.data,1);
            self.visibleActivations = zeros(self.numSamples,self.numHidden);
            self.hiddenActivations = zeros(self.numSamples,self.numVisible);
            self.hiddenProbs = zeros(self.numSamples,self.numHidden);
            self.visibleProbs = zeros(self.numSamples,self.numVisible);
            self.hiddenStates = zeros(self.numSamples,self.numHidden);
            self.visibleStates = zeros(self.numSamples,self.numVisible);            
            self.positiveAssociations = zeros(self.numVisible,self.numHidden);
            self.negativeAssociations = zeros(self.numVisible,self.numHidden);
            
            %Ierate for specified epochs
            for epoch=1:self.numEpochs
                %Negative phase - sample hidden from visible
                self = self.sampleHidden();
                self.positiveAssociations = self.data' * self.hiddenProbs;
                self.hiddenProbsPos = self.hiddenProbs;
                %Contrastive divergence step
                for k = 1:1:self.cdk
                    %Positive phase - sample visible from hidden
                    self.phase = true;
                    self = self.sampleVisible();
                    self = self.sampleHidden();
                end
                self.phase = false;
                self.negativeAssociations = self.visibleProbs' * self.hiddenProbs;
                self.hiddenProbsNeg = self.hiddenProbs;
                %Update weights and biases to learn parameters
                self = self.updateWeightsAndBiases();
                %Display current progress
                fprintf('Epoch: %d | Error = %.10f \n', epoch, self.error);
                %Show weights in tiled grid of images
                tiledWeights = showWeights(self,[10 10],[28 28]);
                imshow(tiledWeights, []), title('Weights');
                pause(0.1);
            end
            
            %Clear unused members
            self.visibleActivations = [];   self.hiddenActivations = [];
            self.hiddenProbs = [];          self.visibleProbs= [];
            self.hiddenStates= [];          self.visibleStates = [];
            self.positiveAssociations = []; self.negativeAssociations =[];
            self.hiddenProbsNeg = [];       self.hiddenProbsPos = [];
            
        end
        
        function self = sampleHidden(self)
        %function to sample hidden units given visible units

            %Negative phase
            if ~self.phase
                hiddenActivations = self.data * self.weights;
                self.hiddenActivations = bsxfun(@plus,hiddenActivations,self.hiddenBiases);
            %Positive phase
            else
                hiddenActivations = self.visibleProbs * self.weights;
                self.hiddenActivations=bsxfun(@plus,hiddenActivations,self.hiddenBiases);
            end
            %Set hidden probabilities and states
            self.hiddenProbs = self.sigmoid(self.hiddenActivations);
            self.hiddenStates = self.hiddenProbs > rand(size(self.hiddenProbs));
        end
        
        function self = sampleVisible(self)
        %function to sample hidden units given visible units

            visibleActivations = self.hiddenStates * transpose(self.weights);
            self.visibleActivations = bsxfun(@plus,visibleActivations,self.visibleBiases);
            self.visibleProbs = self.sigmoid(self.visibleActivations);
            self.visibleStates = self.visibleProbs > rand(size(self.visibleProbs));
        end
        
        function self = updateWeightsAndBiases(self)
        %function to update weights and biases to learn parameters

            %update weights
            self.dWeights = self.learningRate * (self.positiveAssociations-self.negativeAssociations);
            self.weights = self.weights + self.dWeights;

            %calculate current error
            self.error = sqrt(sum(mean((self.data-self.visibleProbs).^2)));
            
            %update visible biases
            self.dVisibleBiases = self.learningRate * mean(self.data - self.visibleProbs);
            self.visibleBiases = self.visibleBiases + self.dVisibleBiases;
            
            %update hidden biases
            self.dHiddenBiases = self.learningRate * mean(self.hiddenProbsPos - self.hiddenProbsNeg);
            self.hiddenBiases = self.hiddenBiases + self.dHiddenBiases;
        end
        
        function tiledWeightImage = showWeights(self, weightGridSize, weightImageShape)
        %function to display training progress by tiling weights in a grid
        %INPUT:
        %   weightGridSize - size of grid of weights
        %   weightImageShape - shape of subimage in grid
        %OUTPUT
        %   tiledWeightImage - tiled images of weights arranged in a grid

            %Initialise image
            tiledWeightImage = zeros(weightGridSize(1)*weightImageShape(1) + weightGridSize(1) + 1,weightGridSize(2)*weightImageShape(2) + weightGridSize(2) + 1);
            for i = 1:weightGridSize(1)
                for j = 1:weightGridSize(2)
                    weightIndex = (i - 1)*weightGridSize(2) + j;
                    weightImage = self.weights(:, weightIndex);
                    weightImage = reshape(weightImage, weightImageShape);
                    
                    rows = weightImageShape(1)*(i - 1) + i + 1;
                    rowe = rows + weightImageShape(1) - 1;
                    cols = weightImageShape(2)*(j - 1) + j + 1;
                    cole = cols + weightImageShape(2) - 1;
                    
                    tiledWeightImage(rows:rowe, cols:cole) = weightImage;
                end
            end
            tiledWeightImage = (tiledWeightImage - min(tiledWeightImage(:)))/(max(tiledWeightImage(:)) - min(tiledWeightImage(:)));
            tiledWeightImage(1:weightImageShape(1)+1:end, :) = 1;
            tiledWeightImage(:, 1:weightImageShape(2)+1:end) = 1;
        end
        
        function print(self)
            properties(self)
            methods(self)
        end
        
    end
    
end
