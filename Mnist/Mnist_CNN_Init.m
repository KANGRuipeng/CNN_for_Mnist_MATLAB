function Net= Mnist_CNN_Init(Train_X_Size,Train_Y_Size)

%Define the structure of Net
Net.Layers = {};

% Input layer;
Net.Layers{ end + 1} = struct('type', 'i'); 

% Convolution layer
Net.Layers{ end + 1} = struct('type', 'c', ...
                              'outputmaps', 6,...
                              'kernelsize', 5);
                          
% Pooling layer
Net.Layers{ end + 1} = struct('type', 's', ...
                              'scale', 2);
                          
% Convolution layer
Net.Layers{ end + 1} = struct('type', 'c', ...
                              'outputmaps', 12,...
                              'kernelsize', 5);
                          
% Pooling layer
Net.Layers{ end + 1} = struct('type', 's', ...
                              'scale', 2);


Inputmaps = 1;
% Get the size of input picture
Mapsize(:,1:2) = Train_X_Size(:,1:2);

%This part initial kernel and bias
for layer = 1 : numel(Net.Layers)
    
    if strcmp (Net.Layers{layer}.type, 's')
        
        Mapsize = Mapsize / Net.Layers{layer}.scale;
        for j = 1: Inputmaps           
            Net.Layers{layer}.Bias{j} = 0;
        end
        
    elseif strcmp (Net.Layers{layer}.type, 'c')   
        
        Mapsize = Mapsize - Net.Layers{layer}.kernelsize + 1;        
        for j = 1 : Net.Layers{layer}.outputmaps            
            for i = 1 : Inputmaps
                  Net.Layers{layer}.Kernel{i}{j} = 0.01* randn(Net.Layers{layer}.kernelsize);
            end
            Net.Layers{layer}.Bias{j} = 0;
        end
        Inputmaps = Net.Layers{layer}.outputmaps;  
        
    end
        
end

% 4 * 4 * 12
Features_Num = prod(Mapsize) * Inputmaps;

%This part is process output
Net.FeaturesBias = zeros(Train_Y_Size(1), 1);
Net.FeaturesOmega = 0.01*randn(Train_Y_Size(1),Features_Num);

end