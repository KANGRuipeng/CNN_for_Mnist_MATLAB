function Net = CNN_ForwardProgatation(Net,X)
% This function used both in train and test 

Layer_Num = numel(Net.Layers);
Net.Layers{1}.Data{1} = X;
Inputmaps = 1;

for layer = 2 : Layer_Num
    
    if strcmp(Net.Layers{layer}.type, 'c')
        
        for j = 1 : Net.Layers{layer}.outputmaps
            % This is used calculate the sizez of this layer
            z = zeros(size(Net.Layers{layer - 1}.Data{1}) - [Net.Layers{layer}.kernelsize - 1 Net.Layers{layer}.kernelsize - 1 0]);
            % For every output, it is the sum of input after convn
            for i = 1 : Inputmaps
                z = z + convn(Net.Layers{layer -1}.Data{i},Net.Layers{layer}.Kernel{i}{j},'valid');
            end
            Net.Layers{layer}.Data{j} = 1./( 1 + exp(-(z + Net.Layers{layer}.Bias{j})));            
        end        
        Inputmaps = Net.Layers{layer}.outputmaps;
        
    elseif strcmp(Net.Layers{layer}.type, 's')
        
        for j = 1 : Inputmaps
           Net.Layers{layer}.Data{j} = Pooling(Net.Layers{layer - 1}.Data{j});
        end
        
        
    end
    
end

Net.Features = [];

for j = 1 : numel(Net.Layers{Layer_Num}.Data)
    SA = size(Net.Layers{Layer_Num}.Data{j});
    Net.Features = [Net.Features; reshape(Net.Layers{Layer_Num}.Data{j}, SA(1) * SA(2), SA(3))];
end

Net.Output = 1./(1 + exp( - (Net.FeaturesOmega * Net.Features + repmat(Net.FeaturesBias, 1, size(Net.Features,2)))));

end