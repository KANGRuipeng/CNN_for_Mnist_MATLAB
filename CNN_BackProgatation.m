function Net = CNN_BackProgatation(Net,Y)

Layer_Num = numel(Net.Layers);
%Here the error is the batch and Y is the batch Y
Net.Error = Net.Output - Y;

%  loss function, which is normed, all term add together and devided by
%  batch num
Net.Loss = 1/2* sum(Net.Error(:) .^ 2) / size(Net.Error, 2);

%  backprop deltas
%  output delta
Net.OutputDelta = Net.Error .* (Net.Output .* (1 -Net.Error)); 
%  feature vector delta 192* 10 * 10 * 50;
Net.FeaturesDelta = (Net.FeaturesOmega' * Net.OutputDelta);  


% For this net, this part doesn't work, but for other net, it is necessary
% only conv layers has sigm function
if strcmp(Net.Layers{Layer_Num}.type, 'c')          
    Net.FeaturesDelta = Net.FeaturesDelta .* (Net.Features .* (1 - Net.Features));
end

%  reshape feature vector deltas into output map style
% SA is the size of last layer
SA = size(Net.Layers{Layer_Num}.Data{1});
Features_Num = SA(1) * SA(2);
for j = 1 : numel(Net.Layers{Layer_Num}.Data)
    Net.Layers{Layer_Num}.Delta{j} = reshape(Net.FeaturesDelta(((j - 1) * Features_Num + 1) : j * Features_Num, :), SA(1), SA(2), SA(3));
end

for layer = (Layer_Num - 1) : - 1 : 1
    
    if strcmp(Net.Layers{layer}.type,'c')
        for j = 1 : numel(Net.Layers{layer}.Data)    
            Net.Layers{layer}.Delta{j} = Net.Layers{layer}.Data{j} .* (1 - Net.Layers{layer}.Data{j}) .* ( Expand(Net.Layers{layer + 1}.Delta{j},[Net.Layers{layer+1}.scale  Net.Layers{layer+1}.scale  1]) /  Net.Layers{layer+1}.scale ^ 2);
        end
    elseif strcmp(Net.Layers{layer}.type,'s')
        for i = 1 : numel(Net.Layers{layer}.Data)          
            z  = zeros(size(Net.Layers{layer}.Data{1}));
            for j = 1 : numel(Net.Layers{layer + 1}.Data)
                z = z + convn(Net.Layers{layer + 1}.Delta{j}, Rot180(Net.Layers{layer + 1}.Kernel{i}{j}), 'full');
            end
            Net.Layers{layer}.Delta{i} = z;
        end        
    end
    
    
end

for layer = 2 : Layer_Num
    
    if strcmp(Net.Layers{layer}.type, 'c')
        for j = 1 : numel(Net.Layers{layer}.Data)
           for i = 1 : numel(Net.Layers{layer - 1}.Data)
                Net.Layers{layer}.DeltaKernel{i}{j} = Flipall(convn(Net.Layers{layer -1}.Data{i},Flipall(Net.Layers{layer}.Delta{j}),'valid')) / size(Net.Layers{layer}.Delta{j}, 3);
           end
           Net.Layers{layer}.DeltaBias{j} = sum(Net.Layers{layer}.Delta{j}(:)) / size(Net.Layers{layer}.Delta{j}, 3);
        end        
    end
    
end

Net.DeltaFeaturesOmega = Net.OutputDelta * Net.Features' /size(Net.OutputDelta, 2);
Net.DeltaFeaturesBias = mean(Net.OutputDelta,2);

end