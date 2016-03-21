function Net = CNN_ApplyGradient(Net,opts)

    for layer = 2 : numel(Net.Layers)
        if strcmp(Net.Layers{layer}.type , 'c')
           for j = 1 : numel(Net.Layers{layer}.Data)
              for i = 1 : numel(Net.Layers{layer - 1}.Data)
                  Net.Layers{layer}.Kernel{i}{j} = Net.Layers{layer}.Kernel{i}{j} - opts.alpha *Net.Layers{layer}.DeltaKernel{i}{j};
              end
              Net.Layers{layer}.Bias{j} = Net.Layers{layer}.Bias{j} - opts.alpha * Net.Layers{layer}.DeltaBias{j};
           end
        end
    end

    Net.FeaturesOmega = Net.FeaturesOmega - opts.alpha *  Net.DeltaFeaturesOmega;
    Net.FeaturesBias = Net.FeaturesBias - opts.alpha * Net.DeltaFeaturesBias;

end