function Net = CNN_Train(Net,Train_X,Train_Y,opts)

Data_Num = size(Train_X, 3);
Num_Batches = Data_Num / opts.batchsize;
% Judge whether batchsize is suitable
if rem(Num_Batches, 1) ~= 0
    error('numbatches not integer');
end

for i = 1 : opts.numepochs
    
    disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)]);
    tic;    
    % This is used to randomly batch data
    Batch_Random = randperm(Data_Num);
    
    for k = 1 : Num_Batches
        batch_x = Train_X(:, :, Batch_Random((k - 1) * opts.batchsize + 1 : k * opts.batchsize));
        batch_y = Train_Y(:,    Batch_Random((k - 1) * opts.batchsize + 1 : k * opts.batchsize));
        
        Net = CNN_ForwardProgatation(Net, batch_x);
        Net = CNN_BackProgatation(Net, batch_y);
        Net = CNN_ApplyGradient(Net, opts);
        
    end   
    toc;
end

end