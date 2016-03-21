function [bad, er] = CNN_Test(Net, Test_X, Test_Y)

Net = CNN_ForwardProgatation (Net,Test_X);
[~,h] = max(Net.Output);
[~, a] = max(Test_Y);
bad = find(h ~= a);

er = numel(bad) / size(Test_Y, 2);


end