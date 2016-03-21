function Picture = Pooling(Image)

SA = size(Image);
Image_Buffer = zeros(SA);

for j = 1:2:SA(2)-1   
    Image_Buffer(:,(j+1)/2,:) = Image(:,j,:) + Image(:,j+1,:);        
end

for i = 1:2:SA(1)-1
    Image_Buffer((i+1)/2,1:SA(2)/2,:) = Image_Buffer(i,1:SA(2)/2,:) + Image_Buffer(i+1,1:SA(2)/2,:);       
end


Picture = Image_Buffer(1:SA(1)/2,1:SA(2)/2,1:SA(3))./4;

end