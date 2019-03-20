function [pro,feature] = Class_CNN(X, spe, spa)

%% load CNN model parameters
weights_conv11=X{1};biases_conv11=X{2};
weights_conv12=X{3};biases_conv12=X{4};
weights_conv21=X{5};biases_conv21=X{6};
weights_conv22=X{7};biases_conv22=X{8};
weights_fc11=X{9};biases_fc11=X{10};
weights_fc21=X{11};biases_fc21=X{12};
weights_fc2=X{13};biases_fc2=X{14};
weights_fc3=X{15};biases_fc3=X{16};
weights_fc4=X{17};biases_fc4=X{18};



[hei_spe, wid_spe] = size(spe);
[hei_spa, wid_spa] = size(spa);

%% conv11
weights_conv11 = single(reshape(weights_conv11, size(weights_conv11,1), 1, size(weights_conv11,2)));
conv11_data = single(zeros(hei_spe-size(weights_conv11,1)+1, 1, size(weights_conv11,3)));
for i = 1 : size(weights_conv11,3)
    conv11_data(:,:,i) = filter2(weights_conv11(:,:,i), spe, 'valid');
    conv11_data(:,:,i) = conv11_data(:,:,i) + biases_conv11(i);
end

%% pool11
conv11_pool=single(zeros(floor(size(conv11_data,1)/5),1,size(conv11_data,3)));
for i = 1 : size(weights_conv11,3)
    conv11_pool(:,:,i) = Pooling_my(5, 1, 5, conv11_data(:,:,i));
    conv11_pool(:,:,i) = max(conv11_pool(:,:,i), 0);
    %conv11_pool(:,:,i) = (exp(conv11_pool(:,:,i)+ biases_conv11(i))-exp(-conv11_pool(:,:,i)- biases_conv11(i)))./(exp(conv11_pool(:,:,i)+ biases_conv11(i))+exp(-conv11_pool(:,:,i)- biases_conv11(i)));
end

%% conv12
conv12_data = single(zeros(size(conv11_pool,1)-size(weights_conv12,2)+1, 1, size(weights_conv12,3)));
for i = 1 : size(weights_conv12,3)
    for j = 1 : size(weights_conv11,3)
        filt_conv12 = single(reshape(weights_conv12(j,:,i),16,1));
        conv12_data(:,:,i) = conv12_data(:,:,i) + filter2(filt_conv12, conv11_pool(:,:,j), 'valid');
    end
    conv12_data(:,:,i) = max(conv12_data(:,:,i) + biases_conv12(i), 0);
end



%% flatten
fla_spe=reshape(conv12_data,size(conv12_data,1)*size(conv12_data,2)*size(conv12_data,3),1);

%% conv21
weights_conv21 = single(reshape(weights_conv21, sqrt(size(weights_conv21,1)), sqrt(size(weights_conv21,1)), size(weights_conv21,2)));
conv21_data = single(zeros(hei_spa-size(weights_conv21,1)+1, wid_spa-size(weights_conv21,2)+1, size(weights_conv21,3)));
for i = 1 : size(weights_conv21,3)
    tmp=weights_conv21(:,:,i);
    tmp=tmp';
    conv21_data(:,:,i) = filter2(tmp, spa, 'valid');
    conv21_data(:,:,i) = conv21_data(:,:,i) + biases_conv21(i);
end


%% pool21
conv21_pool=single(zeros(floor(size(conv21_data,1)/2)+1,floor(size(conv21_data,2)/2)+1,size(conv21_data,3)));
for i = 1 : size(weights_conv21,3)
    conv21_pool(:,:,i) = Pooling_my(2, 2, 2, conv21_data(:,:,i));
    conv21_pool(:,:,i) = max(conv21_pool(:,:,i), 0);
end

%% conv22
conv22_data = single(zeros(size(conv21_pool,1)-sqrt(size(weights_conv22,2))+1, size(conv21_pool,2)-sqrt(size(weights_conv22,2))+1, size(weights_conv22,3)));
for i = 1 : size(weights_conv22,3)
    for j = 1 : size(conv21_data,3)
        filt_conv22 = reshape(weights_conv22(j,:,i),3,3);
        filt_conv22 = filt_conv22';
        conv22_data(:,:,i) = conv22_data(:,:,i) + filter2(filt_conv22, conv21_pool(:,:,j), 'valid');
    end
    conv22_data(:,:,i) = max(conv22_data(:,:,i) + biases_conv22(i), 0);
end



%% flatten
fla_spa = single(zeros(1,1));
for num=1:size(conv22_data,3)
    tmp=conv22_data(:,:,num);
    tmp=tmp';
    tmp=tmp(:);
    fla_spa=cat(1,fla_spa,tmp);
end
fla_spa(1,:)=[];
%% fc11
ip11 = max(weights_fc11'*fla_spe + biases_fc11, 0);
%ip11 = (exp(ip11)-exp(-ip11))./(exp(ip11)+exp(-ip11));

%% fc21
ip21 = max(weights_fc21'*fla_spa + biases_fc21, 0);

%% concatenate
fla = [ip11', ip21']';

%% fc2
ip2 = max(weights_fc2'*fla + biases_fc2, 0);


%% fc3
ip3 = max(weights_fc3'*ip2 + biases_fc3, 0);


%% fc4
ip4 = weights_fc4'*ip3 + biases_fc4;
feature = ip3;    % output the feature of final layer

%% softmax
pro = exp(ip4-max(ip4))/sum(exp(ip4-max(ip4)));
% pro = exp(ip3)/sum(exp(ip3));
% pro=ip3;
















