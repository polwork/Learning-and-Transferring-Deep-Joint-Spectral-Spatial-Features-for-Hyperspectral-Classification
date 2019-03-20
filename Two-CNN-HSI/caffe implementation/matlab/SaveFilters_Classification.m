caffe.reset_all();
clear; close all;
%% settings
folder = 'D:\Code\Caffe-windows\caffe-windows-master\caffe-windows-master\matlab\demo\models\classification\';
model = [folder 'classification_spe_spa_train_test_8_mat.prototxt'];
weights = [folder 'classification_spe_spa_iter_300000.caffemodel'];
savepath = [folder 'classification_spe_spa_8.mat'];
layers = 9;

%% load model using mat_caffe
net = caffe.Net(model,weights,'test');

%% reshap parameters
% save conv11
conv_filters = net.layers(['conv11']).params(1).get_data();
[fsize_w,fsize_h,channel,fnum] = size(conv_filters);
if channel == 1
    weights = single(ones(fsize_h*fsize_w, fnum));
else
    weights = single(ones(channel, fsize_h*fsize_w, fnum));
end
for i = 1 : channel
    for j = 1 : fnum
         temp = conv_filters(:,:,i,j);
         if channel == 1
            weights(:,j) = temp(:);
         else
            weights(i,:,j) = temp(:);
         end
    end
end
weights_conv11 = weights;

% save conv12
conv_filters = net.layers(['conv12']).params(1).get_data();
[fsize_w,fsize_h,channel,fnum] = size(conv_filters);
if channel == 1
    weights = single(ones(fsize_h*fsize_w, fnum));
else
    weights = single(ones(channel, fsize_h*fsize_w, fnum));
end
for i = 1 : channel
    for j = 1 : fnum
         temp = conv_filters(:,:,i,j);
         if channel == 1
            weights(:,j) = temp(:);
         else
            weights(i,:,j) = temp(:);
         end
    end
end
weights_conv12 = weights;




% save conv21
conv_filters = net.layers(['conv21']).params(1).get_data();
[fsize_w,fsize_h,channel,fnum] = size(conv_filters);
if channel == 1
    weights = single(ones(fsize_h*fsize_w, fnum));
else
    weights = single(ones(channel, fsize_h*fsize_w, fnum));
end
for i = 1 : channel
    for j = 1 : fnum
         temp = conv_filters(:,:,i,j);
         if channel == 1
            weights(:,j) = temp(:);
         else
            weights(i,:,j) = temp(:);
         end
    end
end
weights_conv21 = weights;

% save conv22
conv_filters = net.layers(['conv22']).params(1).get_data();
[fsize_w,fsize_h,channel,fnum] = size(conv_filters);
if channel == 1
    weights = single(ones(fsize_h*fsize_w, fnum));
else
    weights = single(ones(channel, fsize_h*fsize_w, fnum));
end
for i = 1 : channel
    for j = 1 : fnum
         temp = conv_filters(:,:,i,j);
         if channel == 1
            weights(:,j) = temp(:);
         else
            weights(i,:,j) = temp(:);
         end
    end
end
weights_conv22 = weights;





% save ip11
fc_filters = net.layers(['ip11']).params(1).get_data();
weights_fc11=fc_filters;
% save ip21
fc_filters = net.layers(['ip21']).params(1).get_data();
weights_fc21=fc_filters;
% save ip2
fc_filters = net.layers(['ip2']).params(1).get_data();
weights_fc2=fc_filters;




% save ip3
fc_filters = net.layers(['ip3']).params(1).get_data();
weights_fc3=fc_filters;


% save ip5
fc_filters = net.layers(['ip5']).params(1).get_data();
weights_fc5=fc_filters;

%% save parameters
biases_conv11 = net.layers('conv11').params(2).get_data();
biases_conv12 = net.layers('conv12').params(2).get_data();
biases_conv21 = net.layers('conv21').params(2).get_data();
biases_conv22 = net.layers('conv22').params(2).get_data();
biases_fc11 = net.layers('ip11').params(2).get_data();
biases_fc21 = net.layers('ip21').params(2).get_data();
biases_fc2 = net.layers('ip2').params(2).get_data();
biases_fc3 = net.layers('ip3').params(2).get_data();
biases_fc5 = net.layers('ip5').params(2).get_data();

save(savepath,'weights_conv11','biases_conv11','weights_conv12','biases_conv12',...
    'weights_conv21','biases_conv21','weights_conv22','biases_conv22',...
    'weights_fc11','biases_fc11','weights_fc21','biases_fc21','weights_fc2','biases_fc2','weights_fc3','biases_fc3','weights_fc5','biases_fc5');

