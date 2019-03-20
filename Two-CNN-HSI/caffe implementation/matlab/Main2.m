clc;clear;close all;
addpath('D:\Code\Caffe-windows\caffe-windows-master\caffe-windows-master\matlab\demo');
%% download


load Salinas.mat
load Salinas_gt
data=single(salinas);gt=single(salinas_gt);
data_aug=single(zeros(size(data,1)+20,size(data,2)+20,size(data,3)));
for num=1:size(data,3)    % normalizing of data
    data(:,:,num)=(data(:,:,num)-min(min(data(:,:,num))))/(max(max(data(:,:,num)))-min(min(data(:,:,num))));
    data_aug(:,:,num)=scale_change(data(:,:,num),21);    % augment the ground truth
end
data_1=data_aug;data_1(:,:,224)=[];data_1(:,:,222)=[];data_1(:,:,153:167)=[];data_1(:,:,107:113)=[];
pan=mean(data_1,3);    % simulated panchromatic image
data=data_1;

% load PaviaU;load PaviaU_gt
% data=single(paviaU);
% gt=single(paviaU_gt);
% data_aug=single(zeros(size(data,1)+20,size(data,2)+20,size(data,3)));
% for num=1:size(data,3)    % normalizing of data
%     data(:,:,num)=(data(:,:,num)-min(min(data(:,:,num))))/(max(max(data(:,:,num)))-min(min(data(:,:,num))));
%     data_aug(:,:,num)=scale_change(data(:,:,num),21);    % augment the ground truth
% end

% data_1=data_aug;data_1(:,:,3)=[];data_1(:,:,2)=[];data_1(:,:,1)=[];
% pan=mean(data_1,3);    % simulated panchromatic image
% data=data_1;


load classification_spe_spa_8.mat % load the mat that is saved from SaveFilters_Classification.m 
X{1}=weights_conv11;X{2}=biases_conv11;
X{3}=weights_conv12;X{4}=biases_conv12;
X{5}=weights_conv21;X{6}=biases_conv21;
X{7}=weights_conv22;X{8}=biases_conv22;
X{9}=weights_fc11;X{10}=biases_fc11;
X{11}=weights_fc21;X{12}=biases_fc21;
X{13}=weights_fc2;X{14}=biases_fc2;
X{15}=weights_fc3;X{16}=biases_fc3;
X{17}=weights_fc5;X{18}=biases_fc5;


class_map=zeros(size(gt));
for row=11:size(data,1)-10
    for col=11:size(data,2)-10
        if gt(row-10,col-10)==0
            continue
        else
            spe=data(row,col,:);
            spe=spe(:);
            spa=pan(row-10:row+10,col-10:col+10);
            [pro,fea]=Class_CNN(X, spe, spa);
            [p,idx]=max(pro);
            class_map(row-10,col-10)=idx;
            lab=gt(row-10,col-10);
        end
    end
end
imshow(class_map,[]);









