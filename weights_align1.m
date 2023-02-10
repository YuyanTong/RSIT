function [all_weights, reverse_all_weights]=weights_align1(weight_a,weight_b,weight_c,input_dimension)
% post_nodes_num=[256,256,128,35];
% post_nodes_num=[256,128,15];
%post_nodes_num=[128,64,20];        %good nodes setting, only xiongben
%post_nodes_num=[200,100,60];        %good nodes setting, xiongben, dafen, gongqi
%post_nodes_num=[128,64,30];      %good nodes setting, only fudao
%post_nodes_num=[200,100,50];      % good nodes setting, fudao, shanxing
%post_nodes_num=[200,100,80];
%post_nodes_num=[200,100,60];     %  yanshou, gongcheng
%post_nodes_num=[120,100,60];      %good nodes setting, 北海道
%post_nodes_num=[110,60,30];     %xinni
%post_nodes_num=[250,200,150];   %year 176北海道
%post_nodes_num=[150,120,64,20];  %year 16北海道
post_nodes_num=[weight_a,weight_b,weight_c];
layer_nodes_num=[input_dimension, post_nodes_num];
all_weights={};
reverse_all_weights={};
for l=2:length(layer_nodes_num)
    all_weights{l-1}=randn(layer_nodes_num(l-1),layer_nodes_num(l));
    reverse_all_weights{length(layer_nodes_num)-l+1}=pinv(all_weights{l-1});
%     reverse_all_weights{length(layer_nodes_num)-l+1}=(all_weights{l-1})';
end

% %
% pre_layer_nodes_num=input_dimension;
% all_weights={};
% for l=1:length(layer_nodes_num)
%     curr_layer_nodes_num=layer_nodes_num(l);
%     for it=1:curr_layer_nodes_num
%         curr_node_input_weight=randn(1,pre_layer_nodes_num);
%         all_weights{l,it}=curr_node_input_weight;
%     end
%     pre_layer_nodes_num=curr_layer_nodes_num;
% end
% 
