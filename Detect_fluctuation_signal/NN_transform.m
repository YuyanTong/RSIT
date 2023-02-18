function output = NN_transform(traindata, weights)
[input_dimension, trainlength]=size(traindata);
clear output;
for i=1:trainlength
    pre_layer_nodes_num=input_dimension;
    pre_layer_nodes_value=traindata(:,i)';         % input
    for l=1:length(weights)
        curr_weights=weights{l};
        curr_layer_nodes_value=tanh(pre_layer_nodes_value*curr_weights/3);
        curr_layer_nodes_value=myLeakyRelu(pre_layer_nodes_value*curr_weights,0.01, 1);
        pre_layer_nodes_value=curr_layer_nodes_value;
    end
    output(:, i)=curr_layer_nodes_value;
end


% clear output
% for i=1:trainlength
%     pre_layer_nodes_num=input_dimension;
%     pre_layer_nodes_value=traindata(:,i);         % input
%     for l=1:length(layer_nodes_num)
%         curr_layer_nodes_num=layer_nodes_num(l);
%         clear curr_layer_nodes_value;
%         for it=1:curr_layer_nodes_num
%             curr_node_input_weight=all_weights{l,it};
% %              size(pre_layer_nodes_value) %
% %              size(curr_node_input_weight)
%             xx=sum(pre_layer_nodes_value.*curr_node_input_weight');
%             curr_layer_nodes_value(it)=tanh(xx/2.5);
%         end
%         pre_layer_nodes_value=curr_layer_nodes_value';
%         pre_layer_nodes_num=curr_layer_nodes_num;
%     end
%     output(:, i)=curr_layer_nodes_value;
% end
% 