function [ output ] = myLeakyRelu(x, leak, flag)
% x is a vector
% flag=1, LeakyRelu; flag=2, aLeakyRelu
if flag==1
    myleak=leak;
elseif flag==2
    myleak=1/leak;
end
output=zeros(size(x));
output(find(x>=0))=x(find(x>=0));
output(find(x<0))=myleak*x(find(x<0));
end

