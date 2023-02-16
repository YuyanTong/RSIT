clear;
close all;
warning off;



load GNSS_Sichuan_20180101_20190111        %176锟斤拷锟斤拷锟秸撅拷�???366锟斤拷锟斤拷锟秸?
altitude_matrix_databeihaidao1=GNSS_Sichuan_20180101_20190111;
altitude_matrix_data_YanshouGongcheng=altitude_matrix_databeihaidao1;


% sliding window for smoothing
window_len=4;
for i=1:size(altitude_matrix_data_YanshouGongcheng, 2)-window_len+1
    newdata(:, i)=mean(altitude_matrix_data_YanshouGongcheng(:, i: i+window_len-1), 2);
end


ww=0;
mm=1;   %��¼ͼƬ�ĸ���

predict_len=5;    % embedding length
trainlength=10;
noisestrength=0;
X=newdata+noisestrength*rand(size(newdata));% noise could be added


weight_a=300;
weight_b=300;
weight_c=200;
weight_d=200;
weight_e=100;

clear traindata_x_NN
%aim_station=[1 3 5 9 11 15 17 20 24 27 31 35];
for i=1:23
    aim_station(i)=i;
end
for tt=1:length(aim_station)
    ii=0;
    clear real_y all_real_y traindata_y pcc Loss
    while size(X,2)-ii>=predict_len+trainlength-1
        ii = ii+1;
        xx=X(: , ii:end);
        
        traindata=xx(:,1:trainlength);
        k=30;              % could be changed according to the dimension of X
        
        jd=aim_station(tt);
        real_y=xx(jd, :);
        all_real_y=X(jd, :);
        traindata_y=real_y(1:trainlength);
        clear NN_traindata;
        
        for i=1:trainlength
            traindata_x_NN(:,i)=NN_F2_test(weight_a,weight_b,weight_c,weight_d,weight_e,traindata(:,i));
        end
        
        w_flag=zeros(size(traindata_x_NN,1));
        B_w=zeros(size(traindata_x_NN,1),predict_len);
        for iter=1:1         % cal coeffcient matrix B
            indexr=randperm(setdiff(size(traindata_x_NN,1),jd));
            random_idx=sort([jd,indexr(1:k-1)]);
            traindata_x=traindata_x_NN(random_idx,1:trainlength);        % random chose k variables from F(D)
            %traindata_x=traindata(indexr(1:k-1),1:trainlength);
            clear predict_y super_bb super_AA w;
            for i=1:size(traindata_x,1)
                %  Ax=b,  1: x=pinv(A)*b,    2: x=A\b,    3: x=lsqnonneg(A,b)
                b=traindata_x(i,1:trainlength-predict_len+1)';     % 1*(m-L+1)
                clear A;
                for j=1:trainlength-predict_len+1
                    A(j,:)=traindata_y(j:j+predict_len-1);
                end
                %w(i,:)=(pinv(A)*b)';
                w=(A\b)';
                B_w(random_idx(i),:)=(B_w(random_idx(i),:)+w+w*(1-w_flag(random_idx(i))))/2;
                w_flag(random_idx(i))=1;
            end
        end
        
        for i=1:size(traindata_x_NN,1)
            kt=0;
            clear bb;
            AA=zeros(predict_len-1,predict_len-1);
            for j=(trainlength-(predict_len-1))+1:trainlength
                kt=kt+1;
                bb(kt)=traindata_x_NN(i,j);
                %col_unknown_y_num=j-(trainlength-(predict_len-1));
                col_known_y_num=trainlength-j+1;
                for r=1:col_known_y_num
                    bb(kt)=bb(kt)-B_w(i,r)*traindata_y(trainlength-col_known_y_num+r);
                end
                AA(kt,1:predict_len-col_known_y_num)=B_w(i,col_known_y_num+1:predict_len);
            end
            
            super_bb((predict_len-1)*(i-1)+1:(predict_len-1)*(i-1)+predict_len-1)=bb;
            super_AA((predict_len-1)*(i-1)+1:(predict_len-1)*(i-1)+predict_len-1,:)=AA;
        end
        %  AA*x=bb,  1: x=pinv(AA)*bb,    2: x=AA\bb,    3: x=lsqnonneg(AA,bb)
        %predict_super_y=(pinv(super_AA)*super_bb')';
        pred_y=(super_AA\super_bb')';       % prediction result
        
        myreal=real_y(trainlength+1:trainlength+predict_len-1);
        pcc=corr(pred_y',myreal') ;
        pcc_y(ii)=pcc;
        Loss=sqrt(immse(pred_y, myreal))/std(myreal);
        rmse=sqrt(immse(pred_y, myreal));
        Loss_y(ii)=Loss;
        rmse_y(ii)=rmse;
        
    end
    pcc_station(tt,:)=pcc_y;
    loss_station(tt,:)=Loss_y;
    rmse_station(tt,:)=rmse_y;
end

pcc_station_all_1=pcc_station;
loss_station_all_1=loss_station;
rmse_station_all_sc=rmse_station;
save rmse_station_all_sc rmse_station_all_sc 
