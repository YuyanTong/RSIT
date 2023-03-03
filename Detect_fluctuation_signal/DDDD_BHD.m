clear;
close all;
warning off;


samples=11;
stationnum=16;
sample_use=365;
embeddings_num=5;

timepoint=sample_use-samples+1;
normal_sd_y=zeros(timepoint,1);
normal_sd_original=zeros(stationnum,timepoint);     
critical_sd_y=0;
ii=0;
good_ii=0;
reps=0; 
nreps=0;

load data_2018_ave_new        %176个监测站，366个监测站
%从2018年9月6日凌晨发生地震

altitude_matrix_databeihaidao1=data_2018_ave_new';
altitude_matrix_databeihaidao1=altitude_matrix_databeihaidao1;


mm=1;
weight_a=1;
weight_b=1;
weight_c=1;
KK=1;
for i1 = 200:-10:50
    for i2=10
        for i3=5
            A(KK,1)=i1;
            A(KK,2)=i2;
            A(KK,3)=i3;
            KK=KK+1;
        end
    end
end

% select=[2 9 9]
% for i=1:length(select)
%     data(i,:)=A(select(i),:);
% end
data=A;
for i=1:length(data)
    weight_a=data(i,1);
    weight_b=data(i,2);
    weight_c=data(i,3);
            
            
            timepoint=sample_use-samples+1;
            normal_sd_y=zeros(timepoint,1);
            normal_sd_original=zeros(stationnum,timepoint);
            critical_sd_y=0;
            ii=0;
            good_ii=0;
            reps=0;
            nreps=0;

    while ii<5000                                                                                                                                                                                                                            %寰幆鎵ц�?3000娆★紝鍙栧潎鍊硷紝鏀瑰杽璇�?
        ii=ii+1;
        %sample_use=184;
        input_dimensions=stationnum;   
        size(altitude_matrix_databeihaidao1);    
          [all_weights, reverse_all_weights]=weights_align1(weight_a,weight_b,weight_c,size(altitude_matrix_databeihaidao1,1));    %随机给定神经网络的权值矩阵和逆矩阵，即F和F-1
                %critical zone
        %samples=13;   
        critical_zone=[sample_use-samples+1:sample_use];   
        normal_start=0;      %寰幆寰�?埌姝ｅ父鏃舵湡鐨勬暟鎹紝姣?5涓椂闂寸偣浣滀负涓?涓椂闂寸偣
        %瀵规墍鏈夌殑鏃堕棿鐐?5�??5涓彇锛屽緱鍒颁竴涓皬鍧楋紝浣�?负涓?涓椂闂寸偣鐨勬暟鎹紝�??鍚庣湅杩欎釜鏃堕棿鍧楁暟鎹殑SD鍊硷紙鍙湁灏忓潡鐨勬暟鎹墠鑳界湅SD锛屽崟涓暟鎹棤娉曟煡鐪婼D�??
        while normal_start+samples-1 < sample_use-samples+samples     %5涓暟鎹?5涓暟鎹繘琛屽垏鍒?
            normal_start = normal_start + 1;
            normal_zone=[normal_start:normal_start+samples-1];
            myzone{1}=normal_zone;   %姝ｅ父鏃舵湡鏁版�?
            
            normal_sd_original(:,normal_start)=std(altitude_matrix_databeihaidao1(:,myzone{1}),0,2);
            
            for mys=1:1
                UU=altitude_matrix_databeihaidao1(:, myzone{mys});
                [input_dimensions, time_points]=size(UU);
                %     figure((mys-1)*3+1)
                %     plot(std(reshape(UU(1,:,:),time_points,samples),0,2));
                flat_y=zeros(time_points+embeddings_num-1,1);
                xx=UU;
                noisestrength=0;          % noise could be added
                xx_noise=xx+noisestrength*rand(size(xx));
                D=size(xx_noise,1);     % number of variables in the system.
                trainlength=time_points;
                %embeddings_num=11;   %寤惰繜宓屽叆鐨勬暟锛孡锛�?<m
                
                traindata=xx_noise(:,1:trainlength);
                
                traindata_x_NN=NN_transform(traindata,all_weights);
                test_NN=traindata_x_NN;
                %test
                %         traindata_x_NN=traindata;
                F_dimensions=size(traindata_x_NN,1);
                
                %% solve weight matrix A
                %  AA, coefficient matrix
                AA=zeros((trainlength-1)*(embeddings_num-1)+1, embeddings_num*F_dimensions);
                b=zeros((trainlength-1)*(embeddings_num-1)+1,1);
                for k=0:embeddings_num-2
                    for t=2:trainlength
                        AA(k*(trainlength-1)+t-1, k*F_dimensions+1:(k+1)*F_dimensions)=traindata_x_NN(:, t);
                        AA(k*(trainlength-1)+t-1, (k+1)*F_dimensions+1:(k+2)*F_dimensions)=-traindata_x_NN(:, t-1);
                    end
                end
                AA((trainlength-1)*(embeddings_num-1)+1, 1:F_dimensions)=traindata_x_NN(:, 2);
                %         b((trainlength-1)*(embeddings_num-1): (trainlength-1)*(embeddings_num-1)+1)=1;
                b((trainlength-1)*(embeddings_num-1)+1)=traindata_x_NN(1,1);
                flat_A=AA\b;
                weight_A=zeros(embeddings_num,F_dimensions);
                for i=1:embeddings_num
                    weight_A(i, :)=flat_A((i-1)*F_dimensions+1: i*F_dimensions);
                end
                
                
                %% solve Y
                Y=weight_A*traindata_x_NN;      %Y size: L*m
                %Y(1,1)=-1;
                
                %% solve weight matrix B
                
                % For AE
                B=traindata_x_NN/Y;
                X_real=xx_noise(:,1:trainlength);  %娌℃湁缁忚繃绁炵粡缃戠粶涔嬪墠鐨勬暟�??
                II= weight_A*B;
                
                for t=1:trainlength
                    for j=1:min(t,embeddings_num)
                        m=(t+1)-j;
                        l=(t+1)-m;
                        % disp([num2str(l),', ', num2str(m)]);
                        flat_y(t)=flat_y(t)+ Y(l,m);
                    end
                    flat_y(t)=flat_y(t)/min(t,embeddings_num);
                end
                for t=trainlength+1:trainlength+embeddings_num-1
                    for j=1:embeddings_num-(t-trainlength)
                        m=trainlength+1-j;
                        l=t+1-m;
                        flat_y(t)=flat_y(t)+ Y(l,m);
                    end
                    flat_y(t)=flat_y(t)/(embeddings_num-(t-trainlength));
                end
                
                %     flat_y(:,iter)=Y(1,:);
                
                sd1(mys,:)=std(flat_y,0,2);
                test_flat_y(mys,:)=flat_y;
                %     figure((mys-1)*3+2);
                %     plot(sd1(3:end));
                % %     figure((mys-1)*3+3);
                % %     plot([3:time_points], Y(1,3:end));
                %     hold on;
            end
            %normal琛ㄧず鏃堕棿鐐癸紝姣忎簲涓椂闂翠綔涓轰竴涓椂闂寸偣锛屼竴鍏辨�?346涓椂闂寸偣
            %浠ユ鏉ユ帹娴嬩复鐣�?偣鏄摢�??涓椂闂寸偣
            normal_sd_y(normal_start)=normal_sd_y(normal_start) + std(test_flat_y(1,1:end));    %灏嗘瘡涓?娆℃眰寰楃殑鏍囧噯宸疮鍔犺捣鏉?
            reps=reps+1;
        end
        
        %     if critical_sd_y>normal_sd_y
        %         good_ii=good_ii+1;
        %     end
        if mod(ii,100)==0       %姣忎竴鐧炬杈撳嚭涓?�??
            ii
        end
    end
            %li是随机试验的次数
            %critical时间点其实就是normal时间点的�?后一�?
            normal_sd_y=normal_sd_y/ii;      %求出正常时期的平均�??
            critical_sd_y=critical_sd_y/reps;    %求出临界时期的平均�??
            combined_sd_y=[normal_sd_y',critical_sd_y];
            %             figure;
            %             plot(combined_sd_y);      %降维之后的数�?
            %             figure;
            %             a=mean(normal_sd_original,1);
            %             plot(a)        %直接的原始数据做
            %           saveas(gcf, 'test', 'png')
            %           saveas(gcf,[C:\Users\tongyuyuan\Documents\MATLAB\tyy\ARNN\picture','yanbao',num2str(k),'.jpg']);
            %
            plot(combined_sd_y);    %读入图片，如1_predict_prob.png
            %*********处理图片（省略）**********%
            saveas(gcf,['/home/tongyuyan/地震/重跑数据/阿拉斯加/每日数据/2018/pic_11_5/','selet_cc',num2str(mm),'.fig']);
            %saveas(gcf,['D:\MATLAB7\work','yanbao',num2str(k),'.jpg']);
            %imwrite(Image,[savepath,num2str(mm,'%04d'),'.png']);%将处理后的图片保存在D:\yl\数据\下，命名�?0001.png
            mm
            mm=mm+1;
                 
        end



%滑动窗口�?后移动，本来�?350�?6个为�?个窗口，时间点变�?365-3=361   184-12=181




% good_ii
% ii
% good_ii/ii
% plot(sd1(1,1:end),'b','LineWidth',2);
% hold on;
% plot(sd1(2,1:end),'r','LineWidth',2);
% title('SD of Y')

% figure;
% plot(test_flat_y(1,1:end),'b','LineWidth',2);
% hold on;
% plot(test_flat_y(2,1:end),'r','LineWidth',2);
% title(['Flat y, SD of red:', num2str(critical_sd_y),', SD of blue:', num2str(normal_sd_y)])

%
% for i=1:15
%     figure;
%     plot(std(reshape(test_NN(i,:,:),time_points,samples),0,2));
% end