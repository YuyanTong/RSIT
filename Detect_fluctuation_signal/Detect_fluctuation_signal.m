clc
clear;
close all;
warning off;

samples=11; 
embeddings_num=5;
weight_a=190;
weight_b=10;
weight_c=5;
stationnum=16;
sample_use=365;


load GNSS_South_central_Alaska_20180101_20181231_new      
altitude_matrix_databeihaidao1=GNSS_South_central_Alaska_20180101_20181231_new';
altitude_matrix_databeihaidao1=altitude_matrix_databeihaidao1;

timepoint=sample_use-samples+1;
normal_sd_y=zeros(timepoint,1);
normal_sd_original=zeros(stationnum,timepoint);
critical_sd_y=0;
ii=0;
good_ii=0;
reps=0;
nreps=0;

while ii<30000                                                                                                                                                                                                                            %寰幆鎵ц�??3000娆★紝鍙栧潎鍊硷紝鏀瑰杽璇�??
    ii=ii+1;
    input_dimensions=stationnum;
    size(altitude_matrix_databeihaidao1);
    [all_weights, reverse_all_weights]=weights_align1(weight_a,weight_b,weight_c,size(altitude_matrix_databeihaidao1,1));   
    critical_zone=[sample_use-samples+1:sample_use];
    normal_start=0;        
    while normal_start+samples-1 < sample_use-samples+samples    
        normal_start = normal_start + 1;
        normal_zone=[normal_start:normal_start+samples-1];
        myzone{1}=normal_zone;   
        normal_sd_original(:,normal_start)=std(altitude_matrix_databeihaidao1(:,myzone{1}),0,2); 
        for mys=1:1
            UU=altitude_matrix_databeihaidao1(:, myzone{mys});
            [input_dimensions, time_points]=size(UU);
            flat_y=zeros(time_points+embeddings_num-1,1);
            xx=UU;
            noisestrength=0;          % noise could be added
            xx_noise=xx+noisestrength*rand(size(xx));
            D=size(xx_noise,1);     % number of variables in the system.
            trainlength=time_points;        
            traindata=xx_noise(:,1:trainlength);  
            traindata_x_NN=NN_transform(traindata,all_weights);
            test_NN=traindata_x_NN;
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
            X_real=xx_noise(:,1:trainlength); 
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
            sd1(mys,:)=std(flat_y,0,2);
            test_flat_y(mys,:)=flat_y;
        end
        normal_sd_y(normal_start)=normal_sd_y(normal_start) + std(test_flat_y(1,1:end));    %灏嗘瘡涓?娆℃眰寰楃殑鏍囧噯宸疮鍔犺捣鏉?
        reps=reps+1;
    end
    if mod(ii,100)==0      
        ii
    end
end

normal_sd_y=normal_sd_y/ii;     
combined_sd_y_South_central_Alaska=[normal_sd_y'];
figure
plot(combined_sd_y_South_central_Alaska);  
save combined_sd_y_South_central_Alaska combined_sd_y_South_central_Alaska

