clc
clear
close all
load rmse_station_all_sc
rmse_station_all_use=rmse_station_all_sc;
load combined_sd_y_sc
ydata=[zeros(1,376-length(combined_sd_y_sc)) combined_sd_y_sc];
rmse_station_all_predi_4=[zeros(23,376-size(rmse_station_all_use,2)) rmse_station_all_use];
rmse_stationall=rmse_station_all_predi_4;

windows=18;
for j=1:1:size(rmse_stationall,1) 
      now_rmse_station=rmse_stationall(j,:);
    for i=14+1:length(now_rmse_station)
        long_pic=now_rmse_station(i-14:i-1);    
        now_num=now_rmse_station(i);    
        [h,p]= ttest(long_pic,now_num);
        loss_p(j,i)=p;
    end   
end
YYY=zeros(1,length(ydata));
for i=1:length(ydata)
    YY_pic=ydata(276:286);
    YY_num=ydata(i);
    YYY(i) = ttest(YY_pic,YY_num,'Alpha',0.01);
end
k=1;
clear mark1
for i=1:length(YYY)   
    if YYY(i)>0 
        mark1(k)=i;
        k=k+1;
    end
end 

load sichuan_earthquake_event
earthquake_matrix=sichuan_earthquake_event(1,:);
for i=1:length(earthquake_matrix)
    if earthquake_matrix(i)<5
        earthquake_matrix(i)=0;
    end
end
        
Y2=earthquake_matrix;
t_end=datetime(2019,1,11);
t_long=t_end-length(ydata)+1;
t1 = t_long+caldays(0:length(ydata)-1);


number=size(rmse_stationall,1);
mmm=sum(loss_p(1:number,:));
mmm=mmm./number;
for i=1:length(mmm)
    if mmm(i)~=0
        mmm(i)=1/mmm(i);
    end
end

k=1;
for i=1:length(mmm)
    if mmm(i)>20
        mark3(k)=i;
        k=k+1;
    end
end
k=1;
clear mark
earthquake_all=[];
accur_earthqu=[];
for i=1:length(mark1)   
    m=mark1(i);
    a=max(1,m-14);
    b=min(length(ydata),m);
    c=min(length(ydata),m+14);
    for j=1:length(mark3)
        if mark3(j)>=a && mark3(j)<=b
            mark(k)=m;
            k=k+1;
            break
        elseif mark3(j)>b && mark3(j)<=c
            mark(k)=mark3(j);
            k=k+1;
            break
        end
    end
end
mark = unique(mark);
single=zeros(1,length(ydata));
for i=1:length(mark)
    a=mark(i);
    single(a)=1;
end
starday=1;
earthquake=0;
k=1;
for i=starday:length(Y2)
    if Y2(i)~=0  
        a=max(1,i-7);
        b=min(length(ydata),i+1);
        c=min(length(ydata),i+7);
        for j=b:c
            if single(j)==1
                single(j)=0;
            end
            
        end
    end
end

%正确预测地震
acuur_single=zeros(1,length(single));
for i=starday:length(Y2)
    if Y2(i)~=0   %真实地震
        earthquake=earthquake+1;%记录地震的数量
        a=max(1,i-14);
        b=min(length(ydata),i);
        for j=a:b
            if single(j)==1 
                accur_earthqu(k)=i;  %正确预测的地震
                acuur_single(j)=1;
                k=k+1;
            end
        end
    end
end
accur_earthqu = unique(accur_earthqu);
k=1;
acuur_mark=[];
for uu=1:length(acuur_single)
    if acuur_single(uu)==1
        acuur_mark(k)=uu;
        k=k+1;
    end
end
TP=length(accur_earthqu)
FN=earthquake-TP
erro_single=single;
for i=1:length(accur_earthqu)
    mmm=accur_earthqu(i);
    a=max(1,mmm-14);
    b=min(length(ydata),mmm);
    for j=a:b
        if single(j)==1
            erro_single(j)=0;  %正确预测的地震
        end
    end
end
k=1;
for uu=1:length(erro_single)
    if erro_single(uu)==1
        erro_mark(k)=uu;
        k=k+1;
    end
end
FP=sum(erro_single(1,starday:end))
TN=length(ydata(starday:end))-earthquake-FP
Precision = TP/(TP+FP)
TPR = TP/(TP+FN)
FPR = FP/(FP+TN)
Specificity= TN/(FP+TN)
%= 1 - FPR 
Accuracy =(TP+TN)/(TP+TN+FP+FN)
error= (FP+FN)/(TP+TN+FP+FN)
for i=1:length(acuur_single)
    if acuur_single(i)~=0
        acuur_single(i)=4;
    end
end
for i=1:length(erro_single)
    if erro_single(i)~=0
        erro_single(i)=4;
    end
end
figure;
hold on
demo=zeros(1,376);
plot(t1(:,1:end),demo(:,1:end),'DatetimeTickFormat','M/d/yyyy','Color','k')
hold on
bar(t1(:,starday:end),acuur_single(:,starday:end),'FaceColor','r','BarWidth',4);
set(gca,'fontsize',40,'fontname','Times');
set(gca,'yTick',[-6 -3 0 4]) 
set(gca,'yticklabel',{'6','3','0','1'})
set(gca,'linewidth',6)
hold on
Y4=-Y2;
title('Sichuan, China (1/1/2018-1/11/2019)')
hold on
bar(t1(:,starday:end),erro_single(:,starday:end),'FaceColor',[ .50 .50 .50],'BarWidth',4);
for i=1:length(erro_mark)
    if erro_mark(i)>starday
        x=t1(erro_mark(i));
        y=4;
        hold on
        plot(x,y,'p','MarkerSize',40,'MarkerFaceColor',[ .50 .50 .50],'MarkerEdgeColor',[ .50 .50 .50],'LineWidth',2);
    end
end

for i=starday:length(Y4)
    if Y4(i)~=0
        hold on
     text(t1(i),Y4(i)-1.2,num2str(-Y4(i)),'VerticalAlignment','bottom','HorizontalAlignment','center','fontsize',20);

    end
end
hold on
Y4=-Y2;
bar(t1(:,starday:end),Y4(:,starday:end),'FaceColor','b','BarWidth',4);
for i=1:length(acuur_mark)
    if acuur_mark(i)>starday
        x=t1(acuur_mark(i));
        y=4;
        hold on
        plot(x,y,'p','MarkerSize',40,'MarkerFaceColor','r','MarkerEdgeColor','r','LineWidth',2);
    end
end
accuur_earthq_all=zeros(1,length(acuur_single));
for i=1:length(accur_earthqu)
    accuur_earthq_all(accur_earthqu(i))=5;
end
k=1;
for i=1:length(Y2)
    if Y2(i)>0
        earthquake_all(k)=i;
        k=k+1;
    end
end
set(gca,'xlim',datetime([2018 2019],[1 1],[1 11]),'ylim',[-10 9]);
AAAA(1)=TP;
AAAA(2)=FN ;
AAAA(3)=FP ;
AAAA(4)=TN ;
AAAA(5)=Precision ;
AAAA(6)=TPR ;
AAAA(7)=FPR ;
AAAA(8)=Specificity ;
AAAA(9)=Accuracy ;
AAAA(10)=error ;
warning_effect=AAAA;