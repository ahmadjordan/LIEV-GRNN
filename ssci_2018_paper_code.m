clc
clear all
%%
%code written by Ahmad Al-Mahasneh
%email:ahmadalmahasneh@yahoo.com
%please cite the paper https://doi.org/10.1109/SSCI.2018.8628909

%%
%initilaization

steps=1000;
y=zeros(1,steps);
yd=y;
yg=y;
dt=0.05;
t=0;
net1=newgrnn(1,1,0.1);

%%
%%hyper-parameters set by the user

Mse_thresold=0.001;% the MSE threshold to start the evolution
size_limit=10;% the maximum allowed size of GRNN's hidden layer

%the evolving method either input distance or output distance

method='input_distance';
%method='output_distance';

%check if mistakenly you did not set the method then set it to the input
%distance as default method
if exist('method','var')==0
    method='input_distance';
end



%%
%add some noise
load noise
%%
%%the main loop

for k=2:steps
    
    yd(k)=0.2*sin(0.5*t)+0.5*sin(0.3*t);
    
    %  noise(k)=0.05*randn;
    
    sig(k)=var(mat2gray(yd));%%estimate sigma
    
    %%the bencmark system
    y(k+1)=y(k)*y(k-1)*(2.5+y(k-1))/(1+y(k).^2+y(k-1).^2)+2*(exp(-yd(k))-1/(exp(-yd(k))+1))+noise(k);%example3
    
    
    %%predict the output using the current data you have
    yg(k+1)=net1([yd(k)]);
    
    %evaluate your current prediction
    
    e(k)=mse(yg(k+1),y(k+1));
    
    %% if MSE > threshold (0.001)
    if(e(k)>Mse_thresold)
        
        
        %% limited to size_limit nodes always
        if k<=size_limit
            %evolve GRNN
            net1=newgrnn(yd(1:k),y(1:k),sig(k));
            
        end
        
        tic
        %%here is the pruning part
        if(strcmp(method,'input_distance'))
            
            net1=d_ev_in(net1,yd(k),y(k));
            
        end
        
        if(strcmp(method,'output_distance'))
            
            net1=d_ev_out(net1,yd(k),y(k));
            
        end
        
        time_r=toc;
        
        %predict your output after pruning
        yg(k+1)=net1(yd(k));
        
        %find the network size
        drawnow;
        
        % plot(sig(1:k));
        % plot(e_a_pruning(1:k));
        % hold on
        % plot(e_b_pruning(1:k));
    end
    
    
    tt(k+1)=t;
    
    yg(k+1)=net1([yd(k)]);
    e(k)=mse(yg(k+1),y(k+1));
    t=t+dt;
end

%%
%Visualization part
figure(1)

plot(tt,yg,'-.','LineWidth',1.5);
hold on
plot(tt,y,'LineWidth',1.5);
legend('Estimated output','Actual output');
xlabel('Times(sec)');
ylabel('y');
title('LEIV-GRNN');
set(gca,'fontsize',15)

figure(2)
plot(tt(2:end),sig,'LineWidth',1.5);
xlabel('Times(sec)');
ylabel('\sigma');
title('Sigma \sigma adapation');
set(gca,'fontsize',15)



figure(3)

plot(tt(2:end),e,'LineWidth',2);
xlabel('Times(sec)');
ylabel('MSE');
title('MSE limited Incremmental evolution of GRNN');
set(gca,'fontsize',15)
ylim([-1 8]);

figure(4)


plot(tt(2:end),noise,'LineWidth',1.5);
xlabel('Times(sec)');
ylabel('Noise');
title('Gaussian noise with zero mean');
set(gca,'fontsize',15)

RMSE=sqrt(mse(y,yg));

function netn = d_ev_in(net1,yd,y)

% input distance evolution

din=dist(net1.IW{1},yd);

[~,dd2]=max(din);

net1.IW{1}(dd2)=yd;
net1.LW{2,1}(dd2)=y;
netn=net1;
end

function netn = d_ev_out(net1,yd,y)

% output distance evolution

dout=dist(net1.LW{2,1},y);

[~,dd3]=max(dout);

net1.IW{1}(dd3)=yd;
net1.LW{2,1}(dd3)=y;
netn=net1;

end
