clc;clear;close all;
%load('linsep.mat')
ker='linear';C=1;

%importing iris dataset
[Data,Col]=iris_dataset;
Data=Data';
Data=Data(:,1:4);
Col=vec2ind(Col)';

%Dimensionality reduction using PCA
[Covariance,PC,EigValue,EigVector]=pca(Data,'NumComponents',2);

%Creating imbalance in iris dataset
Cnew(1:50,1)=-1;
Cnew(51:150,1)=1;

X=PC;
Y=Cnew;
figure
gscatter(X(:,1),X(:,2),Y,[],[],[]);

%--------------------------------------------------------------------------------%

trainCorr = 0;
testCorr = 0;
% K-FOLD CROSS VALIDATION
% eg: K=5 MEANS 5-FOLD-CROSS-VALIDATION, K=10 MEANS 10-FOLD-CROSS-VALIDATION
k=10;

% if k=0 no correctness is calculated, just run the algorithm 
% ONLY TRAINING, NO TESTING
if k==0
  [nsv ,alpha ,bias] = svc(X,Y,ker,C,10);
cpu_time = toc;
end

%if k==1 only training set correctness is calculated
if k==1
  [nsv ,alpha ,bias] = svc(X,Y,ker,C,10);
  trainCorr = correctness(X,Y,X,Y,alpha,bias,ker,1);
  cpu_time = toc;
  
  fprintf(1,'\nTraining set correctness: %3.2f%% \n',trainCorr);
  fprintf(1,'\nElapse time: %10.2f\n',toc);
  
  %return
end


if k>1
    
[sm ,sn]=size(X);
accuIter = 0;
lastToc=0;    % used for calculating time
indx = [0:k];
indx = floor(sm*indx/k);    %last row numbers for all 'segments'

% split training set from test set
tic;
for i = 1:k
Ctest = []; dtest = [];Ctrain = []; dtrain = [];

Ctest = X((indx(i)+1:indx(i+1)),:);
dtest = Y(indx(i)+1:indx(i+1));

Ctrain = X(1:indx(i),:);
Ctrain = [Ctrain;X(indx(i+1)+1:sm,:)];
dtrain = [Y(1:indx(i));Y(indx(i+1)+1:sm,:)];


   [nsv ,alpha ,bias] = svc(Ctrain,dtrain,ker,C,10);
   
   tmpTrainCorr = correctness(Ctrain,dtrain,Ctrain,dtrain,alpha,bias,ker,1);
   trainCorr = trainCorr + tmpTrainCorr;
   tmpTestCorr = correctness(Ctest,dtest,Ctrain,dtrain,alpha,bias,ker,2);
   testCorr = testCorr + tmpTestCorr;

 
 fprintf(1,'________________________________________________\n');
 fprintf(1,'Fold %d\n',i);
 fprintf(1,'Training set correctness: %3.2f%%\n',tmpTrainCorr);
 fprintf(1,'Testing set correctness: %3.2f%%\n',tmpTestCorr);    
 fprintf(1,'Elapse time: %10.2f\n',toc);
 end


%FINAL AVERAGE PERFORMANCE OF THE MODEL ON K-FOLD CROSS VALIDATION
 trainCorr = trainCorr/k;
 testCorr = testCorr/k;
 cpu_time = toc/k;

  fprintf(1,'___________________________________________________\n');
  fprintf(1,'\nAVG. Training set correctness: %3.2f%% \n',trainCorr);
  fprintf(1,'\nAVG. Testing set correctness: %3.2f%% \n',testCorr);
  fprintf(1,'\nAVG. CPU time is: %3.2f \n',cpu_time);

end 
%-------------------------------------------------------------------------------------------%

a=min(X);
b=max(X);
[nsv ,alpha ,bias] = svc(X,Y,ker,C,10);
% GENERATING A TEST SET FOR MAKING PREDICTIONS ON THE ABOVE VALIDATED MODEL
predX=[(a(:,1)+(b(:,1)-a(:,1))*rand(20,1)),(a(:,2)+(b(:,2)-a(:,2))*rand(20,1))];
%PREDICTING THE LABELS OF TEST SET
predY = svcoutput(X,Y,predX,ker,10,alpha,bias,0);
%PLOTTING TRAIN AND TEST SET
figure
svcplot(X,Y,ker,alpha,bias,predX,predY) 







