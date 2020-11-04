clc;clear;close all;
load('linsep.mat')
ker='linear';C=1;

[nsv ,alpha ,bias] = svc(X,Y,ker,C,10);

a=min(X);
b=max(X);

tstX=[(a(:,1)+(b(:,1)-a(:,1))*rand(10,1)),(a(:,2)+(b(:,2)-a(:,2))*rand(10,1))];

predY = svcoutput(X,Y,tstX,ker,p1,alpha,bias,0);

svcplot(X,Y,ker,alpha,bias,tstX,predY)

