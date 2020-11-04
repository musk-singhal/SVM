    function predictedY = svcoutput(trnX,trnY,tstX,ker,p1,alpha,bias,actfunc)
%SVCOUTPUT Calculate SVC Output
%
%  Usage: predictedY = svcoutput(trnX,trnY,tstX,ker,alpha,bias,actfunc)
%
%  Parameters: trnX   - Training inputs
%              trnY   - Training targets
%              tstX   - Test inputs
%              ker    - kernel function
%              beta   - Lagrange Multipliers
%              bias   - bias
%              actfunc- activation function (0(default) hard | 1 soft) 
%
%  Author: Steve Gunn (srg@ecs.soton.ac.uk)

  if (nargin < 6 | nargin > 8) % check correct number of arguments
    help svcoutput
  else

    if (nargin == 6)
      actfunc = 0;
    end
    n = size(trnX,1);
    m = size(tstX,1);
    H = zeros(m,n);  
    for i=1:m
      for j=1:n
        H(i,j) = trnY(j)*svkernel(ker,tstX(i,:),trnX(j,:),p1);
      end
    end
    %Z=H*alpha + bias;
%     Z1=trnY.*(H*alpha + bias);
%     Z2=1-Z1;
%     Z3=max(Z2,0);
%     idx=find(Z3<1);
%     X1=trnX(idx,:);
%     Y1=trnY(idx,:);
%     save 2moonslin.mat X1 Y1;
    
    
    if (actfunc)
      predictedY = softmargin(H*alpha + bias);
    else
      predictedY = sign(H*alpha + bias);
    end
    
    
  end
