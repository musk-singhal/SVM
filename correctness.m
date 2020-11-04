% %%%%%%%%%%%%%%% correctness calculation %%%%%%%%%%%%%%%%%%%%
function corr = correctness(AA,dd,AAA,ddd,alpha,bias,ker,mode)
    %mode=1: checking correctness of training set
    if mode==1
        n = size(AA,1);
        H = zeros(n,n);  
    for i=1:n
       for j=1:n
          H(i,j) = dd(i)*dd(j)*svkernel(ker,AA(i,:),AA(j,:),10);
       end
    end
    p = sign(H*alpha + bias);
    corr=length(find(p==dd))/size(AA,1)*100;
    return
    end
    %mode=2: checking correctness of testing set
    if mode==2
        m = size(AA,1); %length of test set=no. of rows of H
        n = size(AAA,1); %length of train set=no. of columns of H
        H = zeros(m,n); 
    for i=1:m
      for j=1:n
        H(i,j) = ddd(j)*svkernel(ker,AA(i,:),AAA(j,:),10);
      end
    end
     p = sign(H*alpha + bias);
     corr=length(find(p==dd))/size(AA,1)*100;
     return
     end
end
