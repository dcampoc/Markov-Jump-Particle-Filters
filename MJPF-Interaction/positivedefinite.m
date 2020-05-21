function[cov]=positivedefinite(cov2)%makes cov 4x4 positive definite
cov=cov2;%put equal to previous one
if(cov2(1,1)~=1e+100)%if elements aren't infinite
eigen=eig(cov);%find eigenvalues
while(eigen(1,1)<=0.001||eigen(2,1)<=0.001||eigen(3,1)<=0.001||eigen(4,1)<=0.001)
    cov=diag(eig(cov)+1e-3);%add a little value to make covariance positive definite
    eigen=eig(cov);
end
end
end