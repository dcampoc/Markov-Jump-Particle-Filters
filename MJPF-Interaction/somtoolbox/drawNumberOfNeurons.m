function[] = drawNumberOfNeurons(c,m, n)
%input are number of vectors in each neuron, rows m, column n of lattice
% we plot in each neuron number of vectors in it
h = figure;
som_cplane('hexa', [m n],'none');%draw map
for u = 1:m*n%for all nodes
    ch = mod(u-1,(n))+1;%horizontal position
    cv = floor((u-1)/(n))+1;%vertical position
    if mod(cv,2) == 0
    shift1 = +.4;
    else
    shift1 = -.1;
    end
    text(ch+shift1,cv,num2str(c(u)) ,'FontSize',7);%number of input vectors
end
h.Position = [2507 189 756 696];
end