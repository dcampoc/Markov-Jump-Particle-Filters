function [] = unifiedMatrix(m)
%input is SOM struct sm
h = figure;
colormapigray = ones(64,3) - colormap('gray');
colormap(colormapigray);
Um = som_umat(m);
som_cplane('hexaU', m.topol.msize, Um(:));%draw umatrix
h.Position = [2507 189 756 696];
end