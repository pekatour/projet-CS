clear all; close all;
figure;
n=1000;
for imat = 1:4
    genere = 0;
    v = 10;
    [W, V, flag, q, qv] = eigen_2024(imat, n, v, [], [], [], [], [], genere);
    semilogy(sort(W)); hold on;
end
hold off;
title(" Distribution des vaps en fonction de imat avec n = " + n);
legend("imat : 1","imat : 2","imat : 3","imat : 4");
xlabel("Valeurs propres");

