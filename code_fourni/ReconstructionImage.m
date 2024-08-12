%%  Application de la SVD : compression d'images

clear all
close all

% Lecture de l'image
I = imread('BD_Asterix_1.png');
I = rgb2gray(I);
I = double(I);

[q, p] = size(I)

% Décomposition par SVD
fprintf('Décomposition en valeurs singulières\n')
tic
[U, S, V] = svd(I);
toc

l = min(p,q);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% On choisit de ne considérer que 200 vecteurs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% vecteur pour stocker la différence entre l'image et l'image reconstruite
inter = 1:40:(200+40);
inter(end) = 200;
differenceSVD = zeros(size(inter,2), 1);

skip = 1;

if ~skip

% images reconstruites en utilisant de 1 à 200 vecteurs (avec un pas de 40)
ti = 0;
td = 0;
for k = inter

    % Calcul de l'image de rang k
    Im_k = U(:, 1:k)*S(1:k, 1:k)*V(:, 1:k)';

    % Affichage de l'image reconstruite
    ti = ti+1;
    figure(ti)
    colormap('gray')
    imagesc(Im_k), axis equal
    
    % Calcul de la différence entre les 2 images
    td = td + 1;
    differenceSVD(td) = sqrt(sum(sum((I-Im_k).^2)));
    pause
end

% Figure des différences entre image réelle et image reconstruite
ti = ti+1;
figure(ti)
hold on 
plot(inter, differenceSVD, 'rx')
ylabel('RMSE')
xlabel('rank k')
pause

end

% Plugger les différentes méthodes : eig, puissance itérée et les 4 versions de la "subspace iteration method" 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% QUELQUES VALEURS PAR DÉFAUT DE PARAMÈTRES, 
% VALEURS QUE VOUS POUVEZ/DEVEZ FAIRE ÉVOLUER
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% tolérance
eps = 1e-8;
% nombre d'itérations max pour atteindre la convergence
maxit = 1000000;

% taille de l'espace de recherche (m)
search_space = 202;

% pourcentage que l'on se fixe
percentage = 0.995;

% p pour les versions 2 et 3 (attention p déjà utilisé comme taille)
puiss = 6;

%%%%%%%%%%%%%
% À COMPLÉTER
%%%%%%%%%%%%%

close all;

%%
% calcul des couples propres
%%
% TODO
Itournee = p>q;
if Itournee
    I = I';
    [q,p]=size(I);
end
M=I'*I;

% vecteur pour stocker la différence entre l'image et l'image reconstruite
inter = 1:40:(200+40);
inter(end) = 200;
differenceSVD = zeros(size(inter,2), 6);
for v=1:1:5
    switch v
        case 0
            [V, D] = eig(M);
            n_ev = p;
        case 1
            [V, D, n_ev, itv, flag] = power_v12(M, search_space, percentage, eps, maxit);
            % A m percentage eps maxit
        case 2
            [ V, D, it, flag ] = subspace_iter_v0(M, search_space, eps, maxit);
            if flag > -3
                n_ev = size(D,1);
            end
        case 3
            [ V, D, n_ev, it, itv, flag ] = subspace_iter_v1(M, search_space, percentage, eps, maxit);
        case 4
            [ V, D, n_ev, it, itv, flag ] = subspace_iter_v2(M, search_space, percentage, puiss, eps, maxit);
        case 5
            [ V, D, n_ev, it, itv, flag ] = subspace_iter_v3(M, search_space, percentage, puiss, eps, maxit);
    end
    if flag > -3
    %%
    % calcul des valeurs singulières
    %%
    % TODO
    [S_diag,ind] = sort(diag(sqrt(D)),'descend');
    S = diag(S_diag);
    
    S = [S; zeros(q-n_ev, size(S,2))];
    S = [S zeros(q, p-n_ev)];

    %%
    % calcul de l'autre ensemble de vecteurs
    %%
    % TODO

    V = V(:,ind);
    V = [V zeros(p, p-n_ev)];
    U = zeros(q,q);

    for i = 1:p
        U(:,i) = I*V(:,i)/S(i,i);
    end
    
    %%
    % calcul des meilleures approximations de rang faible
    %%
    
    % images reconstruites en utilisant de 1 à 200 vecteurs (avec un pas de 40)
    ti = 0;
    td = 0;
    
    for k = inter
        k
        % TODO
        % Calcul de l'image de rang k
        if k < n_ev
            Im_k = U(:, 1:k)*S(1:k, 1:k)*V(:, 1:k)';

            % Affichage de l'image reconstruite
            ti = ti+1;
            ti
            figure(ti)
            %title("k = " + k)
            colormap('gray')
            if Itournee
                imagesc(Im_k'), axis equal
            else
                imagesc(Im_k), axis equal
            end
            % Calcul de la différence entre les 2 images
            td = td + 1;
            differenceSVD(td,v+1) = sqrt(sum(sum((I-Im_k).^2)));
            pause
        end
    end
    flag = 0;
    end
end

% Figure des différences entre image réelle et image reconstruite
%ti = ti+1;
figure(1000)
hold on 
plot(inter, differenceSVD, 'rx')
ylabel('RMSE')
xlabel('rank k')
