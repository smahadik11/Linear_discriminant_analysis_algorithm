
cd('C:\Users\Sudha\Documents\MATLAB\csc872\FP3\Male')
d = dir('*.TIF');
Xn = [];

for i=1:length(d)
   c_fname=d(i).name;
   c_image = imread(c_fname);
   images{i} = c_image(:);
   Xn = horzcat(Xn, images{i});
end

Xn = double(Xn);


cd('C:\Users\Sudha\Documents\MATLAB\csc872\FP3\Female')
d = dir('*.TIF');
Xp = [];

for i=1:length(d)
   c_fname=d(i).name;
   c_image = imread(c_fname);
   images{i} = c_image(:);
   Xp = horzcat(Xp, images{i});
end

Xp = double(Xp);

 
X = horzcat(Xp, Xn);

%Get eigenvectors and eigenvalues of X
S = cov(X');
[V D] = eig(S);
d = diag(D);

[numc, order]=sort(d,'descend');
V2 = V(:, order);

t_var = sum(d);

add_sum(1) = numc(1);
frac(1) = numc(1)/t_var;
for i=2:length(numc)
   add_sum(i) = numc(i) + add_sum(i-1);
   frac(i) = add_sum(i)/t_var;
end

W = V2(:, 1:30); 

m = mean(X, 2); %Mean of X

Pn = [];
for i=1:size(Xn, 2)
   Ptn = W'*(Xn(:, i)-m); 
   Pn = horzcat(Pn, Ptn);
end

Pp = [];
for i=1:size(Xp, 2)
   Ptp = W'*(Xp(:, i)-m); 
   Pp = horzcat(Pp, Ptp);
end

mn = mean(Pn, 2); %Mean of projected male images
mp = mean(Pp, 2); %Mean of projected female images

P = horzcat(Pp, Pn); %All projected images
mall = mean(P, 2); %Mean of all projected images

%Compute Sw and Sb for LDA
Sw = 0;
Sb = 0;
for i=1:size(Pn, 2)
    Sw = Sw + (Pn(:,i) - mn)*(Pn(:,i) - mn)';
end
for i=1:size(Pp, 2)
    Sw = Sw + (Pp(:,i) - mp)*(Pp(:,i) - mp)';
end
Sb = Sb + size(Pn, 2)*(mn - mall)*(mn - mall)';
Sb = Sb + size(Pp, 2)*(mp - mall)*(mp - mall)';

%Get eigenvectors and eigenvalues for LDA
[V_l D_l] = eig(Sb,Sw);
d_l = diag(D_l);

[c_lda, order_lda]=sort(d_l,'descend');
V2_l = V_l(:, order_lda);

W_lda = V2_l(:, 1); % Top eigenvector

%Compute slope and intercept
w = W_lda'*W';
dintcp = W_lda'*((mp + mn)/2);

%Read male images for testing and classification
cd('C:\Users\Sudha\Documents\MATLAB\csc872\FP3\Male')

fn = dir('*.TIF');
Xdn = [];
An = [];
n = 1;
figure;
for i=1:length(fn)
   c_fname=fn(i).name;
   c_image = imread(c_fname);
   imagesfn{i} = c_image(:);
   Xdn = double(imagesfn{i});
   Ann = sign(w*(Xdn-m)-dintcp); %Classification 
   An = horzcat(An, Ann);
   if Ann ~= -1 %Plot in case not classified correctly
       subplot(3,3,n);
       colormap('gray');
       imagesc(c_image);
       n = n + 1;
   end
end

%Read female images for testing and classification
cd('C:\Users\Sudha\Documents\MATLAB\csc872\FP3\Female')

fp = dir('*.TIF');
Xdp = [];
Ap = [];
p = 1;
figure;
for i=1:length(fp)
   c_fname=fp(i).name;
   c_image = imread(c_fname);
   imagesfp{i} = c_image(:);
   Xdp = double(imagesfp{i});
   App = sign(w*(Xdp-m)-dintcp); %Classification 
   Ap = horzcat(Ap, App);
   if App ~= 1 %Plot in case not classified correctly
       subplot(3,3,p);
       colormap('gray');
       imagesc(c_image);
       p = p + 1;
   end
end




