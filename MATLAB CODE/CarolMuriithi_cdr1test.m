%% Read data
clc; clear all; close all;

data = tdfread('mouse_train_500.txt','tab');
data1 = tdfread('mouse_test_500.txt','tab');
%data = tdfread('mouse_train.tsv');
%data1 = tdfread('mouse_test.tsv');

fields = fieldnames(data);    % column names
fields1 = fieldnames(data1);

field_len = numel(fields);    % number of columns
field_len1 = numel(fields1);

n_train = 500;   % number of rows
n_test = 500;

% clean the data
%istart=29;
%iend=55;
%n_col = iend -istart; % number of columns selected from the data
%data_selected = data(:,istart:iend);
%data_selected(1,:)
for i = 1:field_len   % for train data
    temp = double(data.(fields{i}));  % change to ASCII numbers
    temp(isnan(temp)) = 0;    % set missing data to 0
    data.(fields{i}) = temp;
end
%data.(fields{29})
for i = 1:field_len1   % for test data
    temp1 = double(data1.(fields{i}));  % change to ASCII numbers
    temp1(isnan(temp1)) = 0;    % set missing data to 0
    data1.(fields{i}) = temp1;
end

%Read feature table 
%features = xlsread("Features_Table_New.xlsx") % read excel file
features = readtable("Features_Table_New.xlsx");  % read the feature table
[n_letters, n_features] = size(features);
n_features = n_features - 2;  % exclude the first column
%features(2,2)
%features_num = cell2mat(table2array(features(2:20,:)));  % convert table to array

tmp = cell2mat(table2array(features(:,2)));
letters = double(tmp);   % convert letters to ASCII numbers; in letters
%farray = table2array(features(:,3:8))
features_num = table2array(features(:,3:8));  % convert table to array


%% select features
istart=29;   % H22
iend=38; %55;     % H36
n_col = iend -istart;
nx_col = n_col*n_features;  % number of columns in x matrix

%%Generating x_train and y_train and also acoounting for missing data in
%%dataset by assigning 0 to it
% train data
x_train = zeros(n_train,nx_col);
y_train = zeros(n_train,1);

% test data
x_test = zeros(n_test,nx_col);
y_test = zeros(n_test,1);

%% Feature Engineering 
%Extracting CDrH1 sequence and then calculating corresponding hydropathy,mol_weight,pI,soulbility,pka of Amino Acids
% x_train structure: A row corresponds to an antibody acid. 
% The columns are the features (...) from H22 to H36. 

for j = 1:n_train  % loop of rows of data
    
       %if data.H22(j) == "C"  && data.H36(j) == "W"  % condition 1 
       if data.H22(j) ==  double('C')  && data.H36(j) ==  double('W') % condition 1 
          y_train(j) = 1;    % CDR-H1
       end 
       
       % fill x_train
       for k=1:n_letters  % loop of letters exctated from feature table
         for i=istart:iend  % loops between H22 and H36   
           if data.(fields{i})(j) == letters(k)   % compare the data to the 20 letters of the amino acids
             idx = 0 ;
             for m = 1:n_features  % loops of features
                 %idx = (i-istart)*n_features + m;  % index of columns of x_train
                 idx = (m-1)*n_col + i - istart + 1; 
                 x_train(j,idx) = features_num(k,m);  % assign x_train
             end
           end
         end
       end
end
x_train(1,1)   % check the values of x_train
%x_train(1,2)
%x_train(1,7)
%x_train(1,8)
%x_train(2,1)


for j = 1:n_test
    
       if data.H22(j) ==  double('C')  && data.H36(j) ==  double('W') % condition 1 
          y_test(j) = 1;    % CDR-H1
       end 
       
       % fill x_test
       for k=1:n_letters  % loop of letters exctated from feature table
         for i=istart:iend  % loops between H22 and H36   
           if data.(fields{i})(j) == letters(k)   % compare the data to the 20 letters of the amino acids
             idx = 0 ;
             for m = 1:n_features  % loops of features
                 %idx = (i-istart)*n_features + m;  % index of columns of x_train
                 idx = (m-1)*n_col + i - istart + 1; 
                 x_test(j,idx) = features_num(k,m);  % assign x_train
             end
           end
         end
       end
       
end
x_test(1,1)

%% creating pattern matrix
%patt_mat = table(hydropathy,solubility,pka,pI,mol_weight);
%patt_mat_up = table2array(patt_mat);

%x_test = table(hydropathy1,solubility1,pka1,pI1,mol_weight1);
%x_test = table2array(x_test);    

y_train = categorical(y_train);

beta = mnrfit(x_train, y_train);    % logistic regression
y_logit_prob = mnrval(beta, x_test);

[~, y_logit_pred] = max(y_logit_prob');  


%% Post-processing logistic regression model new
cp = classperf(y_test);
cp = classperf(cp, y_logit_pred);

modelAccuracy = cp.CorrectRate; % Model accuracy 
fprintf('Model accuracy = %0.3f\n', modelAccuracy); 

modelSensitivity = cp.Sensitivity; % Model sensitivity 
fprintf('Model sensitivity = %0.3f\n', modelSensitivity);

modelSpecificity = cp.Specificity; % Model specificity 
fprintf('Model specificity = %0.3f\n', modelSpecificity);

%% Estimating area under curve new
[X, Y, ~, AUC] = perfcurve(y_test, y_logit_prob(:,1), 1); % This command generates the outputs to plot the ROC curve 
fprintf('Model AUC = %0.3f\n', AUC); 

%% Plotting the ROC curve new
figure; plot(X, Y,'b-','LineWidth',2); 
title('ROC curve for logistic regression ','FontSize',14,'FontWeight','bold');

xlabel('False positive rate','FontSize',14,'FontWeight','bold'); 
ylabel('True positive rate','FontSize',14,'FontWeight','bold'); 

set(gca,'FontWeight','bold','FontSize',14,'LineWidth',2);


        
        
            
       
        
        

       

