% Read data
%data = readtable('mouse_test.tsv', 'TextType','string');
data1 = tdfread('mouse_train.tsv')
data = tdfread('mouse_test.tsv')

fields = fieldnames(data);
%rows = size(data.H1);
n_train = 51600;
n_test = 22114;
cols = numel(fields);
%data.H22(3)
%data.(fields{29:93})


% ==============  Scheme 1: supervised learning, SVM  ================
% label the train data
y_train = zeros(n_train,1);
nc=149-29+1;
x_train = zeros(n_train,nc);

for i=1:1:n_train
    if data.H22(i) == "C"
        if data.H36(i) == "W"  % condition 1 
          y_train(i) = 1;    % CDR-H1
          for j=1:nc
              x_train(i,j) = data1.(fields{29+j})(i);
          end
          if data1.H45(i) == "L" && data1.H46(i) == "E" && data1.H47(i) == "W" && data1.H48(i) == "I" && data1.H49(i) == "G"  % condition 2: L, E, W, I, G
             y_train(i) = 2;    % CDR-H1 & CDR-H2
             if data1.H92(i) == "L" & data1.H93(i) == "A" & data1.H94(i) == "R"  % C, A, R    
                y_train(i) = 3;    % CDR-H1 & CDR-H2 & condition 3
             end
           end
        end
     end
end
y_train=categorical(y_train);

% label the test data
y_test = zeros(n_test,1);
nc=149-29+1;
x_test = zeros(n_test,nc);

for i=1:1:n_test
    %data.H22(i)
    %data.(fields{29})(i)
    if data.H22(i) == "C"
        %data.H36(i)
        if data.H36(i) == "W"  % condition 1 
          y_test(i) = 1;    % CDR-H1
          %x_test(i,1:nc) = data.(fields{29:149})(i);
          for j=1:nc
              x_test(i,j) = data.(fields{29+j})(i);
          end
          if data.H45(i) == "L" && data.H46(i) == "E" && data.H47(i) == "W" && data.H48(i) == "I" && data.H49(i) == "G"  % condition 2: L, E, W, I, G
             y_test(i) = 2;    % CDR-H1 & CDR-H2
             if data.H92(i) == "L" & data.H93(i) == "A" & data.H94(i) == "R"  % C, A, R    
                y_test(i) = 3;    % CDR-H1 & CDR-H2 & condition 3
             end
           end
        end
     end
end

%  % Predict on test data
%acc_svm = sum(y_svm_pred == y_test)/numel(y_test)  % Prediction accuracy

% SVM for train, test and evaluation
model = fitcecoc(x_train, y_train,'Learners','linear');
y_svm_pred = predict(model, x_test);  % Predict on test data
y_svm_pred1 = double(y_svm_pred);
%acc_svm = sum(y_svm_pred == y_test)/numel(y_test)


% Post processing k-NN model

cp = classperf(y_test);
cp = classperf(cp, y_svm_pred1);

modelAccuracy = cp.CorrectRate; % Model accuracy 
fprintf('SVM model accuracy = %0.3f\n', modelAccuracy); 

modelSensitivity = cp.Sensitivity; % Model sensitivity 
fprintf('SVM model sensitivity = %0.3f\n', modelSensitivity);

modelSpecificity = cp.Specificity; % Model specificity 
fprintf('SVM model specificity = %0.3f\n', modelSpecificity);
%acc_svm = sum(y_svm_pred == y_test)/numel(y_test)  % Prediction accuracy

%y_test=categorical(y_test);

%y_test = (y_test_raw != 0);
%save(y_test)

% Optional: PCA: x_train, y_train, x_test, y_test ---> new x_train, y_train, x_test, y_test


% Train: logistic regression
%[beta,dev,stats] = mnrfit(x_train, y_train);

% Test: predict the result
%y_logit_prob = mnrval(beta, x_test);
%[~, y_logit_pred] = max(y_logit_prob');

% Evaluation:
%y_test_num = grp2idx(y_test);   % Convert categories to numbers
%n_%correct = 0;
%for i=1:1:n_test
    %if y_logit_pred(i) == y_test(i)
       %n_correct = n_correct + 1; % Count correct events 
    %end
%end
%ac%c_logit = n_correct / n_test  % Prediction accuracy


% Do SVM for train, test and evaluation
%model = fitcecoc(x_train, y_train,'Learners','linear');
%y_svm_pred = predict(model, x_test);  % Predict on test data
%acc_svm = sum(y_svm_pred == y_test)/numel(y_test)  % Prediction accuracy




% Further notes:
%y = categorical(output.Prediction);  % if this is done, you do not need to assign a number to a letter

