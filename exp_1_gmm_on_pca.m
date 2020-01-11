clear;
%Author: Alper Þahýstan

rgb_images = dir('./dataset/files/*.png');
dept_images = dir('./dataset/files/*-depth.bin');
conf_images = dir('./dataset/files/*-conf.bin');
labels = readtable('./dataset/labels.csv');
labels = labels{:,2};

indices = randperm(length(rgb_images));
divide = 800;
train_indices = indices(1:divide);
test_indices = indices(divide +1:1320);
scale_to = [70,70];
features = zeros(1320, scale_to(1)*scale_to(2)*5);
k=0;
for i = 1:1320
    k=k+1;
    lb = 1;
    ub = scale_to(1)*scale_to(2)*3;
    [r,c]=size(features);
    disp(string(k) + '/' + string(r));
    RGB = fullfile(rgb_images(i).folder, rgb_images(i).name);
    RGB = imread(RGB);
    RGB = imresize(RGB, scale_to);
    
    features(k, lb:ub) = reshape(RGB, ub, 1);

    D = fullfile(dept_images(i).folder, dept_images(i).name);
    f_id = fopen(D);
    D = fread(f_id, [320, 240],'uint16=>uint8');
    D = D';
    D = imresize(D, scale_to);
    fclose(f_id);
    lb = ub+1;
    ub = ub + scale_to(1)*scale_to(2);
    
    features(k, lb:ub) = reshape(D, scale_to(1)*scale_to(2),1);

    C = fullfile(conf_images(i).folder, conf_images(i).name);
    f_id = fopen(C);
    C = fread(f_id, [320, 240],'uint16=>uint8');
    C = C';
    C = imresize(C, scale_to);
    fclose(f_id);
    lb = ub+1;
    ub = ub + scale_to(1)*scale_to(2);
    features(k, lb:ub) = reshape(C, scale_to(1)*scale_to(2), 1);
end
max_num_base = 120;
clear conf_images;
clear dept_images;
clear rgb_images;
features = normalize(features, 'range');

[coeff,score,latent,tsquared,explained,mu] = pca(features(train_indices, :), 'Algorithm', 'eig', 'Rows','complete');
%clear features;

cum_var = cumsum(explained);

figure;
plot(1:min([999,divide-1]), cum_var);
title('The number of components needed to explain variance');
xlabel('Number of Components');
ylabel('Cumulative variance (%)');


label_types = unique(labels);

disp('Gmm phase')
acc_data = zeros(max_num_base/2, 10);
cntr=1;

for i = 1:2:max_num_base
    for comp_count = 1:10
        GMModels = cell(1,length(label_types));
        options = statset('MaxIter',1100);
        data = score(find(labels(train_indices)==label_types(1)), 1:i);
        [n_r,n_c] = size(data);
        if (n_r<=n_c)
            data = [data;data(1:(n_c - n_r+1),:)];
        end
        % train gmm for each class
        for k = 1:length(label_types)
            GMModels{k} = fitgmdist(data, comp_count,...
                'RegularizationValue', 0.001, 'CovarianceType','full', 'Options', options);
        end
        
        p_test_data = features(test_indices,:) * (coeff(:, 1:i));
        likelihoods = zeros(1320-divide, length(label_types));
        %test accuracy
        for idx = 1:(1320-divide)
            for k = 1:10
                %likelihoods(idx,k) = min(mahal(GMModels{k}, p_test_data(idx,:)), [], 1);
                likelihoods(idx,k) = pdf(GMModels{k}, p_test_data(idx,:));
            end
        end
        
        [~ ,predictions] = max(likelihoods, [], 2);
        %[~ ,predictions] = min(likelihoods, [], 2);
        
        acc_data(cntr,comp_count) = sum(predictions == labels(test_indices))/length(test_indices);
    end
    cntr= cntr+1;
end
