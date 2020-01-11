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
C = 100;
clear conf_images;
clear dept_images;
clear rgb_images;
features = normalize(features, 'range');

disp('pca');
[coeff,score,latent,tsquared,explained,mu] = pca(features(train_indices, :), 'Algorithm', 'eig', 'Rows','complete');

disp('NN');
test_accs = zeros(max_num_base/2, 10);
f=1;
for j = 1:2:max_num_base
    p_test_data = features(test_indices,:) * (coeff(:, 1:j));
    inputs = [score(:,1:j)];
    targets = [labels(train_indices)];

    %turn labels into one-hot-encoding
    unq_lbls = unique(targets);
    temp = zeros(length(unq_lbls), length(targets));

    for i=1:length(targets)
        temp(targets(i), i) = 1; 
    end
    targets = temp;
    
    for hiddenLayerSize = 1:10
        % Create a Pattern Recognition Network
        %hiddenLayerSize = 4;
        net = patternnet(hiddenLayerSize);

        % Set up Division of Data for Training, Validation, Testing
        net.divideParam.trainRatio = 0.7;
        net.divideParam.valRatio = 0.3;
        net.divideParam.testRatio =0.0;

        net.trainParam.goal = 10^-25;
        net.trainParam.lr=0.01;

        % Train the Network
        [net,tr] = train(net, inputs',targets);

        % Test the Network
        outputs = net(p_test_data');
        [~,predictions] = max(outputs, [], 1);
%         errors = gsubtract(targets,outputs);
%         performance = perform(net,targets,outputs);
        test_accs(f, hiddenLayerSize) = sum(predictions == labels(test_indices)')/length(predictions);
    end
    f = f+1;
end


