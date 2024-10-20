%% FCM 
%top 5 final code
clear all, close all, clc
%% Load data
opts = detectImportOptions('cafeSalesReceipts.csv', 'VariableNamingRule', 'preserve');
data = readtable('cafeSalesReceipts.csv', opts);

%% Data Preprocessing
% Remove rows with missing data
data = data(~any(ismissing(data), 2) & ~strcmp(data.gender, 'N'), :);

% Define the columns to keep (excluding store_telephone, store_longitude, and store_latitude)
columns_to_keep = setdiff(data.Properties.VariableNames, {'store_square_feet','store_telephone', 'store_longitude', 'store_latitude'});

% Create a new table with only the desired columns
data = data(:, columns_to_keep);

% Convert dates to datetime format
data.transaction_date = datetime(data.transaction_date, 'InputFormat', 'MM/dd/yyyy');
data.customer_since = datetime(data.customer_since, 'InputFormat', 'MM/dd/yyyy');
data.birthdate = datetime(data.birthdate, 'InputFormat', 'MM/dd/yyyy');
current_date = datetime('now');

% Change gender values: 'F' to 0 and 'M' to 1
data.gender(strcmp(data.gender, 'F')) = {'0'};
data.gender(strcmp(data.gender, 'M')) = {'1'};

% Create additional features
data.customer_age = year(data.transaction_date) - year(data.birthdate);
data.customer_loyalty = calmonths(between(data.customer_since, data.transaction_date, 'months'));
data.promo_item_yn_logical = strcmp(data.promo_item_yn, 'Y');

% Convert the gender column to numeric
data.gender = str2double(data.gender);

%% Analyze data (Customer)

% Initialize arrays to store results
unique_customer_ids = unique(data.customer_id);
num_customers = numel(unique_customer_ids);

customer_age_result = zeros(num_customers, 1);
loyalty_result = zeros(num_customers, 1);
purchase_frequency_result = zeros(num_customers, 1);
total_purchase_price_result = zeros(num_customers, 1);

% Loop through each unique customer ID
for i = 1:num_customers
    customer_id = unique_customer_ids(i);
    
    % Filter data for current customer ID
    customer_data = data(data.customer_id == customer_id, :);
    
    % Calculate customer age and loyalty (assuming they are the same for all rows of the customer)
    customer_age_result(i) = data.customer_age(find(data.customer_id == customer_id, 1));
    loyalty_result(i) = data.customer_loyalty(find(data.customer_id == customer_id, 1));
    
    % Calculate purchase frequency (number of unique transaction IDs)
    unique_transactions = unique(customer_data.transaction_id);
    purchase_frequency_result(i) = numel(unique_transactions);
    
    % Calculate total purchase price (sum of line item amounts)
    total_purchase_price_result(i) = sum(customer_data.line_item_amount);
end

% Create customer table
customer_table = table(unique_customer_ids, customer_age_result, loyalty_result, purchase_frequency_result, total_purchase_price_result, ...
    'VariableNames', {'customer_id', 'age', 'loyalty', 'purchase_frequency', 'total_purchase_price'});

%% Apply FCM
% Define the features for clustering
X = customer_table{:, {'total_purchase_price', 'age', 'purchase_frequency'}};

% Set the number of clusters to 3
optionsCust = fcmOptions('NumClusters', 3, MaxNumIteration=100, MinImprovement=0.0001,Verbose=false);

% Find cluster centers and membership values
[centersCust, U_customer] = fcm(X, optionsCust);

% Classify the data points into the cluster with the highest membership value
[~, maxU_customer] = max(U_customer);

% Add cluster labels to the customer table
customer_table.cluster = maxU_customer';
%customer_table.membership = max(U_customer);

% Display the customer table with cluster labels
% disp(customer_table);

% Save the customer table with cluster labels to a new file
%writetable(customer_table, 'customer_clusters.csv'); % Change 'customer_clusters.csv' to desired output file name

%% Plotting the results
figure;
hold on;
scatter3(X(:, 1), X(:, 2), X(:, 3), 50, customer_table.cluster, 'filled');
plot3(centersCust(:,1),centersCust(:,2),centersCust(:,3), ...
    "xk",MarkerSize=15,LineWidth=3)
xlabel('Total Purchase Amount');
ylabel('Age');
zlabel('Purchase Frequency');
view([-11 63]);
title('3D Scatter Plot of Customer Clusters');
grid on;
hold off;

%% Analyze cluster
% Display cluster characteristics
n_clustersCust = max(customer_table.cluster); % Number of clusters

for i = 1:n_clustersCust
    fprintf('Cluster Customer %d:\n', i);
    cluster_data = customer_table(customer_table.cluster == i, :);
    disp(varfun(@mean, cluster_data, 'InputVariables', {'total_purchase_price', 'purchase_frequency', 'age'}));
end

%% Develop Cross Selling Strategies
% Assume customer_summary has cluster information and data contains transaction details

% Initialize cell array to store cross-sell recommendations
cross_sell_recommendations = cell(n_clustersCust, 1);
cross_sell_top5 = cell(n_clustersCust, 1);

% Loop through each cluster
for i = 1:n_clustersCust
    % Filter customer IDs for current cluster
    cluster_customer_ids = customer_table.customer_id(customer_table.cluster == i);
    
    % Filter transaction data for current cluster customers
    cluster_data = data(ismember(data.customer_id, cluster_customer_ids), :);
    
    % Analyze data
    unique_product_ids = unique(data.product_id);
    num_product = numel(unique_product_ids);

    total_quantity_result = zeros(num_product, 1);
    transaction_frequency_result = zeros(num_product, 1);
    total_sales_amount_result = zeros(num_product, 1);
    average_quantity_per_transaction_result = zeros(num_product, 1);
    
    % Loop through each unique product ID
    for j = 1:num_product
        product_id = unique_product_ids(j);
    
        % Filter data for current product ID
        product_data = cluster_data(cluster_data.product_id == product_id, :);
        
        % Calculate total quantity (sum of quantity)
        total_quantity_result(j) = sum(product_data.quantity);

        % Calculate transaction frequency (number of unique transaction IDs)
        unique_transactions = unique(product_data.transaction_id);
        transaction_frequency_result(j) = numel(unique_transactions);
    
        average_quantity_per_transaction_result(j) = total_quantity_result(j) / transaction_frequency_result(j);
        % Calculate total purchase price (sum of line item amounts)
        total_sales_amount_result(j) = sum(product_data.line_item_amount);
    end

    % Create product table
    product_summary = table(unique_product_ids, total_quantity_result, transaction_frequency_result, total_sales_amount_result, average_quantity_per_transaction_result, ...
    'VariableNames', {'product_id', 'total_quantity', 'transaction_frequency', 'total_sales', 'average_quantity'});
    
    product_summary = rmmissing(product_summary);
    %disp(product_summary)

    % Define the features for clustering
    Y = product_summary{:, {'transaction_frequency', 'total_sales', 'average_quantity'}};

    % Set the number of clusters to 3
    optionsPro = fcmOptions('NumClusters', 3, MaxNumIteration=100,MinImprovement=0.0001,Verbose=false);

    % Find cluster centers and membership values
    [centersPro, U_product] = fcm(Y, optionsPro);

    % Classify the data points into the cluster with the highest membership value
    [~, maxU_product] = max(U_product);

    % Add cluster labels to the product table
    product_summary.cluster = maxU_product';
    product_summary.membership = max(U_product)'; % Add membership value to product summary

    % Display the product table with cluster labels
    %disp(product_summary);

    % Add product names to product summary
    [~, loc] = ismember(product_summary.product_id, data.product_id);
    product_summary.product_name = data.product(loc);

    % Plotting the results
    figure;
    hold on;
    scatter3(Y(:, 1), Y(:, 2), Y(:, 3), 50, product_summary.cluster, 'filled');
    plot3(centersPro(:,1),centersPro(:,2),centersPro(:,3), ...
    "xk",MarkerSize=15,LineWidth=3)
    xlabel('Transaction Frequency');
    zlabel('Average Quantity Per Transaction');
    ylabel('Total Sales');
    view([-11 63]);
    eval(['title(''3D Plot of Product Cluster for Customer Cluster ',num2str(i),''')'])
    grid on;
    hold off;

    n_clustersPro = max(product_summary.cluster);

    % Find all products in each product cluster and their membership values
    product_recommendations = cell(n_clustersPro, 1);
    top5_products_recommendation = cell(n_clustersPro,1);
    for product_cluster_idx = 1:n_clustersPro
        % Get all products in the current cluster
        cluster_products = product_summary(product_summary.cluster == product_cluster_idx, :);
        % Sort products by membership value in descending order
        sorted_products = sortrows(cluster_products, 'membership', 'descend');
        % Get top 5 products
        top_5_products = sorted_products(1:min(5, height(sorted_products)), :);
        top5_products_recommendation{product_cluster_idx} = top_5_products(:, {'product_name'});
        % Store the product names and their membership values as recommendations
        product_recommendations{product_cluster_idx} = sorted_products(:, {'product_name', 'membership'});
    end
    

    % Store the recommendations for the current customer cluster
    cross_sell_recommendations{i} = product_recommendations;
    cross_sell_top5{i} = top5_products_recommendation;
end

for i = 1:n_clustersPro
    fprintf('Cluster Product %d:\n', i);
    clusterCust_data = product_summary(product_summary.cluster == i, :);
    disp(varfun(@mean, clusterCust_data, 'InputVariables', {'transaction_frequency', 'total_sales', 'average_quantity'}));
end

% Display cross-sell recommendations for each cluster (top 5)
for i = 1:n_clustersCust
    fprintf('Customer Cluster %d\n', i);
    for product_cluster_idx = 1:n_clustersPro
        % Extract product names from the table
        product_names = cross_sell_top5{i}{product_cluster_idx}.product_name;
        % Convert product names to a cell array of strings
        product_names_str = cellstr(product_names);
        % Join the product names with a comma separator
        product_names_joined = strjoin(product_names_str, ', ');
        fprintf('  Product Cross Selling Recommendation %d: %s\n', product_cluster_idx, product_names_joined);
    end
end

% Store cross-sell recommendations in individual variables
for i = 1:n_clustersCust
    for j = 1:n_clustersPro
        eval(sprintf('crossSelling%d%d = cross_sell_recommendations{i}{j};', i, j));
    end
end

%% Calculate Silhouette Score for customer clusters
X = customer_table{:, {'total_purchase_price', 'age', 'purchase_frequency'}};
silhouette_customer = silhouette(X, customer_table.cluster);
mean_silhouette_score_customer = mean(silhouette_customer);
fprintf('Mean Silhouette Score for Customer Clusters: %.4f\n', mean_silhouette_score_customer);

%% Calculate Davies-Bouldin Index for customer clusters
davies_bouldin_index_customer = evalclusters(X, customer_table.cluster, 'DaviesBouldin');
fprintf('Davies-Bouldin Index for Customer Clusters: %.4f\n', davies_bouldin_index_customer.CriterionValues);

%% Calculate Silhouette Score for product clusters (Example for one customer cluster)
for i = 1:n_clustersCust
    cluster_customer_ids = customer_table.customer_id(customer_table.cluster == i);
    cluster_data = data(ismember(data.customer_id, cluster_customer_ids), :);
    %product_summary = ... % Your product summary code for this cluster
    Y = product_summary{:, {'transaction_frequency', 'total_sales', 'average_quantity'}};
    silhouette_product = silhouette(Y, product_summary.cluster);
    mean_silhouette_score_product = mean(silhouette_product);
    fprintf('Mean Silhouette Score for Product Clusters in Customer Cluster %d: %.4f\n', i, mean_silhouette_score_product);

    davies_bouldin_index_product = evalclusters(Y, product_summary.cluster, 'DaviesBouldin');
    fprintf('Davies-Bouldin Index for Product Clusters in Customer Cluster %d: %.4f\n', i, davies_bouldin_index_product.CriterionValues);
end
