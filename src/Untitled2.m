avg = mean(fullDataSet,1); 
stdNum = std(fullDataSet,1);
for i=1:size(fullDataSet,1)
    for j=1:size(fullDataSet,2)
        featStd(i,j)=(fullDataSet(i,j) - avg(1,j))/stdNum(1,j);
    end
end


covarianceMatrix = cov(featStd);
fprintf('Covariance Matrix:\n')
disp(covarianceMatrix)
[V,D] = eig(covarianceMatrix);
fprintf('Eigenvectors:\n')
disp(V)
fprintf('Eigenvalues:\n')
disp(D)
sortedEigenVector=zeros(size(fullDataSet,2),size(fullDataSet,2));
[sor,index] = sort(diag(D),'descend');


for i=1:3
    sortedEigenVector(:,i) = V(:,index(i));
end
projectedData = zeros(size(fullDataSet,1),3);
for i = 1:size(fullDataSet,1)
    projectedData(i,:) = featStd(i,:)*sortedEigenVector(:,1:3);
end

figure
scatter3(projectedData(1:385,1),projectedData(1:385,2),projectedData(1:385,3),'filled','MarkerFaceColor','r','MarkerEdgeColor','r');
hold on
scatter3(projectedData(400:800,1),projectedData(400:800,2),projectedData(400:800,3),'filled','MarkerFaceColor','g','MarkerEdgeColor','g');
hold on
scatter3(projectedData(900:1200,1),projectedData(900:1200,2),projectedData(900:1200,3),'filled','MarkerFaceColor','b','MarkerEdgeColor','b');
hold on
scatter3(projectedData(1300:1600,1),projectedData(1300:1600,2),projectedData(1300:1600,3),'filled','MarkerFaceColor','m','MarkerEdgeColor','m');
hold on
scatter3(projectedData(1700:2000,1),projectedData(1700:2000,2),projectedData(1700:2000,3),'filled','MarkerFaceColor','k','MarkerEdgeColor','k');
hold on
axis equal
legend('LR','RR','LW','RW','FI')
xlabel('First Principal Component','FontSize',18)
ylabel('Second Principal Component','FontSize',18)
zlabel('Third Principal Component','FontSize',18)
title('Data distribution on feature space projected to the first three principal conponents','FontSize',18)
