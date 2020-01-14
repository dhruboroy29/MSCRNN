function bargraph_recall_k(filename)

% filename='/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/Research/Deep_Learning_Radar/Displacement_Detection/EMI_RNN_Results/BumbleBee/recall/recall_bumblebee_512.txt';
m=dlmread(filename);

% Plot bar graph
bar(m(:,2:end));

h = gca;
if size(m,2)==3
    legend({'Noise','Target'},'interpreter','latex', 'FontSize', 20);
    ylim([0.8 1.0]);
elseif size(m,2)==4
    legend({'Noise','Human','Non-human'},'interpreter','latex', 'FontSize', 20);
    ylim([0.5 1.0]);
end

h.XLabel.String = 'k';
h.YLabel.String = 'Recall';
h.XLabel.Interpreter='latex';
h.YLabel.Interpreter='latex';
h.XLabel.FontSize = 20;
h.YLabel.FontSize = 20;
h.FontSize = 20;
h.TickLabelInterpreter='latex';

saveas(h,strrep(filename,'.txt','.eps'),'eps2c');