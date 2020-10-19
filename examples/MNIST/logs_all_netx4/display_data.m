fid0 = py.open('times.pkl','rb');
data0 = cell(py.pickle.load(fid0));
facetv = sort(double(data0{1}))+1;
nnv = sort(double(data0{2}))+1;
bakcav = sort(double(data0{3}))+1;


figure()
x = 1:100;
% semilogy(x,facetv,x,reluval,x,mara,x,bakcav, 'LineWidth',1)
semilogy(x,facetv,1:length(nnv),nnv,x,bakcav, 'LineWidth',1)
hold on
% semilogy(x(1:length(eran)),eran,  'LineWidth',1)
grid on
yticks([1 10])
ylim([0 20])
legend('Our method','NNV','nnenum')
xlabel('Number of instances verified')
ylabel('Time(sec)')
yticklabels({0,10})
hold off 

% figure()
% x = 1:180;
% semilogy(x,facetv,x,reluval_2h,'LineWidth',1)
% grid on
% yticks([1 10 100,1000,4000])
% ylim([1 4000])
% legend('Our method','ReluVal')
% xlabel('Number of instances verified')
% ylabel('Time(sec)')
% yticklabels({0,1,100,1000,4000})