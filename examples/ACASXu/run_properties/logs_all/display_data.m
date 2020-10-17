fid0 = py.open('times.pkl','rb');
data0 = cell(py.pickle.load(fid0));
facetv = sort(double(data0{1}))+1;
mara = sort(double(data0{2}))+1;
reluval = sort(double(data0{3}))+1;
bakcav = sort(double(data0{4}))+1;


figure()
x = 1:180;
semilogy(x,facetv,x,reluval,x,mara,x,bakcav,'LineWidth',1)
grid on
yticks([1 10 100])
ylim([1 120])
legend('Our method','ReluVal','Marabou','nnenum')
xlabel('Number of instances verified')
ylabel('Time(sec)')
yticklabels({0,1,100})

figure()
fid0 = py.open('times.pkl','rb');
data0 = cell(py.pickle.load(fid0));
facetv = sort(double(data0{1}))+1;
reluval_2h = sort(double(data0{5}))+1;



x = 1:180;
semilogy(x,facetv,x,reluval_2h,'LineWidth',1)
grid on
yticks([1 10 100,1000,4000])
ylim([1 4000])
legend('Our method','ReluVal')
xlabel('Number of instances verified')
ylabel('Time(sec)')
yticklabels({0,1,100,1000,4000})