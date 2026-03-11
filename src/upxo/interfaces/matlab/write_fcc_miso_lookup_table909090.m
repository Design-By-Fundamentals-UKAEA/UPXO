STD_ORIS = [0, 0, 0; 0, 45, 0; 35, 45, 0;...
            59, 37, 63; 90, 35, 45; 0, 0, 45;...
            0, 35, 45; 90, 25, 45; 90, 74, 45]';
std_ori_names = {'w', 'g', 'b', 's', 'c', 'rw', 'rcu', 'gt', 'cut'};
ORI_table = array2table(STD_ORIS, 'VariableNames', std_ori_names);
folder = 'C:\Development\M2MatMod\upxo_packaged\upxo_private\src\upxo\_written_data\_orientation';
writetable(ORI_table, strcat(folder, filesep, 'STD_ORI_FCC_ROLLED.txt'), 'Delimiter',' ')

cs = crystalSymmetry('cubic');

W = orientation('euler', 0*degree , 0*degree , 0*degree , cs);
G = orientation('euler', 0*degree , 45*degree , 0*degree , cs);
B = orientation('euler', 35*degree , 45*degree , 0*degree , cs);
S = orientation('euler', 59*degree , 37*degree , 63*degree , cs);
C = orientation('euler', 90*degree , 35*degree , 45*degree , cs);
RW = orientation('euler', 0*degree , 0*degree , 45*degree , cs);
RCU = orientation('euler', 0*degree , 35*degree , 45*degree , cs);
GT = orientation('euler', 90*degree , 25*degree , 45*degree , cs);
CUT = orientation('euler', 90*degree , 74*degree , 45*degree , cs);

EA1_range = 0:1:90;
EA2_range = 0:1:90;
EA3_range = 0:1:90;

MIS_W = zeros(length(EA1_range), length(EA2_range), length(EA3_range));
MIS_G = zeros(length(EA1_range), length(EA2_range), length(EA3_range));
MIS_B = zeros(length(EA1_range), length(EA2_range), length(EA3_range));
MIS_S = zeros(length(EA1_range), length(EA2_range), length(EA3_range));
MIS_C = zeros(length(EA1_range), length(EA2_range), length(EA3_range));
MIS_RW = zeros(length(EA1_range), length(EA2_range), length(EA3_range));
MIS_RCU = zeros(length(EA1_range), length(EA2_range), length(EA3_range));
MIS_GT = zeros(length(EA1_range), length(EA2_range), length(EA3_range));
MIS_CUT = zeros(length(EA1_range), length(EA2_range), length(EA3_range));

[EA1, EA2, EA3] = meshgrid(EA1_range, EA2_range, EA3_range);

for i = 1:length(EA1_range)
    disp('=========================')
    disp(EA1_range(i))
    disp('................')
    for j = 1:length(EA2_range)
        disp(j)
        for k = 1:length(EA3_range)
            EA = orientation('euler', ...
                              EA1_range(i)*degree, ...
                              EA2_range(j)*degree, ...
                              EA3_range(k)*degree, CS);
            MIS_W(i, j, k) = angle(W, EA)./degree;
            MIS_G(i, j, k) = angle(G, EA)./degree;
            MIS_B(i, j, k) = angle(B, EA)./degree;
            MIS_S(i, j, k) = angle(S, EA)./degree;
            MIS_C(i, j, k) = angle(C, EA)./degree;
            MIS_RW(i, j, k) = angle(RW, EA)./degree;
            MIS_RCU(i, j, k) = angle(RCU, EA)./degree;
            MIS_GT(i, j, k) = angle(GT, EA)./degree;
            MIS_CUT(i, j, k) = angle(CUT, EA)./degree;
        end
    end
end

MIS = round([EA1(:), EA2(:), EA3(:),...
       MIS_W(:), MIS_G(:), MIS_B(:), ...
       MIS_S(:), MIS_C(:), MIS_RW(:), ...
       MIS_RCU(:), MIS_GT(:), MIS_CUT(:)], 4);
MIS(1:5, :)
cnames = {'ea1', 'ea2', 'ea3', 'w', 'g', 'b', 's', 'c', 'rw', 'rcu', 'gt', 'cut'};
MIS_table = array2table(MIS, 'VariableNames', cnames);

folder = 'C:\Development\M2MatMod\upxo_packaged\upxo_private\src\upxo\_written_data\_orientation';
writetable(MIS_table, strcat(folder, filesep, 'fcc_std_miso_90_90_90.txt'), 'Delimiter',' ')



size(cnames)

size(misorientation_angles)
misorientation_angles(1:10, 1:10, 15)

[i, j, k] = find(misorientation_angles >= 15);
length(i)
numel(misorientation_angles)
size(misorientation_angles)
