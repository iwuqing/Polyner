function [ma_sinogram_all, LI_sinogram_all, ma_CT_all, ...
          LI_CT_all, gt_CT, gt_sinogram_all, metal_trace_all, ...
          mask_all, fanSensorPos] = simulate_metal_artifact(imgCT, imgMetalList, CTpara, MARpara)

% If we want Python hdf5 matrix to have size (N x H x W), 
% Matlab matrix should have size (W x H x N) 
% Therefore, we can permute (H, W, N) to (W x H x N)

n_mask = size(imgMetalList, 3);

%% tissue composition
MiuWater = MARpara.MiuWater;
threshWater = MARpara.threshWater;
threshBone = MARpara.threshBone;

img = imgCT;
gt_CT = img;

imgWater = zeros(size(img));
imgBone = zeros(size(img));
bwWater = img <= threshWater;
bwBone = img >= threshBone;
bwBoth = im2bw(1 - bwWater - bwBone, 0.5);
imgWater(bwWater) = img(bwWater);
imgBone(bwBone) = img(bwBone);
imgBone(bwBoth) = (img(bwBoth) - threshWater) ./ (threshBone - threshWater) .* img(bwBoth);
imgWater(bwBoth) = img(bwBoth) - imgBone(bwBoth);

%% Metal
ma_sinogram_all = single(zeros(CTpara.sinogram_size_y, CTpara.sinogram_size_x, n_mask));
LI_sinogram_all = single(zeros(CTpara.sinogram_size_y, CTpara.sinogram_size_x, n_mask));
metal_trace_all = single(zeros(CTpara.sinogram_size_y, CTpara.sinogram_size_x, n_mask));
gt_sinogram_all = single(zeros(CTpara.sinogram_size_y, CTpara.sinogram_size_x, n_mask));
mask_all = single(zeros(CTpara.imPixNum, CTpara.imPixNum, n_mask));
ma_CT_all = single(zeros(CTpara.imPixNum, CTpara.imPixNum, n_mask));
LI_CT_all = single(zeros(CTpara.imPixNum, CTpara.imPixNum, n_mask));
fanSensorPos = single(zeros(CTpara.sinogram_size_y, 1, 1));
scatterPhoton = 20;

for i = 1:n_mask
    imgMetal_raw = squeeze(imgMetalList(:, :, i));
    imgMetal = imresize(imgMetal_raw, [CTpara.imPixNum, CTpara.imPixNum], 'Method', 'bilinear');
    bwMetal = im2bw(imgMetal);
    imgWater_local = imgWater;
    imgBone_local = imgBone;
    imgWater_local(bwMetal) = 0;
    imgBone_local(bwMetal) = 0;

    %% Synthesize non-metal poly CT
    Pwater_kev = fanbeam(imgWater_local, CTpara.SOD,...
            'FanSensorGeometry', 'arc',...
            'FanSensorSpacing', CTpara.angSize, ...
            'FanRotationIncrement', 360/CTpara.angNum);
    Pwater_kev = Pwater_kev .* CTpara.imPixScale;
    
    Pbone_kev = fanbeam(imgBone_local, CTpara.SOD,...
            'FanSensorGeometry', 'arc',...
            'FanSensorSpacing', CTpara.angSize, ...
            'FanRotationIncrement', 360/CTpara.angNum);
    Pbone_kev = Pbone_kev .* CTpara.imPixScale;


    Pmetal_kev = fanbeam(imgMetal, CTpara.SOD,...
            'FanSensorGeometry','arc',...
            'FanSensorSpacing', CTpara.angSize, ...
            'FanRotationIncrement',360/CTpara.angNum);
    metal_trace = Pmetal_kev > 0;
    Pmetal_kev = Pmetal_kev .* CTpara.imPixScale;
    Pmetal_kev = MARpara.metalAtten * Pmetal_kev;
    % partial volume effect
    Pmetal_kev_bw =imerode(Pmetal_kev>0, [1 1 1]');
    Pmetal_edge = xor((Pmetal_kev>0), Pmetal_kev_bw);
    Pmetal_kev(Pmetal_edge) = Pmetal_kev(Pmetal_edge) / 4;

    % sinogram with metal
    projkevAllLocal(:, :, 1) = Pwater_kev;
    projkevAllLocal(:, :, 2) = Pbone_kev;
    projkevAllLocal(:, :, 3) = Pmetal_kev;

    poly_img = single(zeros(CTpara.imPixNum, CTpara.imPixNum, 101));
    temp_index = 1;
    % 合成poly image
    for ien = MARpara.energies
        ploy_water = MARpara.MiuAll(ien, 7, 1)/MARpara.MiuAll(MARpara.kev, 7, 1)*imgWater_local;
        ploy_bone = MARpara.MiuAll(ien, 7, 2)/MARpara.MiuAll(MARpara.kev, 7, 2)*imgBone_local;
        ploy_metal = MARpara.MiuAll(ien, 7, 3)/MARpara.MiuAll(MARpara.kev, 7, 3)*MARpara.metalAtten*imgMetal;
        poly_img(:, :, temp_index) = double(ploy_water) + double(ploy_bone) + double(ploy_metal);
        temp_index = temp_index + 1;
    end

    [projkvpMetal] = helper.pkev2kvp(projkevAllLocal, MARpara.spectrum, MARpara.energies, MARpara.kev, MARpara.MiuAll);
    temp = round(exp(-projkvpMetal) .* MARpara.photonNum);
    temp = temp + scatterPhoton;
    ProjPhoton = poissrnd(temp);
    ProjPhoton(ProjPhoton == 0) = 1;
    projkvpMetalNoise = -log(ProjPhoton ./ MARpara.photonNum);
    size(projkvpMetalNoise);
    % correction
    p1 = reshape(projkvpMetalNoise, CTpara.sinogram_size_y*CTpara.sinogram_size_x, 1);
    p1BHC = [p1  p1.^2  p1.^3] * MARpara.paraBHC;
    ma_sinogram = reshape(p1BHC, CTpara.sinogram_size_y, CTpara.sinogram_size_x);
    LI_sinogram = helper.interpolate_projection(ma_sinogram, metal_trace);

    % reconstruct   
    ma_CT = ifanbeam(ma_sinogram, CTpara.SOD,...
            'FanSensorGeometry', 'arc',...
            'FanSensorSpacing', CTpara.angSize,...
            'OutputSize', CTpara.imPixNum,...
            'FanRotationIncrement', 360 / CTpara.angNum);
    ma_CT = ma_CT ./ CTpara.imPixScale;
    
    LI_CT = ifanbeam(LI_sinogram, CTpara.SOD,...
            'FanSensorGeometry', 'arc',...
            'FanSensorSpacing', CTpara.angSize,...
            'OutputSize', CTpara.imPixNum,...
            'FanRotationIncrement', 360 / CTpara.angNum);
    LI_CT = LI_CT ./ CTpara.imPixScale;
    
    % gt sinogram
    [gt_sinogram, Pos, ~] = fanbeam(gt_CT, CTpara.SOD,...
                  'FanSensorGeometry', 'arc',...
                  'FanSensorSpacing', CTpara.angSize, ...
                  'FanRotationIncrement', 360/CTpara.angNum);

    gt_sinogram = gt_sinogram .* CTpara.imPixScale;
    fanSensorPos(:, :, 1) = Pos;
    gt_sinogram_all(:, :, i) = gt_sinogram;
    ma_sinogram_all(:, :, i) = ma_sinogram;
    LI_sinogram_all(:, :, i) = LI_sinogram;
    metal_trace_all(:, :, i) = metal_trace;
    ma_CT_all(:, :, i) = ma_CT;
    LI_CT_all(:, :, i) = LI_CT;
    SE = strel('square',3);
    mask_all(:, :, i) = imdilate(imgMetal, SE)>0;
    end
end