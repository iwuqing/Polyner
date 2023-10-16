clear; clc

%% Load params
config_file = fullfile('config_dl.yaml');

config = helper.YAML.read(config_file);
CTpara = config.deep_lesion;
names = fieldnames(CTpara);
for ii=1:numel(names)
    name = names{ii};
    p = CTpara.(name);
    if ischar(p)
        CTpara.(name) = eval(p);
    end
end

%% Load meta data
load(fullfile('./metal/SampleMasks.mat'), 'CT_samples_bwMetal');
metal_masks = CT_samples_bwMetal;
MARpara = helper.get_mar_params('./metal');

mask_indices = CTpara.('mask_indices');

image_size = [CTpara.imPixNum, CTpara.imPixNum, numel(mask_indices)];
sinogram_size = [CTpara.sinogram_size_x, CTpara.sinogram_size_y, numel(mask_indices)];

% prepare metal masks
fprintf('Preparing metal masks...\n')
selected_metal = metal_masks(:, :, mask_indices);
temp_metal = single(zeros(CTpara.imPixNum, CTpara.imPixNum, 1));

num_sample = 10;

k = 1;
for i = 1:num_sample
    if k==11
        k = 1;
    end

    metal(:, :, 1) = uint8(imresize(selected_metal(:, :, k), [CTpara.imPixNum, CTpara.imPixNum], 'Method', 'bilinear'));
    
    fprintf('[%s][%d] Processing \n', 'test', i)

    raw_image = niftiread(['./slice/gt_', num2str(i-1), '.nii']);
    image = imresize(raw_image, [CTpara.imPixNum, CTpara.imPixNum], 'Method', 'bilinear');

    [ma_sinogram, LI_sinogram, ma_CT, LI_CT, gt_CT, gt_sinogram, ...
     metal_trace, mask, fanSensorPos] = helper.simulate_metal_artifact(image, metal, CTpara, MARpara);

    niftiwrite(mask, ['../input/mask_', num2str(i-1), '.nii']);
    niftiwrite(ma_CT, ['../input/ma_', num2str(i-1), '.nii']);
    niftiwrite(gt_CT, ['../input/gt_', num2str(i-1), '.nii']);
    niftiwrite(ma_sinogram, ['../input/ma_sinogram_', num2str(i-1), '.nii']);
    niftiwrite(fanSensorPos, '../input/fanSensorPos.nii');
    
    k = k + 1;
end



