import re
import os
import glob
import yaml
import sys


onednn_keys = [
    'addmm.out',
    '_addmm_activation.out',
    'mm.out',
    'mm',
    'baddbmm.out',
    'baddbmm_',
    'baddbmm',
    'addbmm.out',
    'addbmm_',
    'addbmm',
    'bmm.out',
    'bmm',
    'addmv.out',
    'tensordot.out',
    'convolution_overrideable',
    'convolution_backward_overrideable',
]
onednn_keys = set(onednn_keys)


def parse_keys(folder, backend, filename=None, startswith='m.impl("', pattern=r'm\.impl\("([^"]+)"', check=True):
    if filename is None:
        files = glob.glob(f'{folder}/**/Register{backend}.cpp', recursive=True)
        register_xxx_file = files[0]
    else:
        register_xxx_file = os.path.join(folder, filename)
    print(register_xxx_file)
    with open(register_xxx_file, 'r') as f:
        lines = f.readlines()
    if startswith is not None:
        lines = [l.strip() for l in lines if l.strip().startswith(startswith)]
    else:
        lines = [l.strip() for l in lines]
    # pattern = r'm\.impl\("([^"]+)"'
    keys = []
    for line in lines:
        match = re.search(pattern, line)
        if match is not None:
            keys.append(match.group(1))
    if check:
        assert len(lines) == len(keys)
    keys = set(keys)
    return keys


if __name__ == '__main__':
    root_folder = sys.argv[1].strip()
    cuda_keys = parse_keys(root_folder + '/build', 'CUDA')
    sparse_cuda_keys = parse_keys(root_folder + '/build', 'SparseCUDA')
    # sparse_csr_cuda_keys = parse_keys(root_folder + '/build', 'SparseCsrCUDA')
    sparse_csr_cuda_keys = set()
    num_of_all_cuda_keys = len(cuda_keys) + len(sparse_cuda_keys) + len(sparse_csr_cuda_keys)

    xpu_keys = parse_keys(root_folder + '/build/xpu', 'XPU')
    xpu_keys = xpu_keys & cuda_keys
    sparse_xpu_keys = parse_keys(root_folder, None, 
        'third_party/torch-xpu-ops/src/ATen/native/sparse/SparseTensor.cpp', startswith=None, pattern=r'TORCH_SELECTIVE_NAME\("([^"]+)"\)', check=False)
    # sparse_csr_xpu_keys = parse_keys(root_folder, None, 
    #     'third_party/torch-xpu-ops/src/ATen/native/sparse/SparseCsrTensor.cpp', startswith=None, pattern=r'TORCH_SELECTIVE_NAME\("([^"]+)"\)', check=False)
    sparse_csr_xpu_keys = set()
    num_of_all_xpu_keys = len(xpu_keys) + len(sparse_xpu_keys) + len(sparse_csr_xpu_keys)
    num_of_all_xpu_keys_w_onednn = num_of_all_xpu_keys + len(onednn_keys)

    with open('ipex_functions.yaml', 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
        data = data['supported']
    ipex_keys = set(data)
    ipex_keys = ipex_keys & cuda_keys

    with open('README.txt', 'w') as f:
        print('Description:\nThis is a temporary project for automating the progress report of torch-xpu-ops repo [https://github.com/intel/torch-xpu-ops].', file=f)
        print('Using command `python report.py $FOLDER_TO_PYTORCH` will generate report file named README.txt\n', file=f)

        print('Number of cuda-backend operators (with cudnn):', len(cuda_keys), file=f)
        print('Number of sparse_cuda-backend operators:', len(sparse_cuda_keys), file=f)
        # print('Number of sparse_csr_cuda-backend operators:', len(sparse_csr_cuda_keys), file=f)
        print('Total Number of cuda operators:', num_of_all_cuda_keys, file=f)
        print('', file=f)

        print('Number of xpu-backend operators (without onednn):', len(xpu_keys), file=f)
        print('Number of sparse_xpu-backend operators:', len(sparse_xpu_keys), file=f)
        # print('Number of sparse_csr_xpu-backend operators:', len(sparse_csr_xpu_keys), file=f)
        print('Number of onednn operators:', len(onednn_keys), file=f)
        print('Total Number of xpu operators:', num_of_all_xpu_keys_w_onednn, file=f)
        print('', file=f)

        print('Number of ipex operators (with onednn):', len(ipex_keys), file=f)
        print('', file=f)

        print('Ratio: xpu-ops (with onednn) / cuda:', num_of_all_xpu_keys_w_onednn / num_of_all_cuda_keys, file=f)
    
    sparse_cuda_keys = set(['sparse:'+item for item in sparse_cuda_keys])
    sparse_xpu_keys = set(['sparse:'+item for item in sparse_xpu_keys])
    all_cuda_keys = cuda_keys | sparse_cuda_keys
    all_xpu_keys = xpu_keys | sparse_xpu_keys | onednn_keys

    all_list = []
    for key in all_cuda_keys:
        if key in all_xpu_keys:
            all_list.append([key, 'y'])
        else:
            all_list.append([key, 'n'])
    # print(all_list)
    all_list = sorted(all_list)
    all_list.insert(0, ['op', 'xpu-ready'])
    import csv
    with open('output.csv', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(all_list)
