import re
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


def parse_keys(folder, backend):
    files = glob.glob(f'{folder}/**/Register{backend}.cpp', recursive=True)
    register_xxx_file = files[0]
    print(register_xxx_file)
    with open(register_xxx_file, 'r') as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines if l.strip().startswith('m.impl("')]
    pattern = r'm\.impl\("([^"]+)"'
    keys = []
    for line in lines:
        match = re.search(pattern, line)
        keys.append(match.group(1))
    assert len(lines) == len(keys)
    keys = set(keys)
    return keys


if __name__ == '__main__':
    root_folder = sys.argv[1].strip()
    cuda_keys = parse_keys(root_folder + '/build', 'CUDA')
    xpu_keys = parse_keys(root_folder + '/build/xpu', 'XPU')
    with open('ipex_functions.yaml', 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
        data = data['supported']
    ipex_keys = set(data)

    ipex_and_xpu_keys = ipex_keys | xpu_keys
    ipex_and_xpu_keys_in_cuda = ipex_and_xpu_keys & cuda_keys
    all_xpu_keys = ipex_keys | xpu_keys | onednn_keys
    all_xpu_keys_in_cuda = all_xpu_keys & cuda_keys

    with open('README.txt', 'w') as f:
        print('Description:\nThis is a temporary project for automating the progress report of torch-xpu-ops repo [https://github.com/intel/torch-xpu-ops].', file=f)
        print('Using command `python report.py $FOLDER_TO_PYTORCH` will generate report file named README.txt\n', file=f)

        print('Number of cuda operators (with cudnn):', len(cuda_keys), file=f)
        print('Number of ipex operators (without onednn):', len(ipex_keys), file=f)
        print('Number of xpu-ops operators (without onednn):', len(xpu_keys), file=f)
        print('Number of onednn operators:', len(onednn_keys), file=f)

        print('Total number of operators for xpu-ops+ipex (do intersection with cuda):', len(ipex_and_xpu_keys_in_cuda), file=f)
        print('Total number of operators for xpu-ops+ipex+onednn (do intersection with cuda):', len(all_xpu_keys_in_cuda), file=f)
        print('', file=f)

        print('Ratio: xpu-ops / cuda:', len(xpu_keys) / len(cuda_keys), file=f)
        print('Ratio: ipex / cuda:', len(ipex_keys) / len(cuda_keys), file=f)
        print('Ratio: xpu-ops+ipex (do intersection with cuda) / cuda:', len(ipex_and_xpu_keys_in_cuda) / len(cuda_keys), file=f)
        print('Ratio: xpu-ops+ipex+onednn (do intersection with cuda) / cuda:', len(all_xpu_keys_in_cuda) / len(cuda_keys), file=f)
