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
    return keys


if __name__ == '__main__':
    root_folder = sys.argv[1].strip()
    cuda_keys = parse_keys(root_folder + '/build', 'CUDA')
    xpu_keys = parse_keys(root_folder + '/build/xpu', 'XPU')
    with open('ipex_functions.yaml', 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
        data = data['supported']
    ipex_keys = data

    cuda_keys = set(cuda_keys)
    ipex_keys = set(ipex_keys)
    xpu_keys = set(xpu_keys)
    ipex_and_xpu_keys = ipex_keys | xpu_keys
    ipex_and_xpu_keys_in_cuda = ipex_and_xpu_keys & cuda_keys


    all_xpu_keys = ipex_keys | xpu_keys | onednn_keys

    with open('README.md', 'w') as f:
        print('Number of cuda operators:', len(cuda_keys), file=f)
        print('Number of ipex operators:', len(ipex_keys), file=f)
        print('Number of xpu-ops operators:', len(xpu_keys), file=f)
        print('Total number of operators for xpu-ops + ipex:', len(ipex_and_xpu_keys_in_cuda), file=f)
        print('Total number of operators for xpu-ops + ipex + onednn:', len(all_xpu_keys), file=f)
        print('', file=f)
        print('Ratio: xpu-ops / cuda:', len(xpu_keys) / len(cuda_keys), file=f)
        print('Ratio: ipex / cuda:', len(ipex_keys) / len(cuda_keys), file=f)
        print('Ratio: xpu-ops + ipex / cuda:', len(ipex_and_xpu_keys_in_cuda) / len(cuda_keys), file=f)
        print('Ratio: xpu-ops + ipex + onednn / cuda:', len(all_xpu_keys) / len(cuda_keys), file=f)
