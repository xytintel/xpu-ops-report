Description:
This is a temporary project for automating the progress report of torch-xpu-ops repo [https://github.com/intel/torch-xpu-ops].
Using command `python report.py $FOLDER_TO_PYTORCH` will generate report file named README.txt

Number of cuda-backend operators (with cudnn): 1225
Number of sparse_cuda-backend operators: 171
Number of sparse_csr_cuda-backend operators: 139
Total Number of cuda operators: 1535

Number of xpu-backend operators (without onednn): 697
Number of sparse_xpu-backend operators: 3
Number of sparse_csr_xpu-backend operators: 1
Number of onednn operators: 16
Total Number of xpu operators: 717

Number of ipex operators (with onednn): 815

Ratio: xpu-ops (with onednn) / cuda: 0.4671009771986971
