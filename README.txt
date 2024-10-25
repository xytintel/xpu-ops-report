Description:
This is a temporary project for automating the progress report of torch-xpu-ops repo [https://github.com/intel/torch-xpu-ops].
Using command `python report.py $FOLDER_TO_PYTORCH` will generate report file named README.txt

Number of cuda-backend operators (with cudnn): 1225
Number of sparse_cuda-backend operators: 171
Total Number of cuda operators: 1396

Number of xpu-backend operators (without onednn): 927
Number of sparse_xpu-backend operators: 3
Number of onednn operators: 20
Total Number of xpu operators: 950

Number of ipex operators (with onednn): 815

Ratio: xpu-ops (with onednn) / cuda: 0.6805157593123209
