Description:
This is a temporary project for automating the progress report of torch-xpu-ops repo [https://github.com/intel/torch-xpu-ops].
Using command `python report.py $FOLDER_TO_PYTORCH` will generate report file named README.txt

Number of cuda operators (with cudnn): 1225
Number of ipex operators (with onednn): 872
Number of xpu-ops operators (without onednn): 699
Number of onednn operators: 16
Total number of operators for xpu-ops+ipex+onednn (do intersection with cuda): 1037

Ratio: xpu-ops+onednn / cuda: 0.5763265306122449
