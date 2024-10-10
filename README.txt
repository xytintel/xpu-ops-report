Description:
This is a temporary project for automating the progress report of torch-xpu-ops repo [https://github.com/intel/torch-xpu-ops].
Using command `python report.py $FOLDER_TO_PYTORCH` will generate report file named README.txt

Number of cuda operators (with cudnn): 1225
Number of ipex operators (without onednn): 872
Number of xpu-ops operators (without onednn): 699
Number of onednn operators: 16
Total number of operators for xpu-ops+ipex (do intersection with cuda): 1037
Total number of operators for xpu-ops+ipex+onednn (do intersection with cuda): 1037

Ratio: xpu-ops / cuda: 0.5706122448979591
Ratio: ipex / cuda: 0.7118367346938775
Ratio: xpu-ops+ipex (do intersection with cuda) / cuda: 0.846530612244898
Ratio: xpu-ops+ipex+onednn (do intersection with cuda) / cuda: 0.846530612244898
