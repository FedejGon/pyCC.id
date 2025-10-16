rm -rf dist build *.egg-info  # clean old builds
python -m build
pip install dist/py*-any.whl
twine upload dist/py*-any.whl --verbose 

# useful commands:
# 1) modify py library and submitt to internet
# vi setup.py  # change version
# bash bash_install_and_submit.sh 
#
# 2) update installed library
#   a) on cpu/cluster and gpu nvidia
#   pip install --upgrade pycc.id
#   
#   b) on asus / intel irisxe 
#   conda activate
#   conda env list
#   conda activate pytorch_xpu
#   python3 -m pip install --upgrade pycc.id
#   python3 file.py
#
# 3) git 
# git status
# git add .
# git commit -m ' ' 
# git push


