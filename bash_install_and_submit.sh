rm -rf dist build *.egg-info  # clean old builds
python -m build
pip install dist/py*-any.whl
twine upload dist/py*-any.whl --verbose 

