#!/bin/bash
python setup.py sdist bdist_wheel && python -m twine upload --repository pypi dist/*
rm -rf build dist *.egg-info

# API Token
# pypi-AgEIcHlwaS5vcmcCJGRmMmU1MjFkLWI3MjctNDYyNi05NGQxLWRhMmVmYWI1MTRkNAACJXsicGVybWlzc2lvbnMiOiAidXNlciIsICJ2ZXJzaW9uIjogMX0AAAYgClcuNxmGBqJLzgfuM9ImBNRZcUDfH8MCq3Kw86ohmrk