Install build and distribution dependencies:
`pip install build wheel twine`

Build source distributions and wheel:
`python setup.py bdist_wheel sdist`

Upload package to testpypi:
`python -m twine upload --repository testpypi dist/*`

Test installation from testpypi:
`python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple sillm`

Reset test environment:
`pip uninstall -y -r <(pip freeze)`