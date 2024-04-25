import pathlib

from setuptools import setup, find_packages

pkg_dir = pathlib.Path(__file__).parent.resolve()
readme = (pkg_dir / "README.md").read_text(encoding="utf-8")
requirements = (pkg_dir / "requirements.txt").read_text(encoding="utf-8").splitlines()
requirements_server = (pkg_dir / "requirements-server.txt").read_text(encoding="utf-8").splitlines()

def read_version():
    with open(pkg_dir / "sillm" / "version.py") as fp:
        for line in fp:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"')
            
    raise ValueError("Version not found")

setup(
    name = "sillm-mlx",
    version = read_version(),
    description = "Running and training LLMs on Apple Silicon via MLX",
    long_description = readme,
    long_description_content_type = "text/markdown",
    readme="README.md",
    url = "https://github.com/armbues/SiLLM",
    install_requires=requirements,
    extras_require={
        "server": requirements_server
    },
    packages = find_packages(exclude="examples"),
    include_package_data=True,
    package_data = {
        'sillm': ['templates/*.jinja']
    },
    python_requires = ">=3.9"
)