import pathlib

from setuptools import setup, find_packages

pkg_dir = pathlib.Path(__file__).parent.resolve()
readme = (pkg_dir / "README.md").read_text(encoding="utf-8")
requirements = (pkg_dir / "requirements.txt").read_text(encoding="utf-8").splitlines()
requirements_server = (pkg_dir / "requirements-server.txt").read_text(encoding="utf-8").splitlines()

setup(
    name = "sillm",
    version = "0.1.0",
    description = "Running and training LLMs on Apple Silicon via MLX",
    long_description = readme,
    long_description_content_type = "text/markdown",
    readme="README.md",
    url = "https://github.com/armbues/SiLLM",
    install_requires=requirements,
    extras_require={
        "server": requirements_server
    },
    packages = find_packages(),
    python_requires = ">=3.8" # min for MLX
)