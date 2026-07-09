from setuptools import setup, find_packages

setup(
    name="transformer-summarizer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.1",
        "numpy>=1.24",
        "pandas>=2.0",
        "datasets>=2.14",
        "nltk>=3.8",
        "tqdm>=4.66",
        "plotly>=5.18",
        "streamlit>=1.30",
    ],
)
