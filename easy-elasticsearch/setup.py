from setuptools import setup, find_packages


with open("README.md", "r", encoding="utf-8") as fh:
    readme = fh.read()

setup(
    name="easy-elasticsearch",
    version="0.0.9",
    author="Kexin Wang",
    author_email="kexin.wang.2049@gmail.com",
    description="An easy-to-use Elasticsearch BM25 interface",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/kwang2049/easy-elasticsearch",
    project_urls={
        "Bug Tracker": "https://github.com/kwang2049/easy-elasticsearch/issues",
    },
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "elasticsearch>=7.9.1",  # BeIR requires es==7.9.1
        "tqdm",
        "requests",
    ],
)
