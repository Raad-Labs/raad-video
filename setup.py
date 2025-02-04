from setuptools import setup, find_packages

setup(
    name="raad-video",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "redis>=4.5.0",
        "pyzmq>=24.0.0",
        "requests>=2.26.0",
        "pytest>=7.0.0",
        "pytest-asyncio>=0.20.0",
        "pytest-benchmark>=4.0.0",
        "pytest-cov>=4.0.0",
    ],
    python_requires=">=3.8",
)
