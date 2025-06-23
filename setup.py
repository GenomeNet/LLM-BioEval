from setuptools import setup, find_packages

setup(
    name="microbellm",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "tqdm",
        "colorama",
        "requests",
        "flask>=2.0.0",
        "flask-socketio>=5.0.0"
    ],
    entry_points={
        "console_scripts": [
            "microbellm=microbellm.microbellm:main",
            "microbellm-web=microbellm.web_app:main",
        ],
    },
    package_data={
        "microbellm": ["*.py", "*.txt"],
    },
    include_package_data=True,
    python_requires=">=3.8",
    description="Evaluate LLMs on microbial phenotype prediction tasks",
    author="MicrobeLLM Team",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)