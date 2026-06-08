from setuptools import setup, find_packages

setup(
    name="rl-ftc-quadrotor",
    version="0.1.0",
    description="Reinforcement Learning Based Fault-Tolerant Control of a Quadrotor",
    author="Your Name",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "stable-baselines3>=2.0.0",
        "gymnasium>=0.29.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "ruff",
            "black",
            "isort",
            "mypy",
            "types-PyYAML",
        ]
    },
)
