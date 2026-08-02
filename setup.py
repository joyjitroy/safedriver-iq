from setuptools import setup, find_packages

setup(
    name="VehicleSafetyResearch",
    version="0.1.0",
    description="Multi-Phase Research Framework for Proactive Vehicle Safety and VRU-Aware Risk Communication",
    author="Joyjit Roy",
    author_email="joyjit.roy.tech@gmail.com",
    packages=find_packages(where="phase1-safedriver-iq/src"),
    package_dir={"": "phase1-safedriver-iq/src"},
    py_modules=[
        "data_loader",
        "preprocessing", 
        "feature_engineering",
        "models",
        "visualization",
        "safety_score",
        "realtime_calculator",
        "scenario_simulator",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "shap>=0.42.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "jupyter>=1.0.0",
        ]
    },
)
