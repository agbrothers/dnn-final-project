from setuptools import find_packages, setup

setup(
    name="dnn-final-project",
    version="0.0.0",
    author="Greyson Brothers",
    author_email="greysonbrothers@gmail.com",
    description="Lumbar spine image classification",
    long_description="Final project repo for the JHU Deep Neural Network Class.",
    long_description_content_type="text/markdown",
    url="https://github.com/agbrothers",
    packages=[package for package in find_packages() if package.startswith("dnn-final-project")],
    zip_safe=False,
    # previous working versions in comments
    install_requires=[
        "six", 
        "bleach",
        "certifi",
        "requests",
        "tqdm",
        "urllib3",
        "kaggle", 
    ],
    project_urls={
        "Bug Tracker": "https://gitlab.jhuapl.edu/fordnp1/acessim_game/-/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.8",  # 3.8.13
)
