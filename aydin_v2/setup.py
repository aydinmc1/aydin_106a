from glob import glob

from setuptools import find_packages, setup

package_name = "aydin_v2"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.py")),
        (f"share/{package_name}/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Your Name",
    maintainer_email="you@example.com",
    description="Computer vision nodes for detecting white patches in camera images.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "example_node = aydin_v2.example_node:main",
            "white_patch_detector_node_v2 = aydin_v2.white_patch_detector_node:main",
        ],
    },
)
