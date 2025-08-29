#!/usr/bin/env python3
"""
Setup script for Hybrid Search Engine.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="hybrid-search-engine",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Production-ready hybrid search engine with BM25 + vector similarity",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/hybrid-search-engine",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Indexing",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-asyncio>=0.21.1",
            "black>=23.11.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "hybrid-search=cli_search:search",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)