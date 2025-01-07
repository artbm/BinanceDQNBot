from setuptools import setup, find_packages
from pathlib import Path
import re

def get_version():
    """Extract version from __init__.py"""
    init = Path(__file__).parent / 'src' / 'trading_bot' / '__init__.py'
    with open(init, 'r', encoding='utf-8') as f:
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
        if version_match:
            return version_match.group(1)
    raise RuntimeError('Unable to find version string.')

def read_requirements():
    """Read requirements from requirements.txt"""
    with open('requirements.txt', 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read the README for the long description
long_description = (Path(__file__).parent / 'README.md').read_text(encoding='utf-8')

setup(
    name='trading-bot',
    version=get_version(),
    description='A reinforcement learning based cryptocurrency trading bot',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Your Name',
    author_email='arthur@bensimon.dev',
    url='https://github.com/artbm/BinanceDQNBot.git',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires='>=3.8',
    install_requires=read_requirements(),
    entry_points={
        'console_scripts': [
            'trading-bot=trading_bot.cli:main',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Financial and Insurance Industry',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.12',
        'Topic :: Office/Business :: Financial :: Investment',
    ],
    include_package_data=True,
    package_data={
        'trading_bot': ['config/*.yaml'],
    },
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-asyncio',
            'black',
            'mypy',
            'flake8',
        ],
    }
)
