with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(name='MuMoT',
    version='0.0',
    install_requires=requirements)
