from setuptools import setup, find_packages

setup(
    name='s2m',
    version='0.1.0',    
    description='A package to train a Diffusion Model fro Sketch 2 Motion Generation',
    url='https://github.com/JanErikHuehne/HumanMotionGeneration',
    author='Jan-Erik Hühne',
    author_email='jan.huehne@tum.de',
    license='MIT',
    py_modules=['data_loaders', 'model'],
    packages=find_packages(),
    install_requires=[],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
      
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
    ],
)