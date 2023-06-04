from typing import List
from setuptools import find_packages,setup

HYPEN_E_DOT = '-e .'

def get_requirements(file_path : str )->List[str]:
    '''
    This function will return list of requirements.
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n','') for req in requirements]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
        return requirements

setup(
    name = 'Student Performance Indicator ML Project',
    version='0.0.1',
    description='End to End Machine Learning project on Student Performance which will be running in Azure cloud',
    author='Santosh',
    author_email='ssantoshpro@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')

)