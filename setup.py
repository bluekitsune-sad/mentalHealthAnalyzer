from setuptools import setup, find_packages
from typing import List

def get_requrements(file_Path:str)->List[str]:
    '''
    this function is used to get the requrements 
    from the file_Path
    '''
    requirements=[]
    with open(file_Path)as file_obj:
        requirements=file_obj.readlines()
        [req.replace('\n','') for req in requirements]
        if "-e ." in requirements:
            requirements.remove("-e .")
    return requirements

setup(
    name='mentalHealthAnalyzer',
    version='0.0.1',
    author='Saad Yousuf',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        get_requrements('requirements.txt')
    ]
)