from setuptools import find_packages, setup

setup(
	name='GeoTorch',
	packages=find_packages(include=['geotorch']),
	version='0.1.0',
	description='GeoTorch: A Spatiotemporal Deep Learning Framework',
	author='Data Systems Lab, ASU',
	license='',
	install_requires=[],
	setup_requires=['pytest-runner'],
    tests_require=['pytest'],
	test_suite='tests',
	)