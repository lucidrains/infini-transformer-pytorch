from setuptools import setup, find_packages

setup(
  name = 'infini-transformer-pytorch',
  packages = find_packages(exclude = []),
  version = '0.0.11',
  license='MIT',
  description = 'Infini-Transformer in Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/lucidrains/infini-transformer-pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'transformers',
    'attention mechanism',
    'long context',
    'memory'
  ],
  install_requires=[
    'einops>=0.8.0',
    'rotary-embedding-torch>=0.5.3',
    'torch>=2.0',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.7',
  ],
)
